//! SWE-bench: the batch runner (`mini-extra swebench`) and the single-instance runner
//! (`mini-extra swebench-single`), as upstream's `run/benchmarks/swebench*.py`.

mod dataset;
mod progress;
mod pyrandom;
mod run;
mod single;

use std::sync::Arc;

use serde_json::{Map, Value as Json};

pub use dataset::{DATASET_MAPPING, DATASETS_SERVER, DatasetError, load_dataset, load_file};
pub use progress::BatchProgress;
pub use pyrandom::PyRandom;
pub use run::{SwebenchArgs, preds_remove, preds_update, run_batch};
pub use single::{SingleArgs, run_single, select_instance};

use crate::env::{Environment, get_environment};

/// The Docker image of an instance: its `image_name` or `docker_image`, else SWE-bench's
/// image for its id (`__` becomes `_1776_`, which Docker allows).
pub fn docker_image_name(instance: &Json) -> String {
    if let Some(image) = ["image_name", "docker_image"]
        .iter()
        .find_map(|key| instance[*key].as_str().filter(|image| !image.is_empty()))
    {
        return image.to_owned();
    }
    let id = instance["instance_id"].as_str().unwrap_or_default();
    format!(
        "docker.io/swebench/sweb.eval.x86_64.{}:latest",
        id.replace("__", "_1776_")
    )
    .to_lowercase()
}

/// Filters, slices and shuffles instances as upstream's `filter_instances`: shuffling sorts by
/// id and shuffles with Python's `random.seed(42)`, the filter is a regular expression matched
/// at the start of the id, the slice is Python's `start:stop:step`.
pub fn filter_instances(instances: Vec<Json>, filter: &str, slice: &str, shuffle: bool) -> Result<Vec<Json>, String> {
    let mut instances = instances;
    if shuffle {
        instances.sort_by(|a, b| a["instance_id"].as_str().cmp(&b["instance_id"].as_str()));
        PyRandom::new(42).shuffle(&mut instances);
    }
    let pattern = regex::Regex::new(&format!("^(?:{filter})")).map_err(|error| format!("Invalid --filter: {error}"))?;
    instances.retain(|instance| pattern.is_match(instance["instance_id"].as_str().unwrap_or_default()));
    if !slice.is_empty() {
        instances = python_slice(instances, slice)?;
    }
    Ok(instances)
}

/// `items[start:stop:step]` with Python's semantics (negative indices count from the end).
pub fn python_slice<T>(items: Vec<T>, spec: &str) -> Result<Vec<T>, String> {
    let parts: Vec<Option<i64>> = spec
        .split(':')
        .map(|part| {
            let part = part.trim();
            if part.is_empty() {
                Ok(None)
            } else {
                part.parse().map(Some).map_err(|_| format!("Invalid --slice: {spec}"))
            }
        })
        .collect::<Result<_, _>>()?;
    let (start, stop, step) = match parts.as_slice() {
        [stop] => (None, *stop, None),
        [start, stop] => (*start, *stop, None),
        [start, stop, step] => (*start, *stop, *step),
        _ => return Err(format!("Invalid --slice: {spec}")),
    };
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err("Invalid --slice: the step cannot be zero".to_owned());
    }
    let len = items.len() as i64;
    let clamp = |index: i64, low: i64, high: i64| {
        let index = if index < 0 { index + len } else { index };
        index.clamp(low, high)
    };
    let indices: Vec<i64> = if step > 0 {
        let start = start.map_or(0, |start| clamp(start, 0, len));
        let stop = stop.map_or(len, |stop| clamp(stop, 0, len));
        (start..stop).step_by(step as usize).collect()
    } else {
        let start = start.map_or(len - 1, |start| clamp(start, -1, len - 1));
        let stop = stop.map_or(-1, |stop| clamp(stop, -1, len - 1));
        let mut indices = Vec::new();
        let mut index = start;
        while index > stop {
            indices.push(index);
            index += step;
        }
        indices
    };
    let mut slots: Vec<Option<T>> = items.into_iter().map(Some).collect();
    Ok(indices
        .into_iter()
        .filter_map(|index| slots.get_mut(index as usize).and_then(Option::take))
        .collect())
}

/// The environment of an instance (upstream's `get_sb_environment`): the config's
/// `environment` with class `docker` by default and the instance's image, then
/// `run.env_startup_command` rendered with the instance.
pub async fn instance_environment(config: &Json, instance: &Json) -> Result<Arc<dyn Environment>, String> {
    let mut env_config = match &config["environment"] {
        Json::Object(object) => object.clone(),
        _ => Map::new(),
    };
    let class = env_config
        .get("environment_class")
        .and_then(Json::as_str)
        .unwrap_or("docker")
        .to_owned();
    env_config.insert("environment_class".to_owned(), Json::String(class.clone()));
    if crate::env::environment_kind(&class).map_err(|error| error.to_string())? == "docker" {
        env_config.insert("image".to_owned(), Json::String(docker_image_name(instance)));
    }
    let env = get_environment(&Json::Object(env_config), "docker")
        .await
        .map_err(|error| error.to_string())?;
    if let Some(command) = config["run"]["env_startup_command"].as_str() {
        let command = crate::template::render(command, instance).map_err(|error| error.to_string())?;
        let ctx = delta_chord::context::Context::background();
        let output = env.execute(&command, None, None, &ctx).await;
        if output.returncode != 0 {
            env.cleanup().await;
            return Err(format!("Error executing startup command: {}", output.to_json()));
        }
    }
    Ok(env)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn image_names_follow_swebench() {
        assert_eq!(
            docker_image_name(&json!({ "instance_id": "astropy__astropy-12907" })),
            "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
        );
        assert_eq!(
            docker_image_name(&json!({ "instance_id": "X__Y-1", "image_name": "custom:tag" })),
            "custom:tag"
        );
    }

    #[test]
    fn slices_follow_python() {
        let items: Vec<i32> = (0..10).collect();
        assert_eq!(python_slice(items.clone(), "0:5").unwrap(), [0, 1, 2, 3, 4]);
        assert_eq!(python_slice(items.clone(), "-3:").unwrap(), [7, 8, 9]);
        assert_eq!(python_slice(items.clone(), "1:8:3").unwrap(), [1, 4, 7]);
        assert_eq!(python_slice(items.clone(), "::-4").unwrap(), [9, 5, 1]);
        assert_eq!(python_slice(items.clone(), ":100").unwrap().len(), 10);
        assert!(python_slice(items, "a:b").is_err());
    }

    #[test]
    fn filtering_matches_upstream() {
        let instances: Vec<Json> = (0..10)
            .map(|index| json!({ "instance_id": format!("repo__{index}") }))
            .collect();
        let ids = |instances: Vec<Json>| -> Vec<String> {
            instances
                .iter()
                .map(|instance| instance["instance_id"].as_str().unwrap().to_owned())
                .collect()
        };
        // Python: x = sorted ids; random.seed(42); random.shuffle(x).
        let shuffled = filter_instances(instances.clone(), "", "", true).unwrap();
        let expected: Vec<String> = [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]
            .iter()
            .map(|index| format!("repo__{index}"))
            .collect();
        assert_eq!(ids(shuffled), expected);
        assert_eq!(
            ids(filter_instances(instances.clone(), "repo__[12]", "", false).unwrap()),
            ["repo__1", "repo__2"]
        );
        assert_eq!(
            ids(filter_instances(instances, "", "2:4", false).unwrap()),
            ["repo__2", "repo__3"]
        );
    }
}
