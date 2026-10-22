//! SWE-bench datasets: a Hugging Face dataset read through the datasets server's `/rows`
//! endpoint (upstream uses the `datasets` library), or a local JSON or JSON Lines file.

use std::path::Path;

use serde_json::Value as Json;

/// Upstream's subset names.
pub const DATASET_MAPPING: [(&str, &str); 8] = [
    ("full", "princeton-nlp/SWE-Bench"),
    ("verified", "princeton-nlp/SWE-Bench_Verified"),
    ("lite", "princeton-nlp/SWE-Bench_Lite"),
    ("multimodal", "princeton-nlp/SWE-Bench_Multimodal"),
    ("multilingual", "swe-bench/SWE-Bench_Multilingual"),
    ("smith", "SWE-bench/SWE-smith"),
    ("_test", "klieret/swe-bench-dummy-test-dataset"),
    ("rebench", "nebius/SWE-rebench"),
];

/// The datasets server, unless `MSWEA_DATASETS_SERVER_URL` names another.
pub const DATASETS_SERVER: &str = "https://datasets-server.huggingface.co";

/// Rows per request; the server's maximum.
const PAGE: usize = 100;

#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("Could not read {path}: {message}")]
    File { path: String, message: String },
    #[error("Could not load {dataset} from the datasets server: {message}")]
    Server { dataset: String, message: String },
    #[error("{dataset} has no split {split} (splits: {available})")]
    NoSplit {
        dataset: String,
        split: String,
        available: String,
    },
}

/// The dataset `subset` names (a subset name, a Hugging Face dataset, or a local file), its
/// `split`. `server` overrides the datasets server.
pub async fn load_dataset(subset: &str, split: &str, server: Option<&str>) -> Result<Vec<Json>, DatasetError> {
    let path = Path::new(subset);
    if path.is_file() {
        return load_file(path);
    }
    let dataset = DATASET_MAPPING
        .iter()
        .find(|(name, _)| *name == subset)
        .map_or(subset, |(_, dataset)| dataset);
    let server = server
        .map(str::to_owned)
        .or_else(|| crate::global::env_var("MSWEA_DATASETS_SERVER_URL"))
        .unwrap_or_else(|| DATASETS_SERVER.to_owned());
    load_from_server(server.trim_end_matches('/'), dataset, split).await
}

/// Instances from a file: a JSON array (or an object holding one under `instances` or
/// `rows`), or one JSON object per line.
pub fn load_file(path: &Path) -> Result<Vec<Json>, DatasetError> {
    let error = |message: String| DatasetError::File {
        path: path.display().to_string(),
        message,
    };
    let text = std::fs::read_to_string(path).map_err(|e| error(e.to_string()))?;
    if path.extension().is_some_and(|extension| extension == "jsonl") {
        return text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).map_err(|e| error(e.to_string())))
            .collect();
    }
    match serde_json::from_str::<Json>(&text).map_err(|e| error(e.to_string()))? {
        Json::Array(instances) => Ok(instances),
        Json::Object(mut object) => match object.remove("instances").or_else(|| object.remove("rows")) {
            Some(Json::Array(instances)) => Ok(instances),
            _ => Err(error("expected a list of instances".to_owned())),
        },
        _ => Err(error("expected a list of instances".to_owned())),
    }
}

async fn load_from_server(server: &str, dataset: &str, split: &str) -> Result<Vec<Json>, DatasetError> {
    let error = |message: String| DatasetError::Server {
        dataset: dataset.to_owned(),
        message,
    };
    let client = reqwest::Client::new();
    let token = crate::global::env_var("HF_TOKEN");
    let get = |url: String, query: Vec<(&'static str, String)>| {
        let url = reqwest::Url::parse_with_params(&url, &query);
        let client = client.clone();
        let token = token.clone();
        async move {
            let mut request = client.get(url.map_err(|e| e.to_string())?);
            if let Some(token) = &token {
                request = request.bearer_auth(token);
            }
            let response = request.send().await.map_err(|e| e.to_string())?;
            let status = response.status();
            let body = response.text().await.map_err(|e| e.to_string())?;
            if !status.is_success() {
                return Err(format!("{status}: {body}"));
            }
            serde_json::from_str::<Json>(&body).map_err(|e| e.to_string())
        }
    };

    // The config that holds the split.
    let splits = get(format!("{server}/splits"), vec![("dataset", dataset.to_owned())])
        .await
        .map_err(error)?;
    let entries = splits["splits"].as_array().cloned().unwrap_or_default();
    let config = entries
        .iter()
        .find(|entry| entry["split"] == split)
        .and_then(|entry| entry["config"].as_str())
        .ok_or_else(|| DatasetError::NoSplit {
            dataset: dataset.to_owned(),
            split: split.to_owned(),
            available: entries
                .iter()
                .filter_map(|entry| entry["split"].as_str())
                .collect::<Vec<_>>()
                .join(", "),
        })?
        .to_owned();

    let mut rows = Vec::new();
    loop {
        let page = get(
            format!("{server}/rows"),
            vec![
                ("dataset", dataset.to_owned()),
                ("config", config.clone()),
                ("split", split.to_owned()),
                ("offset", rows.len().to_string()),
                ("length", PAGE.to_string()),
            ],
        )
        .await
        .map_err(error)?;
        let page_rows = page["rows"].as_array().cloned().unwrap_or_default();
        if page_rows
            .iter()
            .any(|row| row["truncated_cells"].as_array().is_some_and(|cells| !cells.is_empty()))
        {
            return Err(error(
                "the server truncated some cells; download the dataset and pass its file".to_owned(),
            ));
        }
        let fetched = page_rows.len();
        rows.extend(page_rows.into_iter().map(|row| row["row"].clone()));
        let total = page["num_rows_total"]
            .as_u64()
            .map_or(usize::MAX, |total| total as usize);
        if fetched == 0 || rows.len() >= total {
            break;
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    #[tokio::test]
    async fn rows_come_page_by_page_from_the_split_s_config() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/splits"))
            .and(query_param("dataset", "princeton-nlp/SWE-Bench_Lite"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "splits": [
                    { "dataset": "princeton-nlp/SWE-Bench_Lite", "config": "default", "split": "dev" },
                    { "dataset": "princeton-nlp/SWE-Bench_Lite", "config": "default", "split": "test" },
                ]
            })))
            .mount(&server)
            .await;
        let rows = |range: std::ops::Range<usize>| -> Vec<Json> {
            range
                .map(|index| json!({ "row_idx": index, "row": { "instance_id": format!("i__{index}") }, "truncated_cells": [] }))
                .collect()
        };
        Mock::given(method("GET"))
            .and(path("/rows"))
            .and(query_param("config", "default"))
            .and(query_param("split", "dev"))
            .and(query_param("offset", "0"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({ "rows": rows(0..100), "num_rows_total": 130 })),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/rows"))
            .and(query_param("offset", "100"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({ "rows": rows(100..130), "num_rows_total": 130 })),
            )
            .mount(&server)
            .await;
        let instances = load_dataset("lite", "dev", Some(&server.uri())).await.unwrap();
        assert_eq!(instances.len(), 130);
        assert_eq!(instances[129]["instance_id"], "i__129");
        assert!(matches!(
            load_dataset("lite", "train", Some(&server.uri())).await,
            Err(DatasetError::NoSplit { .. })
        ));
    }

    #[test]
    fn files_hold_lists_or_lines() {
        let dir = tempfile::tempdir().unwrap();
        let list = dir.path().join("instances.json");
        std::fs::write(&list, json!([{ "instance_id": "a" }]).to_string()).unwrap();
        assert_eq!(load_file(&list).unwrap(), [json!({ "instance_id": "a" })]);
        let lines = dir.path().join("instances.jsonl");
        std::fs::write(&lines, "{\"instance_id\": \"a\"}\n\n{\"instance_id\": \"b\"}\n").unwrap();
        assert_eq!(load_file(&lines).unwrap().len(), 2);
    }
}
