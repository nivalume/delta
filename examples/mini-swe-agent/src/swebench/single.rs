//! `mini-extra swebench-single`: one instance with the interactive agent (upstream's
//! `run/benchmarks/swebench_single.py`).

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use serde_json::{Map, Value as Json};

use super::{instance_environment, load_dataset};
use crate::agent::{AgentKind, Frontend, Terminal};
use crate::cli::{BuildOptions, build_agent};
use crate::config::{self, Overrides};

/// Run on a single SWE-bench instance.
#[derive(Clone, Debug, clap::Args)]
pub struct SingleArgs {
    /// SWE-bench subset to use or path to a dataset.
    #[arg(long, default_value = "lite")]
    pub subset: String,
    /// Dataset split.
    #[arg(long, default_value = "dev")]
    pub split: String,
    /// SWE-bench instance ID or index (into the sorted IDs).
    #[arg(short = 'i', long = "instance", default_value = "0")]
    pub instance: String,
    /// Model to use.
    #[arg(short = 'm', long = "model")]
    pub model: Option<String>,
    /// Model class to use (e.g., 'litellm' or 'litellm_textbased').
    #[arg(long = "model-class")]
    pub model_class: Option<String>,
    /// Agent class to use (e.g., 'interactive' or 'default').
    #[arg(long = "agent-class")]
    pub agent_class: Option<String>,
    /// Environment class to use (`docker`).
    #[arg(long = "environment-class")]
    pub environment_class: Option<String>,
    /// Run without confirmation.
    #[arg(short = 'y', long = "yolo")]
    pub yolo: bool,
    /// Cost limit. Set to 0 to disable.
    #[arg(short = 'l', long = "cost-limit")]
    pub cost_limit: Option<f64>,
    /// Path to config files, filenames, or key-value pairs; default `swebench.yaml`.
    #[arg(short = 'c', long = "config")]
    pub config: Vec<String>,
    /// Exit immediately when the agent wants to finish instead of prompting.
    #[arg(long = "exit-immediately")]
    pub exit_immediately: bool,
    /// Output trajectory file.
    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,
    /// The Hugging Face datasets server.
    #[arg(long = "datasets-server", hide = true)]
    pub datasets_server: Option<String>,
}

/// The instance `spec` names: an id, or an index into the sorted ids.
pub fn select_instance(instances: Vec<Json>, spec: &str) -> Result<Json, String> {
    let mut by_id: Vec<(String, Json)> = instances
        .into_iter()
        .map(|instance| {
            (
                instance["instance_id"].as_str().unwrap_or_default().to_owned(),
                instance,
            )
        })
        .collect();
    by_id.sort_by(|a, b| a.0.cmp(&b.0));
    if !spec.is_empty() && spec.chars().all(|character| character.is_ascii_digit()) {
        let index: usize = spec.parse().map_err(|_| format!("Invalid instance index {spec}"))?;
        let count = by_id.len();
        return by_id
            .into_iter()
            .nth(index)
            .map(|(_, instance)| instance)
            .ok_or_else(|| format!("Instance index {index} is out of range ({count} instances)"));
    }
    by_id
        .into_iter()
        .find(|(id, _)| id == spec)
        .map(|(_, instance)| instance)
        .ok_or_else(|| format!("No instance {spec}"))
}

/// Runs `swebench-single`.
pub async fn run_single(args: SingleArgs) -> ExitCode {
    let fail = |message: String| {
        eprintln!("{message}");
        ExitCode::FAILURE
    };
    eprintln!("Loading dataset from {}, split {}...", args.subset, args.split);
    let instances = match load_dataset(&args.subset, &args.split, args.datasets_server.as_deref()).await {
        Ok(instances) => instances,
        Err(error) => return fail(error.to_string()),
    };
    let instance = match select_instance(instances, &args.instance) {
        Ok(instance) => instance,
        Err(error) => return fail(error),
    };
    let specs = if args.config.is_empty() {
        vec!["swebench.yaml".to_owned()]
    } else {
        args.config.clone()
    };
    eprintln!("Building agent config from specs: {specs:?}");
    let mut config = match config::load(&specs) {
        Ok(config) => config,
        Err(error) => return fail(error.to_string()),
    };
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| crate::global::global_config_dir().join("last_swebench_single_run.traj.json"));
    Overrides::default()
        .set("agent.agent_class", args.agent_class.clone())
        .set("agent.mode", args.yolo.then_some("yolo"))
        .set("agent.cost_limit", args.cost_limit)
        .set("agent.confirm_exit", args.exit_immediately.then_some(false))
        .set("agent.output_path", Some(output.display().to_string()))
        .set("model.model_class", args.model_class.clone())
        .set("model.model_name", args.model.clone())
        .set("environment.environment_class", args.environment_class.clone())
        .apply(&mut config);
    let env = match instance_environment(&config, &instance).await {
        Ok(env) => env,
        Err(error) => return fail(error),
    };
    let history = crate::global::global_config_dir().join("interactive_history.txt");
    let frontend: Arc<dyn Frontend> = match Terminal::new(Some(history)) {
        Ok(terminal) => Arc::new(terminal),
        Err(error) => {
            env.cleanup().await;
            return fail(format!("Could not open the terminal: {error}"));
        }
    };
    let built = match build_agent(
        &config,
        BuildOptions {
            default_kind: Some(AgentKind::Interactive),
            default_environment: "docker",
            frontend: Some(frontend),
            handle_ctrl_c: true,
            env: Some(Arc::clone(&env)),
            ..BuildOptions::default()
        },
    )
    .await
    {
        Ok(built) => built,
        Err(error) => {
            env.cleanup().await;
            return fail(error);
        }
    };
    let task = instance["problem_statement"].as_str().unwrap_or_default();
    let result = built.agent.run(task, Map::new()).await;
    env.cleanup().await;
    match result {
        Ok(run) => {
            eprintln!("Saved trajectory to '{}'", output.display());
            eprintln!("Exit status: {}", run.exit.exit_status);
            ExitCode::SUCCESS
        }
        Err(error) => fail(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn instances_are_picked_by_id_or_sorted_index() {
        let instances = vec![json!({ "instance_id": "b__2" }), json!({ "instance_id": "a__1" })];
        assert_eq!(select_instance(instances.clone(), "0").unwrap()["instance_id"], "a__1");
        assert_eq!(
            select_instance(instances.clone(), "b__2").unwrap()["instance_id"],
            "b__2"
        );
        assert!(select_instance(instances.clone(), "5").is_err());
        assert!(select_instance(instances, "c__3").is_err());
    }
}
