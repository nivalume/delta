//! `mini`: the agent in the local environment, interactive by default (upstream's
//! `run/mini.py`).

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use serde_json::Value as Json;

use super::{BuildOptions, build_agent, default_mini_output, startup_banner};
use crate::agent::{AgentKind, Frontend, Input, Prompt, Terminal};
use crate::config::{self, Overrides};

const CONFIG_HELP: &str = "Path to config files, filenames, or key-value pairs.

IMPORTANT: If you set this option, the default config file will not be used. So you need to explicitly set it e.g., with -c mini.yaml <other options>

Multiple configs will be recursively merged.

Examples:

-c model.model_kwargs.temperature=0   (You forgot to add the default config file! See above.)

-c mini.yaml -c model.model_kwargs.temperature=0.5

-c swebench.yaml -c agent.mode=yolo";

/// Run mini-SWE-agent in your local environment.
#[derive(Debug, Parser)]
#[command(name = "mini", version = crate::trajectory::MINI_VERSION)]
pub struct MiniArgs {
    /// Model to use.
    #[arg(short = 'm', long = "model")]
    pub model: Option<String>,
    /// Model class to use (e.g., 'litellm' or 'litellm_textbased').
    #[arg(long = "model-class")]
    pub model_class: Option<String>,
    /// Agent class to use (e.g., 'interactive' or 'default').
    #[arg(long = "agent-class")]
    pub agent_class: Option<String>,
    /// Environment class to use (e.g., 'local' or 'docker').
    #[arg(long = "environment-class")]
    pub environment_class: Option<String>,
    /// Task/problem statement.
    #[arg(short = 't', long = "task")]
    pub task: Option<String>,
    /// Run without confirmation.
    #[arg(short = 'y', long = "yolo")]
    pub yolo: bool,
    /// Cost limit. Set to 0 to disable.
    #[arg(short = 'l', long = "cost-limit")]
    pub cost_limit: Option<f64>,
    #[arg(short = 'c', long = "config", help = CONFIG_HELP)]
    pub config: Vec<String>,
    /// Output trajectory file.
    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,
    /// Exit immediately when the agent wants to finish instead of prompting.
    #[arg(long = "exit-immediately")]
    pub exit_immediately: bool,
}

/// The config files `mini` reads when `-c` names none: `$MSWEA_MINI_CONFIG_PATH`, else
/// `mini.yaml`.
fn default_specs() -> Vec<String> {
    vec![crate::global::env_var("MSWEA_MINI_CONFIG_PATH").unwrap_or_else(|| "mini.yaml".to_owned())]
}

/// The merged config of `args`.
pub fn mini_config(args: &MiniArgs) -> Result<Json, String> {
    let specs = if args.config.is_empty() {
        default_specs()
    } else {
        args.config.clone()
    };
    eprintln!("Building agent config from specs: {specs:?}");
    let mut config = config::load(&specs).map_err(|error| error.to_string())?;
    let output = args.output.clone().unwrap_or_else(default_mini_output);
    Overrides::default()
        .set("run.task", args.task.clone())
        .set("agent.agent_class", args.agent_class.clone())
        .set("agent.mode", args.yolo.then_some("yolo"))
        .set("agent.cost_limit", args.cost_limit)
        .set("agent.confirm_exit", args.exit_immediately.then_some(false))
        .set(
            "agent.output_path",
            Some(output.display().to_string()).filter(|path| !path.is_empty()),
        )
        .set("model.model_class", args.model_class.clone())
        .set("model.model_name", args.model.clone())
        .set("environment.environment_class", args.environment_class.clone())
        .apply(&mut config);
    Ok(config)
}

/// Runs `mini`.
pub async fn run_mini(args: MiniArgs) -> ExitCode {
    startup_banner();
    if let Err(error) = crate::global::configure_if_first_time() {
        eprintln!("Could not write the global config: {error}");
    }
    let config = match mini_config(&args) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let history = crate::global::global_config_dir().join("interactive_history.txt");
    let frontend: Arc<dyn Frontend> = match Terminal::new(Some(history)) {
        Ok(terminal) => Arc::new(terminal),
        Err(error) => {
            eprintln!("Could not open the terminal: {error}");
            return ExitCode::FAILURE;
        }
    };
    let task = match config["run"]["task"].as_str() {
        Some(task) => task.to_owned(),
        None => {
            let reader = Arc::clone(&frontend);
            let input = tokio::task::spawn_blocking(move || match reader.read_line(Prompt::Task) {
                Input::Line(line) if line == "/m" => reader.read_multiline(),
                other => other,
            })
            .await
            .unwrap_or(Input::Eof);
            let Input::Line(task) = input else {
                return ExitCode::FAILURE;
            };
            eprintln!("Got that, thanks!");
            task
        }
    };
    let built = match build_agent(
        &config,
        BuildOptions {
            default_kind: Some(AgentKind::Interactive),
            default_environment: "local",
            frontend: Some(frontend),
            handle_ctrl_c: true,
            ..BuildOptions::default()
        },
    )
    .await
    {
        Ok(built) => built,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let result = built.agent.run(&task, serde_json::Map::new()).await;
    built.env.cleanup().await;
    match result {
        Ok(output) => {
            if let Some(path) = config["agent"]["output_path"].as_str() {
                eprintln!("Saved trajectory to '{path}'");
            }
            eprintln!(
                "Exit status: {}",
                if output.exit.exit_status.is_empty() {
                    "unknown"
                } else {
                    &output.exit.exit_status
                }
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_line_options_override_the_configs() {
        let args = MiniArgs::parse_from([
            "mini",
            "-c",
            "mini.yaml",
            "-c",
            "agent.cost_limit=1",
            "-m",
            "openai/gpt-5",
            "-y",
            "-t",
            "Fix it",
            "-o",
            "out.traj.json",
            "--exit-immediately",
        ]);
        let config = mini_config(&args).unwrap();
        assert_eq!(config["run"]["task"], "Fix it");
        assert_eq!(config["agent"]["mode"], "yolo");
        assert_eq!(config["agent"]["cost_limit"], 1);
        assert_eq!(config["agent"]["confirm_exit"], false);
        assert_eq!(config["agent"]["output_path"], "out.traj.json");
        assert_eq!(config["model"]["model_name"], "openai/gpt-5");
        assert!(config["model"].get("model_class").is_none());
    }
}
