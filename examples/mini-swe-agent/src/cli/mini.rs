//! `mini`: the agent in the local environment, interactive.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use delta_ai::model::ModelThinkingLevel;
use serde_json::Value as Json;

use super::{BuildOptions, build_agent, default_mini_output, load_config, startup_banner, with_model};
use crate::agent::{AgentKind, Frontend, Input, Prompt, Terminal};
use crate::config::Overrides;

const CONFIG_HELP: &str =
    "Config files (paths or built-in names) and key.path=value pairs, merged in order over the global config.

Setting it replaces the default `mini.yaml`, so name that too:

-c mini.yaml -c model.temperature=0.5

-c mini_textbased.yaml -c agent.cost_limit=1";

/// Run delta-mini in your local environment.
#[derive(Debug, Parser)]
#[command(name = "mini", version)]
pub struct MiniArgs {
    /// Model as provider/model-id (e.g. anthropic/claude-sonnet-4-5), or a model id one
    /// provider has.
    #[arg(short = 'm', long = "model")]
    pub model: Option<String>,
    /// Thinking level.
    #[arg(long, value_parser = parse_thinking)]
    pub thinking: Option<ModelThinkingLevel>,
    /// Environment type: local or docker.
    #[arg(long)]
    pub environment: Option<String>,
    /// Task/problem statement.
    #[arg(short = 't', long = "task")]
    pub task: Option<String>,
    /// Run without confirmation.
    #[arg(short = 'y', long = "yolo")]
    pub yolo: bool,
    /// Cost limit in USD. Set to 0 to disable.
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

/// A thinking level by name.
pub(super) fn parse_thinking(level: &str) -> Result<ModelThinkingLevel, String> {
    ModelThinkingLevel::ALL
        .into_iter()
        .find(|known| known.as_str() == level)
        .ok_or_else(|| "expected off, minimal, low, medium, high, xhigh or max".to_owned())
}

/// What `mini` runs: the merged config, the task when the command line gives one, and the
/// trajectory file.
#[derive(Clone, Debug)]
pub struct MiniRun {
    pub config: Json,
    pub task: Option<String>,
    pub output: PathBuf,
}

/// The run `args` describe.
pub fn mini_config(args: &MiniArgs) -> Result<MiniRun, String> {
    let mut config = load_config(&args.config, "mini.yaml")?;
    with_model(Overrides::default(), args.model.as_deref())
        .set("model.thinking", args.thinking.map(ModelThinkingLevel::as_str))
        .set("environment.type", args.environment.clone())
        .set("agent.mode", args.yolo.then_some("yolo"))
        .set("agent.cost_limit", args.cost_limit)
        .set("agent.confirm_exit", args.exit_immediately.then_some(false))
        .apply(&mut config);
    Ok(MiniRun {
        config,
        task: args.task.clone(),
        output: args.output.clone().unwrap_or_else(default_mini_output),
    })
}

/// Runs `mini`.
pub async fn run_mini(args: MiniArgs) -> ExitCode {
    startup_banner();
    let run = match mini_config(&args) {
        Ok(run) => run,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let history = crate::home::home_dir().join("interactive_history.txt");
    let frontend: Arc<dyn Frontend> = match Terminal::new(Some(history)) {
        Ok(terminal) => Arc::new(terminal),
        Err(error) => {
            eprintln!("Could not open the terminal: {error}");
            return ExitCode::FAILURE;
        }
    };
    let built = match build_agent(
        &run.config,
        BuildOptions {
            kind: AgentKind::Interactive,
            frontend: Some(Arc::clone(&frontend)),
            handle_ctrl_c: true,
            output: Some(run.output.clone()),
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
    let task = match run.task {
        Some(task) => task,
        None => {
            let reader = Arc::clone(&frontend);
            let input = tokio::task::spawn_blocking(move || match reader.read_line(Prompt::Task) {
                Input::Line(line) if line == "/m" => reader.read_multiline(),
                other => other,
            })
            .await
            .unwrap_or(Input::Eof);
            let Input::Line(task) = input else {
                built.env.cleanup().await;
                return ExitCode::FAILURE;
            };
            task
        }
    };
    let result = built.agent.run(&task, serde_json::Map::new()).await;
    built.env.cleanup().await;
    match result {
        Ok(output) => {
            eprintln!("Saved trajectory to '{}'", run.output.display());
            match &output.exit.error {
                Some(error) => eprintln!("Exit status: {} ({error})", output.exit.status),
                None => eprintln!("Exit status: {}", output.exit.status),
            }
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
            "-c",
            "model.id=other",
            "-m",
            "openrouter/minimax/minimax-m2",
            "--thinking",
            "high",
            "-y",
            "-t",
            "Fix it",
            "-o",
            "out.traj.json",
            "--exit-immediately",
        ]);
        let run = mini_config(&args).unwrap();
        let config = &run.config;
        assert_eq!(run.task.as_deref(), Some("Fix it"));
        assert_eq!(run.output, PathBuf::from("out.traj.json"));
        assert_eq!(config["agent"]["mode"], "yolo");
        assert_eq!(config["agent"]["cost_limit"], 1);
        assert_eq!(config["agent"]["confirm_exit"], false);
        assert_eq!(config["model"]["provider"], "openrouter");
        assert_eq!(config["model"]["id"], "minimax/minimax-m2");
        assert_eq!(config["model"]["thinking"], "high");
        assert!(MiniArgs::try_parse_from(["mini", "--thinking", "huge"]).is_err());

        let bare = mini_config(&MiniArgs::parse_from(["mini", "-m", "gpt-5"])).unwrap();
        assert_eq!(
            (
                bare.config["model"]["provider"].as_str(),
                bare.config["model"]["id"].as_str()
            ),
            (Some(""), Some("gpt-5"))
        );
    }
}
