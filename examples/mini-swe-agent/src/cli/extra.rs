//! `mini-extra`: the global config, the inspector and the SWE-bench runners (upstream's
//! `run/utilities/mini_extra.py`).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use crate::swebench::{SingleArgs, SwebenchArgs, run_batch, run_single};

/// The central entry point for all extra commands from mini-swe-agent.
#[derive(Debug, Parser)]
#[command(name = "mini-extra", version = crate::trajectory::MINI_VERSION)]
pub struct ExtraArgs {
    #[command(subcommand)]
    pub command: ExtraCommand,
}

#[derive(Debug, Subcommand)]
pub enum ExtraCommand {
    /// Manage the global config file.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Run inspector (browse trajectories).
    #[command(visible_aliases = ["i", "inspector"])]
    Inspect {
        /// Directory to search for trajectory files or specific trajectory file.
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Show reasoning content (the default).
        #[arg(long, overrides_with = "no_reasoning")]
        reasoning: bool,
        /// Hide reasoning content.
        #[arg(long = "no-reasoning")]
        no_reasoning: bool,
    },
    /// Evaluate on SWE-bench (batch mode).
    Swebench(SwebenchArgs),
    /// Evaluate on SWE-bench (single instance).
    SwebenchSingle(SingleArgs),
}

/// The global config file; you can also edit the `.env` file in the config directory.
#[derive(Debug, Subcommand)]
pub enum ConfigCommand {
    /// Set up the global config file.
    Setup,
    /// Set a key in the global config file.
    Set { key: Option<String>, value: Option<String> },
    /// Unset a key in the global config file.
    Unset { key: Option<String> },
    /// Edit the global config file.
    Edit,
}

/// Runs `mini-extra`.
pub async fn run_extra(args: ExtraArgs) -> ExitCode {
    let io = |result: std::io::Result<()>| match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    };
    match args.command {
        ExtraCommand::Config { command } => io(match command {
            ConfigCommand::Setup => crate::global::setup(),
            ConfigCommand::Set { key, value } => crate::global::set(key, value),
            ConfigCommand::Unset { key } => crate::global::unset(key),
            ConfigCommand::Edit => crate::global::edit(),
        }),
        ExtraCommand::Inspect { path, no_reasoning, .. } => match crate::inspector::trajectory_files(&path) {
            Ok(files) => io(crate::inspector::run(files, !no_reasoning)),
            Err(error) => {
                eprintln!("{error}");
                ExitCode::FAILURE
            }
        },
        ExtraCommand::Swebench(args) => {
            super::startup_banner();
            match run_batch(args).await {
                Ok(summary) => {
                    eprintln!("Ran {} instances.", summary.instances);
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::FAILURE
                }
            }
        }
        ExtraCommand::SwebenchSingle(args) => {
            super::startup_banner();
            run_single(args).await
        }
    }
}
