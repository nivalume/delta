//! `mini-extra`: credentials, models, the global config, the inspector and the SWE-bench
//! runners.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use delta_ai::auth::AuthType;

use super::account;
use crate::swebench::{SingleArgs, SwebenchArgs, run_batch, run_single};

/// delta-mini's extra commands.
#[derive(Debug, Parser)]
#[command(name = "mini-extra", version)]
pub struct ExtraArgs {
    #[command(subcommand)]
    pub command: ExtraCommand,
}

#[derive(Debug, Subcommand)]
pub enum ExtraCommand {
    /// The global config (`config.yaml` in the home directory).
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Log in to a provider and store the credential in `auth.json`.
    Login {
        /// The provider id; asked for when not given.
        provider: Option<String>,
        /// Log in with OAuth (the default where the provider has it).
        #[arg(long, conflicts_with = "api_key")]
        oauth: bool,
        /// Store an API key.
        #[arg(long = "api-key")]
        api_key: bool,
    },
    /// Remove a provider's stored credential.
    Logout { provider: String },
    /// List the models whose provider has credentials.
    Models {
        /// Only this provider's models.
        provider: Option<String>,
        /// Every model, credentials or not.
        #[arg(long)]
        all: bool,
    },
    /// Browse trajectories.
    #[command(visible_aliases = ["i", "inspector"])]
    Inspect {
        /// A trajectory file, or a directory to search for `*.traj.json` files.
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Hide thinking.
        #[arg(long = "no-reasoning")]
        no_reasoning: bool,
    },
    /// Evaluate on SWE-bench (batch mode).
    Swebench(SwebenchArgs),
    /// Evaluate on SWE-bench (single instance).
    SwebenchSingle(SingleArgs),
}

#[derive(Debug, Subcommand)]
pub enum ConfigCommand {
    /// Choose the default model, logging in to its provider when needed.
    Setup,
    /// Show the home directory, the global config and the stored credentials.
    Show,
    /// Edit the global config in `$EDITOR`.
    Edit,
}

fn exit(result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

/// Runs `mini-extra`.
pub async fn run_extra(args: ExtraArgs) -> ExitCode {
    match args.command {
        ExtraCommand::Config { command } => exit(match command {
            ConfigCommand::Setup => account::setup_command().await,
            ConfigCommand::Show => account::show_command().await,
            ConfigCommand::Edit => crate::home::edit_config().map_err(|error| error.to_string()),
        }),
        ExtraCommand::Login {
            provider,
            oauth,
            api_key,
        } => {
            let method = if oauth {
                Some(AuthType::OAuth)
            } else if api_key {
                Some(AuthType::ApiKey)
            } else {
                None
            };
            exit(account::login_command(provider, method).await)
        }
        ExtraCommand::Logout { provider } => exit(account::logout_command(&provider).await),
        ExtraCommand::Models { provider, all } => exit(account::models_command(provider.as_deref(), all).await),
        ExtraCommand::Inspect { path, no_reasoning } => exit(
            crate::inspector::trajectory_files(&path)
                .and_then(|files| crate::inspector::run(files, !no_reasoning).map_err(|error| error.to_string())),
        ),
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
