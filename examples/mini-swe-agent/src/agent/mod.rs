//! The agent on a harness conversation.
//!
//! Each run opens a Harness with one root conversation whose instructions are the rendered
//! system template. The task is the first input; the harness runs model turns and tool
//! rounds, and this module's `harness` pieces (the `bash` tool, the generation hooks) keep
//! mini-swe-agent's step semantics and record the trajectory. Whatever ends a harness run
//! without ending the agent (a rejection, a new task, human mode, Ctrl+C) queues a message,
//! and the driver delivers it as the next run's input.

mod config;
mod frontend;
mod harness;
mod state;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use delta_ai::model::Model;
use delta_ai::types::{Message, UserMessage};
use delta_chord::context::Context;
use delta_harness::agent::{
    AgentChange, Change, CompactionSettings, HarnessSettings, PartialCompactionPolicy, SettingsSource,
};
use delta_harness::generation::GENERATION_TASK;
use delta_harness::harness::{AbortOptions, Harness, HarnessOptions, RootOptions};
use delta_harness::inbox::{SubmissionDraft, USER_ENTRY};
use delta_harness::model::{ModelClient, ModelRef};
use delta_harness::records::EntryDraft;
use delta_harness::registry::{Extension, Registry};
use delta_harness::storage::{MemoryStorage, SqliteOptions, SqliteStorage, Storage};
use delta_harness::tool::ToolExecutionMode;
use regex::Regex;
use serde_json::{Map, Value as Json};
use tokio::sync::Notify;

pub use config::{AgentConfig, AgentKind, Mode};
pub use frontend::{Frontend, Input, Notice, Prompt, Scripted, Terminal, kind_title};
pub use harness::BASH_TOOL_NAME;

use self::harness::{BashTool, Hooks};
use self::state::{Shared, State, unfinished_observation};
use crate::env::Environment;
use crate::model::{ActionFormat, BatchStats, ModelSettings};
use crate::trajectory::{Action, Entry, Exit, ExitStatus, InterruptReason, Trajectory, system_message, user_message};

/// Why a run could not start or continue.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("{0}")]
    Config(String),
    #[error("Template error: {0}")]
    Template(#[from] minijinja::Error),
    #[error("Harness error: {0}")]
    Harness(String),
}

impl AgentError {
    fn harness(error: impl std::fmt::Display) -> Self {
        Self::Harness(error.to_string())
    }
}

/// What an agent is built from.
pub struct AgentOptions {
    /// The merged config: its `agent` section, and its `providers` for the trajectory.
    pub config: Json,
    pub kind: AgentKind,
    pub model: ModelSettings,
    /// The delta-ai model the client streams.
    pub resolved: Model,
    /// Model access for the harness.
    pub client: Arc<dyn ModelClient>,
    pub env: Arc<dyn Environment>,
    /// Shows entries and asks the interactive agent's questions.
    pub frontend: Option<Arc<dyn Frontend>>,
    /// Where the trajectory is saved after every step; with it, the harness database beside.
    pub output: Option<PathBuf>,
    /// The trajectory's `extra`.
    pub extra: Map<String, Json>,
    /// Ctrl+C interrupts the agent instead of ending the process.
    pub handle_ctrl_c: bool,
    /// Called before each model call with the step about to run and the cost so far.
    pub on_step: Option<StepCallback>,
    /// The batch this run belongs to, with its limits.
    pub batch: Option<Arc<BatchStats>>,
}

/// A callback taking the step about to run and the cost so far.
pub type StepCallback = Arc<dyn Fn(u64, f64) + Send + Sync>;

impl std::fmt::Debug for AgentOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentOptions")
            .field("kind", &self.kind)
            .field("model", &self.resolved.id)
            .finish_non_exhaustive()
    }
}

/// How a run ended.
#[derive(Clone, Debug)]
pub struct RunOutput {
    pub exit: Exit,
    /// The trajectory document, as saved.
    pub trajectory: Trajectory,
}

/// An agent: one task at a time.
pub struct Agent {
    shared: Arc<Shared>,
    client: Arc<dyn ModelClient>,
    model_ref: ModelRef,
    handle_ctrl_c: bool,
    interrupt: Arc<Notify>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("model_ref", &self.model_ref)
            .finish_non_exhaustive()
    }
}

/// The harness database beside a trajectory: `x.traj.json` keeps `x.harness.db`.
pub fn harness_db_path(output: &Path) -> PathBuf {
    let name = output
        .file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
    let stem = name
        .strip_suffix(".traj.json")
        .or_else(|| name.strip_suffix(".json"))
        .unwrap_or(&name);
    output.with_file_name(format!("{stem}.harness.db"))
}

impl Agent {
    pub fn new(options: AgentOptions) -> Result<Self, AgentError> {
        let config = AgentConfig::from_section(&options.config["agent"]).map_err(AgentError::Config)?;
        let whitelist = config
            .whitelist
            .iter()
            .map(|pattern| Regex::new(&format!("^(?:{pattern})")))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| AgentError::Config(format!("Invalid agent.whitelist: {error}")))?;
        let script_costs = options
            .model
            .script()
            .map_err(|error| AgentError::Config(error.to_string()))?
            .into_iter()
            .map(|(_, cost)| cost)
            .collect();
        let model_ref = ModelRef::new(&options.resolved.provider, &options.resolved.id);
        let state = State {
            config,
            trajectory: Trajectory::default(),
            n_calls: 0,
            cost: 0.0,
            format_errors: 0,
            round: None,
            queue: Vec::new(),
            interrupted: false,
            last_error: None,
            extra_vars: Json::Object(Map::new()),
        };
        Ok(Self {
            shared: Arc::new(Shared {
                kind: options.kind,
                model: options.model,
                model_info: serde_json::to_value(&options.resolved).unwrap_or(Json::Null),
                script_costs,
                providers: options.config["providers"].clone(),
                env: options.env,
                frontend: options.frontend,
                output_path: options.output,
                extra: options.extra,
                whitelist,
                started: Instant::now(),
                state: Mutex::new(state),
                stop: Notify::new(),
                on_step: options.on_step,
                batch: options.batch,
            }),
            client: options.client,
            model_ref,
            handle_ctrl_c: options.handle_ctrl_c,
            interrupt: Arc::new(Notify::new()),
        })
    }

    /// Notifying it interrupts the running step as Ctrl+C does.
    pub fn interrupter(&self) -> Arc<Notify> {
        Arc::clone(&self.interrupt)
    }

    pub fn kind(&self) -> AgentKind {
        self.shared.kind
    }

    /// The trajectory document so far.
    pub fn serialize(&self) -> Trajectory {
        let state = self.shared.lock();
        self.shared.serialize(&state)
    }

    /// Runs `task` until the agent exits. `extra_vars` are more template variables.
    pub async fn run(&self, task: &str, extra_vars: Map<String, Json>) -> Result<RunOutput, AgentError> {
        let ctx = Context::background();
        let system = {
            let mut state = self.shared.lock();
            let mut vars = extra_vars;
            vars.insert("task".to_owned(), Json::String(task.to_owned()));
            state.extra_vars = Json::Object(vars);
            state.trajectory = Trajectory::default();
            let template_vars = self.shared.template_vars(&state);
            let system = crate::template::render(&state.config.system_template, &template_vars)?;
            let instance = crate::template::render(&state.config.instance_template, &template_vars)?;
            self.shared.push(
                &mut state,
                Entry::System {
                    message: system_message(&system),
                },
            );
            self.shared.push(
                &mut state,
                Entry::Task {
                    message: user_message(&instance),
                },
            );
            state.queue = vec![instance];
            self.shared.save(&state);
            system
        };

        let harness = self.open_harness(&ctx).await?;
        let result = self.drive(&harness, system, &ctx).await;
        let _ = harness.close(&ctx).await;
        result?;

        let mut state = self.shared.lock();
        state.exit(ExitStatus::Error, Some("The run ended without an exit.".to_owned()));
        self.shared.save(&state);
        let trajectory = self.shared.serialize(&state);
        Ok(RunOutput {
            exit: trajectory
                .exit
                .clone()
                .unwrap_or_else(|| Exit::new(ExitStatus::Error, None)),
            trajectory,
        })
    }

    async fn open_harness(&self, ctx: &Context) -> Result<Harness, AgentError> {
        let storage: Arc<dyn Storage> = match &self.shared.output_path {
            Some(output) => {
                let path = harness_db_path(output);
                if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
                    std::fs::create_dir_all(parent).map_err(AgentError::harness)?;
                }
                for suffix in ["", "-wal", "-shm"] {
                    let mut stale = path.clone().into_os_string();
                    stale.push(suffix);
                    let _ = std::fs::remove_file(stale);
                }
                Arc::new(
                    SqliteStorage::open(&path, SqliteOptions::default())
                        .await
                        .map_err(AgentError::harness)?,
                )
            }
            None => Arc::new(MemoryStorage::new()),
        };
        let mut extension = Extension::new("delta-mini").with_hooks(
            GENERATION_TASK,
            Arc::new(Hooks {
                shared: Arc::clone(&self.shared),
            }) as Arc<dyn delta_harness::registry::GenerationHooks>,
        );
        if self.shared.model.config.actions == ActionFormat::ToolCalls {
            extension = extension.with_tool(Arc::new(BashTool {
                shared: Arc::clone(&self.shared),
            }));
        }
        let registry = Registry::new();
        registry.install(extension).map_err(AgentError::harness)?;
        let mut options = HarnessOptions::new(Arc::clone(&self.client));
        options.registry = registry;
        options.settings = SettingsSource::Fixed(HarnessSettings {
            tool_execution: Some(ToolExecutionMode::Sequential),
            max_continuations: Some(u32::MAX),
            compaction: Some(CompactionSettings {
                policy: PartialCompactionPolicy {
                    enabled: Some(false),
                    ..PartialCompactionPolicy::default()
                },
                ..CompactionSettings::default()
            }),
            ..HarnessSettings::default()
        });
        Harness::open(storage, options, ctx).await.map_err(AgentError::harness)
    }

    /// The driver: runs the harness until the agent exits.
    async fn drive(&self, harness: &Harness, system: String, ctx: &Context) -> Result<(), AgentError> {
        let agent = AgentChange {
            model: Change::Set(self.model_ref.clone()),
            thinking_level: self.shared.model.config.thinking.map_or(Change::Keep, Change::Set),
            instructions: Change::Set(system),
            ..AgentChange::default()
        };
        let conversation = harness
            .root(
                RootOptions {
                    agent: Some(agent),
                    init: None,
                },
                ctx,
            )
            .await
            .map_err(AgentError::harness)?;
        loop {
            if self.shared.exited() {
                return Ok(());
            }
            let mode = self.shared.lock().config.mode;
            if self.shared.kind == AgentKind::Interactive && mode == Mode::Human {
                self.human_step(ctx).await;
                continue;
            }
            if !self.shared.check_limits().await {
                return Ok(());
            }
            let inputs = std::mem::take(&mut self.shared.lock().queue);
            let Some((last, earlier)) = inputs.split_last() else {
                self.end_with_error("The agent has no message for the model.");
                return Ok(());
            };
            for text in earlier {
                let entry = EntryDraft {
                    model: Some(vec![Message::User(UserMessage {
                        content: text.clone().into(),
                        timestamp: crate::trajectory::now_millis(),
                    })]),
                    ..EntryDraft::new(USER_ENTRY)
                };
                conversation
                    .submit(SubmissionDraft::write(entry), ctx)
                    .await
                    .map_err(AgentError::harness)?;
            }
            let submission = conversation
                .submit(SubmissionDraft::input(last.clone()), ctx)
                .await
                .map_err(AgentError::harness)?;
            let ctrl_c = async {
                if self.handle_ctrl_c {
                    let _ = tokio::signal::ctrl_c().await;
                } else {
                    std::future::pending::<()>().await;
                }
            };
            let interrupted = tokio::select! {
                record = submission.wait(ctx) => {
                    record.map_err(AgentError::harness)?;
                    false
                }
                () = self.shared.stop.notified() => {
                    conversation.abort(AbortOptions::default(), ctx).await.map_err(AgentError::harness)?;
                    false
                }
                () = ctrl_c => {
                    conversation.abort(AbortOptions::default(), ctx).await.map_err(AgentError::harness)?;
                    true
                }
                () = self.interrupt.notified() => {
                    conversation.abort(AbortOptions::default(), ctx).await.map_err(AgentError::harness)?;
                    true
                }
            };
            conversation.wait_for_idle(ctx).await.map_err(AgentError::harness)?;
            let interrupted = interrupted | std::mem::take(&mut self.shared.lock().interrupted);
            if self.shared.exited() {
                return Ok(());
            }
            if interrupted {
                self.pad_unfinished_round();
                self.interrupt_prompt().await;
                continue;
            }
            if self.shared.lock().queue.is_empty() {
                let failed = self.shared.lock().last_error.clone();
                match failed {
                    Some(answer) => {
                        let message = answer
                            .error_message
                            .clone()
                            .unwrap_or_else(|| "The model call failed".to_owned());
                        self.end_with_error(&format!("The model call failed: {message}"));
                    }
                    None => self.end_with_error("The run ended without an exit."),
                }
                return Ok(());
            }
        }
    }

    /// The run failed.
    fn end_with_error(&self, message: &str) {
        let mut state = self.shared.lock();
        state.exit(ExitStatus::Error, Some(message.to_owned()));
        self.shared.save(&state);
    }

    /// Ends the run with `status`.
    fn end(&self, status: ExitStatus) {
        let mut state = self.shared.lock();
        state.exit(status, None);
        self.shared.save(&state);
    }

    /// After Ctrl+C during a round whose commands never reported: their observations are
    /// "not executed", so every tool call has its result.
    fn pad_unfinished_round(&self) {
        let round = self.shared.lock().round.take();
        let Some(round) = round else { return };
        if round.rejection.is_some() || round.outcome.initialized() {
            return;
        }
        if self.shared.model.config.actions != ActionFormat::ToolCalls {
            return;
        }
        let mut state = self.shared.lock();
        for action in &round.actions {
            unfinished_observation(&self.shared, &mut state, action);
        }
        self.shared.save(&state);
    }

    /// After Ctrl+C: a comment for the model. An agent nobody can ask exits.
    async fn interrupt_prompt(&self) {
        if self.shared.kind != AgentKind::Interactive || self.shared.frontend.is_none() {
            self.end(ExitStatus::Interrupted);
            return;
        }
        let comment = match self.shared.ask_with_commands(Prompt::Interrupted).await {
            Input::Line(line) => line.trim().to_owned(),
            Input::Interrupted => String::new(),
            Input::Eof => {
                self.end(ExitStatus::Eof);
                return;
            }
        };
        let comment = if comment.is_empty() || Mode::from_command(&comment).is_some() {
            "Temporary interruption caught.".to_owned()
        } else {
            comment
        };
        let mut state = self.shared.lock();
        self.shared.push_interrupt(
            &mut state,
            format!("Interrupted by user: {comment}"),
            InterruptReason::UserInterruption,
        );
        self.shared.save(&state);
    }

    /// Human mode: the user's command runs and its observation queues for the model.
    async fn human_step(&self, ctx: &Context) {
        let command = match self.shared.ask_with_commands(Prompt::Human).await {
            Input::Line(command) => command,
            Input::Interrupted => {
                self.interrupt_prompt().await;
                return;
            }
            Input::Eof => {
                self.end(ExitStatus::Eof);
                return;
            }
        };
        if command == "/y" || command == "/c" {
            return;
        }
        {
            let mut state = self.shared.lock();
            let content = format!("User command: \n```bash\n{command}\n```");
            self.shared.push(
                &mut state,
                Entry::Human {
                    message: user_message(&content),
                    command: command.clone(),
                },
            );
            state.queue.push(content);
        }
        let action = Action {
            command,
            tool_call_id: None,
        };
        let outcome = self.shared.run_actions(&[action], true, ctx).await;
        let mut state = self.shared.lock();
        state.queue.extend(outcome.observations);
        state.queue.extend(outcome.interrupt);
    }
}
