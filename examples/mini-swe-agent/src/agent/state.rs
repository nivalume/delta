//! What the bash tool, the generation hooks and the driver share: the trajectory, the
//! counters, the mode and limits, and the steps of a run (confirm, execute, observe, submit,
//! check the limits).

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use delta_ai::types::{AssistantBlock, AssistantMessage, Message, StopReason};
use delta_chord::context::Context;
use regex::Regex;
use serde_json::{Map, Value as Json, json};
use tokio::sync::{Notify, OnceCell};

use super::config::{AgentConfig, AgentKind, Mode};
use super::frontend::{Frontend, Input, Notice, Prompt};
use crate::env::{Environment, ExecOutput};
use crate::model::{ActionFormat, BatchStats, ModelSettings};
use crate::trajectory::{
    Action, Entry, Exit, ExitStatus, InterruptReason, Stats, Trajectory, tool_result, user_message,
};

/// An answer whose actions did not parse: the rendered format error the model gets, and why.
#[derive(Clone, Debug)]
pub(super) struct Rejection {
    pub(super) content: String,
    pub(super) error: String,
}

/// The actions of the model's latest answer.
#[derive(Debug)]
pub(super) struct Round {
    pub(super) actions: Vec<Action>,
    /// Set when the answer did not parse: nothing runs.
    pub(super) rejection: Option<Rejection>,
    /// Filled by the first tool call of the round, which runs every action.
    pub(super) outcome: OnceCell<Outcome>,
}

/// What running a round's actions produced.
#[derive(Clone, Debug, Default)]
pub(super) struct Outcome {
    /// One per action: the rendered observation (padded with "not executed" for tool calls).
    pub(super) observations: Vec<String>,
    /// A message for the model that ends the run (a rejection, a new task); already in the
    /// trajectory.
    pub(super) interrupt: Option<String>,
    /// The run ends after this round.
    pub(super) stop: bool,
}

/// How a round ended early.
enum End {
    Interrupt {
        content: String,
        reason: InterruptReason,
    },
    Submitted(String),
    Exit(ExitStatus),
    /// Ctrl+C at a prompt: the driver asks for a comment.
    Interrupted,
    /// The run was aborted while a command ran.
    Aborted,
}

impl State {
    /// Ends the run with `status`, unless it ended already.
    pub(super) fn exit(&mut self, status: ExitStatus, error: Option<String>) {
        if self.trajectory.exit.is_none() {
            self.trajectory.exit = Some(Exit::new(status, error));
        }
    }
}

/// The mutable part.
#[derive(Debug)]
pub(super) struct State {
    pub(super) config: AgentConfig,
    pub(super) trajectory: Trajectory,
    pub(super) n_calls: u64,
    pub(super) cost: f64,
    pub(super) format_errors: u64,
    pub(super) round: Option<Arc<Round>>,
    /// Messages for the model, delivered with the next run.
    pub(super) queue: Vec<String>,
    /// Ctrl+C at a prompt during a run.
    pub(super) interrupted: bool,
    /// The latest failed answer.
    pub(super) last_error: Option<AssistantMessage>,
    /// `task` and the run's other template variables.
    pub(super) extra_vars: Json,
}

pub(super) struct Shared {
    pub(super) kind: AgentKind,
    pub(super) model: ModelSettings,
    /// The resolved delta-ai model, for the trajectory.
    pub(super) model_info: Json,
    /// The cost of each scripted answer.
    pub(super) script_costs: Vec<f64>,
    /// The config's `providers` section, for the trajectory.
    pub(super) providers: Json,
    pub(super) env: Arc<dyn Environment>,
    pub(super) frontend: Option<Arc<dyn Frontend>>,
    pub(super) output_path: Option<PathBuf>,
    /// The trajectory's `extra`.
    pub(super) extra: Map<String, Json>,
    pub(super) whitelist: Vec<Regex>,
    pub(super) started: Instant,
    pub(super) state: Mutex<State>,
    /// Asks the driver to abort the run: an exit was decided where no tool can end it.
    pub(super) stop: Notify,
    pub(super) on_step: Option<super::StepCallback>,
    pub(super) batch: Option<Arc<BatchStats>>,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shared")
            .field("kind", &self.kind)
            .field("output_path", &self.output_path)
            .finish_non_exhaustive()
    }
}

/// An action that did not run.
fn not_executed() -> ExecOutput {
    ExecOutput::exception(String::new(), "action was not executed")
}

impl Shared {
    pub(super) fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The frontend of an interactive agent.
    fn interactive(&self) -> Option<&Arc<dyn Frontend>> {
        self.frontend.as_ref().filter(|_| self.kind == AgentKind::Interactive)
    }

    /// Adds `entry` to the trajectory and shows it.
    pub(super) fn push(&self, state: &mut State, entry: Entry) {
        if let Some(frontend) = &self.frontend {
            frontend.show_entry(&entry, state.n_calls, state.cost);
        }
        state.trajectory.entries.push(entry);
    }

    /// Adds a message from the user that ends a step, and queues it for the model.
    pub(super) fn push_interrupt(&self, state: &mut State, content: String, reason: InterruptReason) {
        self.push(
            state,
            Entry::Interrupt {
                message: user_message(&content),
                reason,
            },
        );
        state.queue.push(content);
    }

    pub(super) fn exited(&self) -> bool {
        self.lock().trajectory.exit.is_some()
    }

    /// The template variables: the machine's (`system`, `release`, ...), the `agent`,
    /// `model` and `environment` sections, the counters and the run's variables (`task`).
    pub(super) fn template_vars(&self, state: &State) -> Json {
        crate::template::vars([
            crate::template::uname(),
            json!({
                "agent": state.config.dump(),
                "model": self.model.dump(),
                "environment": self.env.config(),
                "n_model_calls": state.n_calls,
                "model_cost": state.cost,
                "elapsed_seconds": self.started.elapsed().as_secs(),
            }),
            state.extra_vars.clone(),
        ])
    }

    /// The trajectory document.
    pub(super) fn serialize(&self, state: &State) -> Trajectory {
        let mut config = Map::new();
        config.insert("agent".to_owned(), state.config.dump());
        config.insert("model".to_owned(), self.model.dump());
        config.insert("environment".to_owned(), self.env.config());
        if !self.providers.is_null() {
            config.insert("providers".to_owned(), self.providers.clone());
        }
        Trajectory {
            stats: Stats {
                calls: state.n_calls,
                cost: state.cost,
                elapsed_seconds: self.started.elapsed().as_secs_f64(),
            },
            model: self.model_info.clone(),
            config: Json::Object(config),
            extra: self.extra.clone(),
            ..state.trajectory.clone()
        }
    }

    /// Saves the trajectory when the agent has an output path.
    pub(super) fn save(&self, state: &State) {
        if let Some(path) = &self.output_path
            && let Err(error) = self.serialize(state).save(path)
        {
            eprintln!("Could not save the trajectory to {}: {error}", path.display());
        }
    }

    /// The run's limit the next model call would exceed.
    fn run_limit(&self, state: &State) -> Option<ExitStatus> {
        let config = &state.config;
        if (config.step_limit > 0 && config.step_limit <= state.n_calls)
            || (config.cost_limit > 0.0 && config.cost_limit <= state.cost)
        {
            return Some(ExitStatus::LimitsExceeded);
        }
        let limit = config.time_limit;
        (limit > 0 && limit <= self.started.elapsed().as_secs()).then_some(ExitStatus::TimeExceeded)
    }

    /// Whether another model call may happen. The batch's limits end the run. Over its own
    /// step or cost limit, an interactive agent with a person at the terminal asks for new
    /// limits; otherwise (and over the time limit, which new limits cannot lift) the run
    /// exits.
    pub(super) async fn check_limits(&self) -> bool {
        let status = {
            let mut state = self.lock();
            if state.trajectory.exit.is_some() {
                return false;
            }
            if let Some(error) = self.batch.as_ref().and_then(|batch| batch.exceeded()) {
                state.exit(ExitStatus::LimitsExceeded, Some(error));
                self.save(&state);
                return false;
            }
            match self.run_limit(&state) {
                None => return true,
                Some(status) => status,
            }
        };
        if status == ExitStatus::LimitsExceeded
            && let Some(frontend) = self.interactive().filter(|frontend| frontend.is_interactive())
        {
            {
                let state = self.lock();
                frontend.notice(Notice::LimitsExceeded {
                    step_limit: state.config.step_limit,
                    cost_limit: state.config.cost_limit,
                    steps: state.n_calls,
                    cost: state.cost,
                });
            }
            let step_limit = self.ask(Prompt::StepLimit).await;
            let cost_limit = self.ask(Prompt::CostLimit).await;
            if let (Input::Line(step_limit), Input::Line(cost_limit)) = (step_limit, cost_limit)
                && let (Ok(step_limit), Ok(cost_limit)) = (step_limit.trim().parse(), cost_limit.trim().parse())
            {
                let mut state = self.lock();
                state.config.step_limit = step_limit;
                state.config.cost_limit = cost_limit;
                if self.run_limit(&state).is_none() {
                    return true;
                }
            }
        }
        let mut state = self.lock();
        state.exit(status, None);
        self.save(&state);
        false
    }

    /// Asks the frontend, off the async workers.
    pub(super) async fn ask(&self, prompt: Prompt) -> Input {
        let Some(frontend) = self.frontend.clone() else {
            return Input::Eof;
        };
        tokio::task::spawn_blocking(move || frontend.read_line(prompt))
            .await
            .unwrap_or(Input::Eof)
    }

    /// Asks `prompt` and handles the slash commands: `/h` shows help and asks again, `/m`
    /// reads several lines, a mode command switches the mode (or, when already in it, asks
    /// again).
    pub(super) async fn ask_with_commands(&self, prompt: Prompt) -> Input {
        let Some(frontend) = self.frontend.clone() else {
            return Input::Eof;
        };
        loop {
            let input = self.ask(prompt).await;
            let Input::Line(line) = &input else {
                return input;
            };
            if line == "/m" {
                let frontend = Arc::clone(&frontend);
                return tokio::task::spawn_blocking(move || frontend.read_multiline())
                    .await
                    .unwrap_or(Input::Eof);
            }
            if line == "/h" {
                frontend.notice(Notice::Help {
                    mode: self.lock().config.mode,
                });
                continue;
            }
            if let Some(mode) = Mode::from_command(line) {
                let mut state = self.lock();
                if state.config.mode == mode {
                    frontend.notice(Notice::AlreadyIn { mode });
                    continue;
                }
                state.config.mode = mode;
                frontend.notice(Notice::Switched { mode });
            }
            return input;
        }
    }

    /// Whether the user must confirm `commands`.
    fn needs_confirmation(&self, commands: &[String]) -> bool {
        self.interactive().is_some()
            && self.lock().config.mode == Mode::Confirm
            && commands
                .iter()
                .any(|command| !self.whitelist.iter().any(|pattern| pattern.is_match(command)))
    }

    /// Asks the user to confirm `commands`; `Some` when the round ends instead.
    async fn confirm(&self, commands: &[String]) -> Option<End> {
        if !self.needs_confirmation(commands) {
            return None;
        }
        match self
            .ask_with_commands(Prompt::Confirm {
                actions: commands.len(),
            })
            .await
        {
            Input::Line(line) => match line.trim() {
                "" | "/y" => None,
                "/u" => Some(End::Interrupt {
                    content: "Commands not executed. Switching to human mode".to_owned(),
                    reason: InterruptReason::UserRejection,
                }),
                comment => Some(End::Interrupt {
                    content: format!(
                        "Commands not executed. The user rejected your commands with the following message: {comment}"
                    ),
                    reason: InterruptReason::UserRejection,
                }),
            },
            Input::Interrupted => Some(End::Interrupted),
            Input::Eof => Some(End::Exit(ExitStatus::Eof)),
        }
    }

    /// The agent wants to finish: a new task, or Enter to submit.
    async fn confirm_exit(&self, submission: String) -> End {
        let asks = self.interactive().is_some() && self.lock().config.confirm_exit;
        if !asks {
            return End::Submitted(submission);
        }
        loop {
            match self.ask_with_commands(Prompt::ConfirmExit).await {
                Input::Line(line) => match line.trim() {
                    "" => return End::Submitted(submission),
                    "/u" => {
                        return End::Interrupt {
                            content: "Switched to human mode.".to_owned(),
                            reason: InterruptReason::UserInterruption,
                        };
                    }
                    "/y" | "/c" => {}
                    task => {
                        return End::Interrupt {
                            content: format!("The user added a new task: {task}"),
                            reason: InterruptReason::UserNewTask,
                        };
                    }
                },
                Input::Interrupted => return End::Interrupted,
                Input::Eof => return End::Exit(ExitStatus::Eof),
            }
        }
    }

    /// Runs `actions`: confirmation, execution until one submits, the observations (tool
    /// results for the model's tool calls, user messages otherwise), then the exit, the
    /// interrupt or, for the model's actions, the limits.
    pub(super) async fn run_actions(&self, actions: &[Action], human: bool, ctx: &Context) -> Outcome {
        let commands: Vec<String> = actions.iter().map(|action| action.command.clone()).collect();
        let mut end = if human { None } else { self.confirm(&commands).await };
        let mut outputs = Vec::new();
        if end.is_none() {
            for action in actions {
                let output = self.env.execute(&action.command, None, None, ctx).await;
                if ctx.is_cancelled() {
                    end = Some(End::Aborted);
                    break;
                }
                let submission = output.submission();
                outputs.push(output);
                if let Some(submission) = submission {
                    end = Some(self.confirm_exit(submission).await);
                    break;
                }
            }
        }

        let mut outcome = self.finish_round(actions, &outputs, end, human);
        if !outcome.stop && !human {
            outcome.stop = !self.check_limits().await;
        }
        outcome
    }

    /// The observation of `action` as the model gets it: the result of its tool call, or a
    /// user message.
    fn observation_message(action: &Action, human: bool, content: &str) -> Message {
        match action.tool_call_id.as_deref().filter(|_| !human) {
            Some(id) => tool_result(id, content),
            None => user_message(content),
        }
    }

    /// Adds a round's observations and its ending to the trajectory.
    fn finish_round(&self, actions: &[Action], outputs: &[ExecOutput], end: Option<End>, human: bool) -> Outcome {
        let mut state = self.lock();
        let vars = self.template_vars(&state);
        let pad = !human && self.model.config.actions == ActionFormat::ToolCalls;
        let mut outcome = Outcome::default();
        for (index, action) in actions.iter().enumerate() {
            let output = match outputs.get(index) {
                Some(output) => output.clone(),
                None if pad => not_executed(),
                None => break,
            };
            let mut vars = vars.clone();
            vars["output"] = output.to_json();
            let content = match crate::template::render(&self.model.config.observation_template, &vars) {
                Ok(content) => content,
                Err(error) => {
                    let message = format!("observation_template: {error}");
                    state.exit(ExitStatus::Error, Some(message.clone()));
                    outcome.stop = true;
                    message
                }
            };
            let message = Self::observation_message(action, human, &content);
            self.push(&mut state, Entry::Observation { message, output });
            outcome.observations.push(content);
        }
        match end {
            Some(End::Interrupt { content, reason }) => {
                self.push(
                    &mut state,
                    Entry::Interrupt {
                        message: user_message(&content),
                        reason,
                    },
                );
                outcome.interrupt = Some(content);
                outcome.stop = true;
            }
            Some(End::Submitted(submission)) => {
                if state.trajectory.exit.is_none() {
                    state.trajectory.exit = Some(Exit {
                        status: ExitStatus::Submitted,
                        submission,
                        error: None,
                    });
                }
                outcome.stop = true;
            }
            Some(End::Exit(status)) => {
                state.exit(status, None);
                outcome.stop = true;
            }
            Some(End::Interrupted) => {
                state.interrupted = true;
                outcome.stop = true;
            }
            Some(End::Aborted) => outcome.stop = true,
            None => {}
        }
        if !outcome.stop {
            state.format_errors = 0;
        }
        self.save(&state);
        outcome
    }

    /// Records an answer (the `after_response` hook): counts the call and its cost, adds the
    /// answer, and parses its actions into the next round, or renders the format error the
    /// model gets instead.
    pub(super) fn record_answer(&self, answer: &AssistantMessage) {
        let mut state = self.lock();
        if matches!(answer.stop_reason, StopReason::Error | StopReason::Aborted) {
            state.last_error = Some(answer.clone());
            return;
        }
        state.last_error = None;
        state.round = None;
        if state.trajectory.exit.is_some() {
            self.stop.notify_one();
            return;
        }
        // A call the step or cost limit should have prevented: its round skipped the check.
        if self.run_limit(&state) == Some(ExitStatus::LimitsExceeded) {
            state.exit(ExitStatus::LimitsExceeded, None);
            self.save(&state);
            self.stop.notify_one();
            return;
        }
        state.n_calls += 1;
        let cost = if self.model.is_scripted() {
            usize::try_from(state.n_calls - 1)
                .ok()
                .and_then(|index| self.script_costs.get(index))
                .copied()
                .unwrap_or_default()
        } else {
            answer.usage.cost.total
        };
        state.cost += cost;
        if let Some(batch) = &self.batch {
            batch.add(cost);
        }

        let (actions, error) = match self.model.config.actions {
            ActionFormat::ToolCalls => toolcall_actions(answer),
            ActionFormat::Text => self.text_actions(answer),
        };
        let message = Message::Assistant(answer.clone());
        match error {
            Some(error) => {
                self.push(
                    &mut state,
                    Entry::Answer {
                        message,
                        actions: Vec::new(),
                        cost,
                    },
                );
                let mut vars = self.template_vars(&state);
                vars["error"] = json!(error.message);
                vars["stop_reason"] = serde_json::to_value(answer.stop_reason).unwrap_or(Json::Null);
                vars["has_tool_calls"] = json!(error.has_tool_calls);
                vars["actions"] = json!(error.found);
                let content = crate::template::render(&self.model.format_error_template, &vars)
                    .unwrap_or_else(|render| format!("format_error_template: {render}"));
                state.format_errors += 1;
                let limit = state.config.format_error_limit;
                if limit > 0 && limit <= state.format_errors {
                    state.exit(ExitStatus::FormatErrors, None);
                    self.stop.notify_one();
                }
                state.round = Some(Arc::new(Round {
                    actions: Vec::new(),
                    rejection: Some(Rejection {
                        content,
                        error: error.message,
                    }),
                    outcome: OnceCell::new(),
                }));
            }
            None => {
                self.push(
                    &mut state,
                    Entry::Answer {
                        message,
                        actions: actions.clone(),
                        cost,
                    },
                );
                state.round = Some(Arc::new(Round {
                    actions,
                    rejection: None,
                    outcome: OnceCell::new(),
                }));
            }
        }
        self.save(&state);
    }

    /// The text actions: exactly one match of `action_regex`.
    fn text_actions(&self, answer: &AssistantMessage) -> (Vec<Action>, Option<FormatError>) {
        let text = answer_text(answer);
        let found: Vec<String> = self
            .model
            .action_regex
            .captures_iter(&text)
            .map(|captures| {
                // The first group when there is one, else the whole match.
                captures
                    .get(1)
                    .or_else(|| captures.get(0))
                    .map_or("", |found| found.as_str())
                    .trim()
                    .to_owned()
            })
            .collect();
        if found.len() == 1 {
            let command = found.into_iter().next().unwrap_or_default();
            return (
                vec![Action {
                    command,
                    tool_call_id: None,
                }],
                None,
            );
        }
        let message = format!("Expected exactly 1 action, found {}.", found.len());
        (
            Vec::new(),
            Some(FormatError {
                message,
                has_tool_calls: false,
                found,
            }),
        )
    }

    /// Continues a run whose answer called no tool (the `on_yield` hook): the format error,
    /// or the observation of the text action. `None` ends the run.
    pub(super) async fn continue_after_answer(&self, ctx: &Context) -> Option<String> {
        let round = {
            let state = self.lock();
            if state.trajectory.exit.is_some() {
                return None;
            }
            state.round.clone()?
        };
        if let Some(rejection) = &round.rejection {
            if !self.check_limits().await {
                return None;
            }
            let mut state = self.lock();
            self.push(
                &mut state,
                Entry::FormatError {
                    message: user_message(&rejection.content),
                    error: rejection.error.clone(),
                },
            );
            self.save(&state);
            return Some(rejection.content.clone());
        }
        if self.model.config.actions != ActionFormat::Text || round.actions.is_empty() {
            return None;
        }
        let outcome = self.run_actions(&round.actions, false, ctx).await;
        if !outcome.stop {
            return outcome.observations.into_iter().next();
        }
        let mut state = self.lock();
        state.queue.extend(outcome.observations);
        state.queue.extend(outcome.interrupt);
        None
    }

    /// A call of the bash tool: the first call of a round runs every action of the round;
    /// each call returns its own observation, and whether the run ends.
    pub(super) async fn tool_call(&self, call_id: &str, ctx: &Context) -> (String, bool) {
        let Some(round) = self.lock().round.clone() else {
            return ("No action to run.".to_owned(), true);
        };
        if let Some(rejection) = &round.rejection {
            let mut state = self.lock();
            self.push(
                &mut state,
                Entry::FormatError {
                    message: tool_result(call_id, &rejection.content),
                    error: rejection.error.clone(),
                },
            );
            self.save(&state);
            return (rejection.content.clone(), state.trajectory.exit.is_some());
        }
        let outcome = round
            .outcome
            .get_or_init(|| async {
                let outcome = self.run_actions(&round.actions, false, ctx).await;
                if let Some(interrupt) = &outcome.interrupt {
                    self.lock().queue.push(interrupt.clone());
                }
                outcome
            })
            .await;
        let index = round
            .actions
            .iter()
            .position(|action| action.tool_call_id.as_deref() == Some(call_id));
        let observation = index
            .and_then(|index| outcome.observations.get(index))
            .cloned()
            .unwrap_or_default();
        (observation, outcome.stop)
    }
}

/// Why an answer did not parse.
#[derive(Debug)]
struct FormatError {
    message: String,
    has_tool_calls: bool,
    /// The text actions found.
    found: Vec<String>,
}

fn answer_text(answer: &AssistantMessage) -> String {
    answer
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The tool-call actions: every call must be `bash` with a `command`.
fn toolcall_actions(answer: &AssistantMessage) -> (Vec<Action>, Option<FormatError>) {
    let calls: Vec<_> = answer
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    if calls.is_empty() {
        return (
            Vec::new(),
            Some(FormatError {
                message: "No tool calls found in the response. Every response MUST include at least one tool call."
                    .to_owned(),
                has_tool_calls: false,
                found: Vec::new(),
            }),
        );
    }
    let mut actions = Vec::new();
    for call in calls {
        let mut message = String::new();
        if let Some(error) = &call.arguments_error {
            let _ = write!(message, "Error parsing tool call arguments: {error}.");
        }
        if call.name != super::BASH_TOOL_NAME {
            let _ = write!(message, "Unknown tool '{}'.", call.name);
        }
        let command = call.arguments.get("command");
        if command.is_none() {
            message.push_str("Missing 'command' argument in bash tool call.");
        }
        if !message.is_empty() {
            return (
                Vec::new(),
                Some(FormatError {
                    message: message.trim().to_owned(),
                    has_tool_calls: true,
                    found: Vec::new(),
                }),
            );
        }
        let command = command.map_or_else(String::new, |command| {
            command.as_str().map_or_else(|| command.to_string(), str::to_owned)
        });
        actions.push(Action {
            command,
            tool_call_id: Some(call.id.clone()),
        });
    }
    (actions, None)
}

/// The observation of an action that did not run, for the trajectory of an interrupted
/// round.
pub(super) fn unfinished_observation(shared: &Shared, state: &mut State, action: &Action) {
    let output = not_executed();
    let mut vars = shared.template_vars(state);
    vars["output"] = output.to_json();
    let content = crate::template::render(&shared.model.config.observation_template, &vars)
        .unwrap_or_else(|error| format!("observation_template: {error}"));
    let message = Shared::observation_message(action, false, &content);
    shared.push(state, Entry::Observation { message, output });
}
