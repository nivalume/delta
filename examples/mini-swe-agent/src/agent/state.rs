//! What the bash tool, the generation hooks and the driver share: the trajectory, the
//! counters, the mode and limits, and the steps upstream's agents take (confirm, execute,
//! observe, submit, check the limits).

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use delta_ai::types::{AssistantBlock, AssistantMessage, StopReason};
use delta_chord::context::Context;
use regex::Regex;
use serde_json::{Map, Value as Json, json};
use tokio::sync::{Notify, OnceCell};

use super::config::{AgentConfig, AgentKind, Mode};
use super::frontend::{Frontend, Input, Notice, Prompt};
use crate::env::{Environment, ExecOutput};
use crate::model::{ActionFormat, GlobalModelStats, ModelSettings, answer_cost, finish_reason};
use crate::trajectory::{self, Trajectory, exit_message, extra, observation_message};

/// One action of a step: a command, and the tool call that asked for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Action {
    pub(super) command: String,
    pub(super) tool_call_id: Option<String>,
}

/// The actions of the model's latest answer.
#[derive(Debug)]
pub(super) struct Round {
    pub(super) actions: Vec<Action>,
    /// The rendered format error, when the answer did not parse: nothing runs.
    pub(super) format_error: Option<String>,
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
        kind: &'static str,
    },
    Submitted(String),
    Exit(&'static str),
    /// Ctrl+C at a prompt: the driver asks for a comment.
    Interrupted,
    /// The run was aborted while a command ran.
    Aborted,
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
    pub(super) env: Arc<dyn Environment>,
    pub(super) frontend: Option<Arc<dyn Frontend>>,
    pub(super) output_path: Option<PathBuf>,
    /// Parts merged into the saved trajectory.
    pub(super) extra_info: Vec<Json>,
    pub(super) whitelist: Vec<Regex>,
    pub(super) started: Instant,
    pub(super) state: Mutex<State>,
    /// Asks the driver to abort the run: an exit was decided where no tool can end it.
    pub(super) stop: Notify,
    pub(super) on_step: Option<super::StepCallback>,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shared")
            .field("kind", &self.kind)
            .field("output_path", &self.output_path)
            .finish_non_exhaustive()
    }
}

/// An action that did not run (upstream's padding).
fn not_executed() -> ExecOutput {
    ExecOutput {
        output: String::new(),
        returncode: -1,
        exception_info: "action was not executed".to_owned(),
        extra: Map::new(),
    }
}

impl Shared {
    pub(super) fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The frontend of an interactive agent.
    fn interactive(&self) -> Option<&Arc<dyn Frontend>> {
        self.frontend.as_ref().filter(|_| self.kind == AgentKind::Interactive)
    }

    /// Adds `message` to the trajectory and shows it.
    pub(super) fn push(&self, state: &mut State, message: Json) {
        if let Some(frontend) = &self.frontend {
            frontend.show_message(&message, state.n_calls, state.cost);
        }
        state.trajectory.messages.push(message);
    }

    /// Adds a user message for the model (`extra.interrupt_type`) and queues it.
    pub(super) fn push_user(&self, state: &mut State, content: String, kind: &str) {
        let message = trajectory::message("user", &content, extra([("interrupt_type", json!(kind))]));
        self.push(state, message);
        state.queue.push(content);
    }

    /// Ends the run with `status`: an exit message whose content is the status.
    pub(super) fn exit(&self, state: &mut State, status: &str, more: Map<String, Json>) {
        self.push(state, exit_message(status, status, "", more));
    }

    pub(super) fn exited(&self) -> bool {
        self.lock().trajectory.exit().is_some()
    }

    /// Upstream's `get_template_vars`: the agent config, the environment's and the model's
    /// variables, the counters and the run's variables.
    pub(super) fn template_vars(&self, state: &State) -> Json {
        crate::template::vars([
            state.config.dump(self.kind),
            self.env.template_vars(),
            self.model.dump(),
            json!({
                "n_model_calls": state.n_calls,
                "model_cost": state.cost,
                "elapsed_seconds": self.started.elapsed().as_secs(),
            }),
            state.extra_vars.clone(),
        ])
    }

    /// The trajectory document.
    pub(super) fn serialize(&self, state: &State) -> Json {
        let parts = [self.model.serialize(), self.env.serialize()]
            .into_iter()
            .chain(self.extra_info.iter().cloned());
        state
            .trajectory
            .serialize(state.cost, state.n_calls, &state.config.serialize(self.kind), parts)
    }

    /// Saves the trajectory when the agent has an output path.
    pub(super) fn save(&self, state: &State) {
        if let Some(path) = &self.output_path
            && let Err(error) = trajectory::save_json(path, &self.serialize(state))
        {
            eprintln!("Could not save the trajectory to {}: {error}", path.display());
        }
    }

    /// The limit the next model call would exceed: `LimitsExceeded` (steps, cost) or
    /// `TimeExceeded` (wall time).
    fn exceeded(&self, state: &State) -> Option<&'static str> {
        let config = &state.config;
        if (config.step_limit > 0 && config.step_limit <= state.n_calls)
            || (config.cost_limit > 0.0 && config.cost_limit <= state.cost)
        {
            return Some("LimitsExceeded");
        }
        let limit = config.wall_time_limit_seconds;
        (limit > 0 && limit <= self.started.elapsed().as_secs()).then_some("TimeExceeded")
    }

    /// Whether another model call may happen. Over the step or cost limit, an interactive
    /// agent with a person at the terminal asks for new limits; otherwise (and over the wall
    /// time limit, which new limits cannot lift) the run exits.
    pub(super) async fn check_limits(&self) -> bool {
        let status = {
            let state = self.lock();
            if state.trajectory.exit().is_some() {
                return false;
            }
            match self.exceeded(&state) {
                None => return true,
                Some(status) => status,
            }
        };
        if status == "LimitsExceeded"
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
                if self.exceeded(&state).is_none() {
                    return true;
                }
            }
        }
        let mut state = self.lock();
        self.exit(&mut state, status, Map::new());
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

    /// Asks `prompt` and handles the slash commands (upstream's
    /// `_prompt_and_handle_slash_commands`): `/h` shows help and asks again, `/m` reads
    /// several lines, a mode command switches the mode (or, when already in it, asks again).
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

    /// Whether the user must confirm `commands` (upstream's `_should_ask_confirmation`).
    fn needs_confirmation(&self, commands: &[String]) -> bool {
        self.interactive().is_some()
            && self.lock().config.mode == Mode::Confirm
            && commands
                .iter()
                .any(|command| !self.whitelist.iter().any(|pattern| pattern.is_match(command)))
    }

    /// Upstream's `_ask_confirmation_or_interrupt`.
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
                    kind: "UserRejection",
                }),
                comment => Some(End::Interrupt {
                    content: format!(
                        "Commands not executed. The user rejected your commands with the following message: {comment}"
                    ),
                    kind: "UserRejection",
                }),
            },
            Input::Interrupted => Some(End::Interrupted),
            Input::Eof => Some(End::Exit("EOFError")),
        }
    }

    /// The agent wants to finish: upstream's `_check_for_new_task_or_submit`.
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
                            kind: "UserInterruption",
                        };
                    }
                    "/y" | "/c" => {}
                    task => {
                        return End::Interrupt {
                            content: format!("The user added a new task: {task}"),
                            kind: "UserNewTask",
                        };
                    }
                },
                Input::Interrupted => return End::Interrupted,
                Input::Eof => return End::Exit("EOFError"),
            }
        }
    }

    /// Runs `actions`: confirmation, execution until one submits, the observations (tool
    /// messages for the model's tool calls, user messages otherwise), then the exit, the
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

    /// Adds a round's observations and its ending to the trajectory.
    fn finish_round(&self, actions: &[Action], outputs: &[ExecOutput], end: Option<End>, human: bool) -> Outcome {
        let mut state = self.lock();
        let vars = self.template_vars(&state);
        let pad = !human && self.model.class.format == ActionFormat::Toolcall;
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
                    let message = error.to_string();
                    self.exit(&mut state, "TemplateError", extra([("exception_str", json!(message))]));
                    outcome.stop = true;
                    message
                }
            };
            let id = if human { None } else { action.tool_call_id.as_deref() };
            if state.trajectory.exit().is_none() {
                self.push(&mut state, observation_message(&content, &output, id));
            }
            outcome.observations.push(content);
        }
        match end {
            Some(End::Interrupt { content, kind }) => {
                let message = trajectory::message("user", &content, extra([("interrupt_type", json!(kind))]));
                self.push(&mut state, message);
                outcome.interrupt = Some(content);
                outcome.stop = true;
            }
            Some(End::Submitted(submission)) => {
                self.push(
                    &mut state,
                    exit_message(&submission, "Submitted", &submission, Map::new()),
                );
                outcome.stop = true;
            }
            Some(End::Exit(status)) => {
                let more = extra([("exception_str", json!(""))]);
                self.push(&mut state, exit_message("", status, "", more));
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

    /// Records an answer (the `after_response` hook): counts the call and its cost, parses
    /// its actions into the next round, and adds the assistant message, or the format error
    /// upstream adds instead.
    pub(super) fn record_answer(&self, answer: &AssistantMessage) {
        let mut state = self.lock();
        if matches!(answer.stop_reason, StopReason::Error | StopReason::Aborted) {
            state.last_error = Some(answer.clone());
            return;
        }
        state.last_error = None;
        state.round = None;
        if state.trajectory.exit().is_some() {
            self.stop.notify_one();
            return;
        }
        // A call the step or cost limit should have prevented: its round skipped the check.
        if self.exceeded(&state) == Some("LimitsExceeded") {
            self.exit(&mut state, "LimitsExceeded", Map::new());
            self.save(&state);
            self.stop.notify_one();
            return;
        }
        state.n_calls += 1;
        let cost = if self.model.class.deterministic {
            self.model.deterministic_cost((state.n_calls - 1) as usize)
        } else {
            answer_cost(answer)
        };
        if cost <= 0.0 && !self.model.class.deterministic && self.model.config.cost_tracking != "ignore_errors" {
            let message = format!(
                "Error calculating cost for model {}: the call reported no cost, perhaps the model has no price in delta-ai's catalog? You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or globally with export MSWEA_COST_TRACKING='ignore_errors'.",
                self.model.config.model_name
            );
            let more = extra([("exception_str", json!(message))]);
            self.push(&mut state, exit_message(&message, "RuntimeError", "", more));
            self.save(&state);
            self.stop.notify_one();
            return;
        }
        state.cost += cost;
        if let Err(message) = GlobalModelStats::get().add(cost) {
            let more = extra([("exception_str", json!(message))]);
            self.push(&mut state, exit_message(&message, "RuntimeError", "", more));
            self.save(&state);
            self.stop.notify_one();
            return;
        }

        let response = serde_json::to_value(answer).unwrap_or(Json::Null);
        let finish = finish_reason(answer.stop_reason);
        let (actions, error) = match self.model.class.format {
            ActionFormat::Toolcall => toolcall_actions(answer),
            ActionFormat::Textbased => self.textbased_actions(answer),
        };
        match error {
            Some(error) => {
                let mut vars = json!({ "error": error.message, "finish_reason": finish });
                match self.model.class.format {
                    ActionFormat::Toolcall => {
                        vars["actions"] = json!([]);
                        vars["has_tool_calls"] = json!(error.has_tool_calls);
                    }
                    ActionFormat::Textbased => vars["actions"] = json!(error.found),
                }
                let content = crate::template::render(&self.model.format_error_template, &vars)
                    .unwrap_or_else(|render| render.to_string());
                let mut fields = extra([("interrupt_type", json!("FormatError"))]);
                if self.model.class.format == ActionFormat::Textbased {
                    fields.insert("n_actions".to_owned(), json!(error.found.len()));
                    fields.insert("model_response".to_owned(), json!(answer_text(answer)));
                }
                fields.insert("cost".to_owned(), json!(cost));
                fields.insert("response".to_owned(), response);
                let message = trajectory::message("user", &content, fields);
                self.push(&mut state, message);
                state.format_errors += 1;
                let max = state.config.max_consecutive_format_errors;
                if max > 0 && max <= state.format_errors {
                    self.exit(&mut state, "RepeatedFormatError", Map::new());
                    self.stop.notify_one();
                }
                state.round = Some(Arc::new(Round {
                    actions: Vec::new(),
                    format_error: Some(content),
                    outcome: OnceCell::new(),
                }));
            }
            None => {
                let listed: Vec<Json> = actions
                    .iter()
                    .map(|action| match &action.tool_call_id {
                        Some(id) => json!({ "command": action.command, "tool_call_id": id }),
                        None => json!({ "command": action.command }),
                    })
                    .collect();
                let fields = extra([
                    ("actions", Json::Array(listed)),
                    ("response", response),
                    ("cost", json!(cost)),
                    ("timestamp", json!(trajectory::timestamp())),
                ]);
                let message = trajectory::assistant_message(answer, fields);
                self.push(&mut state, message);
                state.round = Some(Arc::new(Round {
                    actions,
                    format_error: None,
                    outcome: OnceCell::new(),
                }));
            }
        }
        self.save(&state);
    }

    /// The text-based actions: exactly one match of `action_regex` (upstream's
    /// `parse_regex_actions`).
    fn textbased_actions(&self, answer: &AssistantMessage) -> (Vec<Action>, Option<FormatError>) {
        let text = answer_text(answer);
        let found: Vec<String> = self
            .model
            .action_regex
            .captures_iter(&text)
            .map(|captures| {
                // `re.findall`: the group when there is one, else the whole match.
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
    /// or the observation of the text-based action. `None` ends the run.
    pub(super) async fn continue_after_answer(&self, ctx: &Context) -> Option<String> {
        let round = {
            let state = self.lock();
            if state.trajectory.exit().is_some() {
                return None;
            }
            state.round.clone()?
        };
        if let Some(error) = &round.format_error {
            return self.check_limits().await.then(|| error.clone());
        }
        if self.model.class.format != ActionFormat::Textbased || round.actions.is_empty() {
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
        if let Some(error) = &round.format_error {
            return (error.clone(), self.exited());
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
    /// The text-based actions found.
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

/// The tool-call actions (upstream's `parse_toolcall_actions`): every call must be `bash`
/// with a `command`.
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
        if call.name != "bash" {
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
