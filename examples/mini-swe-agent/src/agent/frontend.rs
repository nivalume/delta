//! What the interactive agent shows and asks: entries as they are added, its prompts, and
//! the answers. [`Terminal`] talks to a terminal; [`Scripted`] answers from a list, for
//! tests.

use std::collections::VecDeque;
use std::io::{IsTerminal as _, Write as _};
use std::path::PathBuf;
use std::sync::Mutex;

use crossterm::style::Stylize as _;

use super::Mode;
use crate::trajectory::{Entry, message_text};

/// One answer to a prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Input {
    Line(String),
    /// Ctrl+C at the prompt.
    Interrupted,
    /// Ctrl+D, or no more input.
    Eof,
}

/// What a prompt asks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Prompt {
    /// Whether to run the model's actions.
    Confirm { actions: usize },
    /// The agent wants to finish: a new task, or Enter to quit.
    ConfirmExit,
    /// After Ctrl+C: a comment for the model.
    Interrupted,
    /// Human mode: a command to run.
    Human,
    /// The task, when the command line gave none.
    Task,
    /// `New step limit: `.
    StepLimit,
    /// `New cost limit: `.
    CostLimit,
}

/// Something the agent tells the user.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Notice {
    /// `/h`: the commands and the current mode.
    Help {
        mode: Mode,
    },
    Switched {
        mode: Mode,
    },
    AlreadyIn {
        mode: Mode,
    },
    LimitsExceeded {
        step_limit: u64,
        cost_limit: f64,
        steps: u64,
        cost: f64,
    },
    /// A new step starts.
    Rule,
}

/// The interactive agent's terminal.
pub trait Frontend: Send + Sync {
    /// Shows an entry as the agent adds it; `step` and `cost` are the counts so far.
    fn show_entry(&self, entry: &Entry, step: u64, cost: f64);

    fn notice(&self, notice: Notice);

    /// Asks `prompt` and reads one line.
    fn read_line(&self, prompt: Prompt) -> Input;

    /// Reads several lines (`/m`).
    fn read_multiline(&self) -> Input;

    /// Whether a person can answer: new limits are asked only then.
    fn is_interactive(&self) -> bool;
}

/// A terminal: colored entries, prompts with history in the home directory.
pub struct Terminal {
    editor: Mutex<rustyline::DefaultEditor>,
    history: Option<PathBuf>,
}

impl std::fmt::Debug for Terminal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Terminal")
            .field("history", &self.history)
            .finish_non_exhaustive()
    }
}

impl Terminal {
    /// A terminal whose prompt history is kept in `history`.
    pub fn new(history: Option<PathBuf>) -> Result<Self, rustyline::error::ReadlineError> {
        let mut editor = rustyline::DefaultEditor::new()?;
        if let Some(history) = &history {
            let _ = editor.load_history(history);
        }
        Ok(Self {
            editor: Mutex::new(editor),
            history,
        })
    }

    fn print(text: &str) {
        let mut stdout = std::io::stdout().lock();
        let _ = write!(stdout, "{text}");
        let _ = stdout.flush();
    }

    fn read(&self, prompt: &str) -> Input {
        let mut editor = self.editor.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        match editor.readline(prompt) {
            Ok(line) => {
                if !line.trim().is_empty() {
                    let _ = editor.add_history_entry(line.as_str());
                    if let Some(history) = &self.history {
                        let _ = editor.save_history(history);
                    }
                }
                Input::Line(line)
            }
            Err(rustyline::error::ReadlineError::Interrupted) => Input::Interrupted,
            Err(_) => Input::Eof,
        }
    }
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Human => "human",
        Mode::Confirm => "confirm",
        Mode::Yolo => "yolo",
    }
}

/// An entry's kind as a heading: `format_error` is `Format error`.
pub fn kind_title(entry: &Entry) -> String {
    let mut title = entry.kind().replace('_', " ");
    if let Some(first) = title.get_mut(0..1) {
        first.make_ascii_uppercase();
    }
    title
}

impl Frontend for Terminal {
    fn show_entry(&self, entry: &Entry, step: u64, cost: f64) {
        let header = if matches!(entry, Entry::Answer { .. }) {
            format!(
                "\n{} (step {}, {}):\n",
                "delta-mini".bold(),
                step.to_string().bold(),
                format!("${cost:.2}").bold()
            )
            .red()
            .to_string()
        } else {
            format!("\n{}:\n", kind_title(entry).bold().green())
        };
        Self::print(&format!("{header}{}\n", message_text(entry.message())));
    }

    fn notice(&self, notice: Notice) {
        let text = match notice {
            Notice::Help { mode } => format!(
                "Current mode: {}\n{} to switch to {} mode (execute LM commands without confirmation)\n{} to switch to {} mode (ask for confirmation before executing LM commands)\n{} to switch to {} mode (execute commands issued by the user)\n{} to enter multiline comment\n",
                mode_name(mode).bold().green(),
                "/y".bold().green(),
                "yolo".bold().yellow(),
                "/c".bold().green(),
                "confirmation".bold().yellow(),
                "/u".bold().green(),
                "human".bold().yellow(),
                "/m".bold().green(),
            ),
            Notice::Switched { mode } => format!("Switched to {} mode.\n", mode_name(mode).bold().green()),
            Notice::AlreadyIn { mode } => format!("{}\n", format!("Already in {} mode.", mode_name(mode)).bold().red()),
            Notice::LimitsExceeded {
                step_limit,
                cost_limit,
                steps,
                cost,
            } => format!(
                "Limits exceeded. Limits: {step_limit} steps, ${cost_limit}.\nCurrent spend: {steps} steps, ${cost:.2}.\n"
            ),
            Notice::Rule => {
                let width = crossterm::terminal::size().map_or(80, |(columns, _)| usize::from(columns));
                format!("{}\n", "─".repeat(width).green())
            }
        };
        Self::print(&text);
    }

    fn read_line(&self, prompt: Prompt) -> Input {
        let arrow = format!("{} ", ">".bold().yellow());
        let text = match prompt {
            Prompt::Confirm { actions } => format!(
                "{} {}, {}, or {}\n",
                format!("Execute {actions} action(s)?").bold().yellow(),
                format!("{} to confirm", "Enter".bold()).green(),
                format!("type {} to reject", "comment".bold()).red(),
                format!("{} to show available commands", "/h".bold()).blue(),
            ),
            Prompt::ConfirmExit => format!(
                "{} {} or {} to quit ({} for commands)\n",
                "Agent wants to finish.".bold().yellow(),
                "Type new task".bold().green(),
                "Enter".bold(),
                "/h".bold(),
            ),
            Prompt::Interrupted => format!(
                "\n\n{} {} (/h for available commands)\n",
                "Interrupted.".bold().yellow(),
                "Type a comment/command".green(),
            ),
            Prompt::Human => String::new(),
            Prompt::Task => format!("{}\n", "What do you want to do?".bold().yellow()),
            Prompt::StepLimit => return self.read("New step limit: "),
            Prompt::CostLimit => return self.read("New cost limit: "),
        };
        Self::print(&text);
        self.read(&arrow)
    }

    fn read_multiline(&self) -> Input {
        Self::print(&format!("{}\n", "Multiline input: an empty line submits.".dark_grey()));
        let mut lines = Vec::new();
        loop {
            match self.read("... ") {
                Input::Line(line) if line.is_empty() && !lines.is_empty() => break,
                Input::Line(line) => lines.push(line),
                Input::Eof if !lines.is_empty() => break,
                other => return other,
            }
        }
        Input::Line(lines.join("\n"))
    }

    fn is_interactive(&self) -> bool {
        std::io::stdin().is_terminal()
    }
}

/// Answers from a list and records what was shown, for tests.
#[derive(Debug, Default)]
pub struct Scripted {
    answers: Mutex<VecDeque<Input>>,
    log: Mutex<Vec<String>>,
    interactive: bool,
}

impl Scripted {
    /// Answers `answers` in order, then end of input.
    pub fn new(answers: impl IntoIterator<Item = Input>, interactive: bool) -> Self {
        Self {
            answers: Mutex::new(answers.into_iter().collect()),
            log: Mutex::new(Vec::new()),
            interactive,
        }
    }

    /// Lines, in order.
    pub fn lines<const N: usize>(lines: [&str; N]) -> Self {
        Self::new(lines.map(|line| Input::Line(line.to_owned())), true)
    }

    /// What was shown and asked, one line each: `entry <kind> (step N): <text>`,
    /// `notice ...`, `prompt ...`.
    pub fn log(&self) -> Vec<String> {
        self.log
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Answers not used yet.
    pub fn remaining(&self) -> usize {
        self.answers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    fn record(&self, entry: String) {
        self.log
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(entry);
    }

    fn next(&self) -> Input {
        self.answers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
            .unwrap_or(Input::Eof)
    }
}

impl Frontend for Scripted {
    fn show_entry(&self, entry: &Entry, step: u64, _cost: f64) {
        self.record(format!(
            "entry {} (step {step}): {}",
            entry.kind(),
            message_text(entry.message())
        ));
    }

    fn notice(&self, notice: Notice) {
        if notice != Notice::Rule {
            self.record(format!("notice {notice:?}"));
        }
    }

    fn read_line(&self, prompt: Prompt) -> Input {
        self.record(format!("prompt {prompt:?}"));
        self.next()
    }

    fn read_multiline(&self) -> Input {
        self.record("prompt multiline".to_owned());
        self.next()
    }

    fn is_interactive(&self) -> bool {
        self.interactive
    }
}
