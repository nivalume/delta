//! The trajectory inspector: browse trajectory files step by step in the terminal.
//!
//! A step is an answer (or a command the user ran in human mode) with the entries after it;
//! the system prompt and the task are the first step. Keys: `l`/→ and `h`/← change the step,
//! `0` and `$` jump to the first and last, `j`/↓ and `k`/↑ scroll (PgDn/PgUp by a page), `L`
//! and `H` change the trajectory, `e` opens the step and `E` the whole file in `jless`, `r`
//! shows or hides thinking, `R` reloads the file and `q` quits.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use delta_ai::types::Message;
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Paragraph, Wrap};

use crate::agent::kind_title;
use crate::trajectory::{Entry, Trajectory, message_text, thinking_text};

/// Groups entries into the pages the inspector shows: a new step starts at each answer and
/// each command the user ran.
pub fn entries_to_steps(entries: &[Entry]) -> Vec<Vec<Entry>> {
    let mut steps = Vec::new();
    let mut current = Vec::new();
    for entry in entries {
        if matches!(entry, Entry::Answer { .. } | Entry::Human { .. }) && !current.is_empty() {
            steps.push(std::mem::take(&mut current));
        }
        current.push(entry.clone());
    }
    if !current.is_empty() {
        steps.push(current);
    }
    steps
}

/// The trajectory files under `path`: the file itself, or every `*.traj.json` below a
/// directory, sorted.
pub fn trajectory_files(path: &Path) -> Result<Vec<PathBuf>, String> {
    if path.is_file() {
        return Ok(vec![path.to_owned()]);
    }
    if !path.is_dir() {
        return Err(format!("Error: Path '{}' does not exist", path.display()));
    }
    let mut files = Vec::new();
    let mut pending = vec![path.to_owned()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().ends_with(".traj.json"))
            {
                files.push(path);
            }
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(format!("No trajectory files found in '{}'", path.display()));
    }
    Ok(files)
}

/// What a key asks the terminal loop to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    None,
    Quit,
    /// Open this file in `jless`; `temporary` files are deleted afterwards.
    Jless {
        path: PathBuf,
        temporary: bool,
    },
}

/// The inspector's state.
#[derive(Debug)]
pub struct Inspector {
    files: Vec<PathBuf>,
    trajectory: usize,
    loaded: Option<Trajectory>,
    steps: Vec<Vec<Entry>>,
    step: usize,
    scroll: u16,
    /// Lines the body had at the last render, and its height: to clamp scrolling.
    body_lines: u16,
    body_height: u16,
    pub show_reasoning: bool,
    /// A notice in the footer: errors, `Reloaded`.
    pub status: Option<String>,
}

impl Inspector {
    pub fn new(files: Vec<PathBuf>, show_reasoning: bool) -> Self {
        let mut inspector = Self {
            files,
            trajectory: 0,
            loaded: None,
            steps: Vec::new(),
            step: 0,
            scroll: 0,
            body_lines: 0,
            body_height: 0,
            show_reasoning,
            status: None,
        };
        inspector.load();
        inspector
    }

    /// Loads the current file.
    fn load(&mut self) {
        self.loaded = None;
        self.steps = Vec::new();
        self.step = 0;
        self.scroll = 0;
        let Some(path) = self.files.get(self.trajectory) else {
            return;
        };
        match Trajectory::load(path) {
            Ok(trajectory) => {
                self.steps = entries_to_steps(&trajectory.entries);
                self.loaded = Some(trajectory);
            }
            Err(error) => {
                let name = path
                    .file_name()
                    .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
                self.status = Some(format!("Error loading {name}: {error}"));
            }
        }
    }

    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn trajectory(&self) -> usize {
        self.trajectory
    }

    pub fn scroll(&self) -> u16 {
        self.scroll
    }

    fn set_step(&mut self, step: usize) {
        if self.steps.is_empty() {
            return;
        }
        let step = step.min(self.steps.len() - 1);
        if step != self.step {
            self.step = step;
            self.scroll = 0;
        }
    }

    fn set_trajectory(&mut self, trajectory: usize) {
        if self.files.is_empty() {
            return;
        }
        let trajectory = trajectory.min(self.files.len() - 1);
        if trajectory != self.trajectory {
            self.trajectory = trajectory;
            self.load();
        }
    }

    fn scroll_by(&mut self, delta: i32) {
        let max = self.body_lines.saturating_sub(self.body_height);
        let scrolled = (i32::from(self.scroll) + delta).clamp(0, i32::from(max));
        self.scroll = u16::try_from(scrolled).unwrap_or(0);
    }

    /// The title: the file and the step.
    pub fn title(&self) -> String {
        if self.steps.is_empty() {
            return "Trajectory Inspector - No Data".to_owned();
        }
        let name = self.files[self.trajectory]
            .file_name()
            .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
        format!(
            "Trajectory {}/{} - {name} - Step {}/{}",
            self.trajectory + 1,
            self.files.len(),
            self.step + 1,
            self.steps.len()
        )
    }

    /// Handles a key.
    pub fn handle_key(&mut self, key: KeyEvent) -> Command {
        if key.kind != KeyEventKind::Press {
            return Command::None;
        }
        self.status = None;
        let page = i32::from(self.body_height.max(1));
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => return Command::Quit,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return Command::Quit,
            KeyCode::Right | KeyCode::Char('l') => self.set_step(self.step + 1),
            KeyCode::Left | KeyCode::Char('h') => self.set_step(self.step.saturating_sub(1)),
            KeyCode::Char('0') | KeyCode::Home => self.set_step(0),
            KeyCode::Char('$') | KeyCode::End => self.set_step(usize::MAX),
            KeyCode::Down | KeyCode::Char('j') => self.scroll_by(15),
            KeyCode::Up | KeyCode::Char('k') => self.scroll_by(-15),
            KeyCode::PageDown | KeyCode::Char(' ') => self.scroll_by(page),
            KeyCode::PageUp => self.scroll_by(-page),
            KeyCode::Char('L') => self.set_trajectory(self.trajectory + 1),
            KeyCode::Char('H') => self.set_trajectory(self.trajectory.saturating_sub(1)),
            KeyCode::Char('r') => self.show_reasoning = !self.show_reasoning,
            KeyCode::Char('R') => {
                let step = self.step;
                self.load();
                if !self.steps.is_empty() {
                    self.step = step.min(self.steps.len() - 1);
                    self.status = Some("Reloaded".to_owned());
                }
            }
            KeyCode::Char('e') => {
                let Some(step) = self.steps.get(self.step) else {
                    self.status = Some("No messages to display".to_owned());
                    return Command::None;
                };
                let path = std::env::temp_dir().join(format!("mini-inspector-{}.json", uuid::Uuid::new_v4().simple()));
                let text = serde_json::to_string_pretty(step).unwrap_or_default();
                if let Err(error) = std::fs::write(&path, text) {
                    self.status = Some(format!("Could not write {}: {error}", path.display()));
                    return Command::None;
                }
                return Command::Jless { path, temporary: true };
            }
            KeyCode::Char('E') => {
                let Some(path) = self.files.get(self.trajectory) else {
                    self.status = Some("No trajectory to display".to_owned());
                    return Command::None;
                };
                return Command::Jless {
                    path: path.clone(),
                    temporary: false,
                };
            }
            _ => {}
        }
        Command::None
    }

    /// The model, the exit and the stats of the loaded trajectory.
    pub fn summary(&self) -> String {
        let Some(trajectory) = &self.loaded else {
            return String::new();
        };
        let model = format!(
            "{}/{}",
            trajectory.model["provider"].as_str().unwrap_or("?"),
            trajectory.model["id"].as_str().unwrap_or("?")
        );
        let exit = trajectory
            .exit
            .as_ref()
            .map_or_else(|| "running".to_owned(), |exit| exit.status.to_string());
        format!(
            "{model}  {exit}  {} calls  ${:.4}  {:.0}s",
            trajectory.stats.calls, trajectory.stats.cost, trajectory.stats.elapsed_seconds
        )
    }

    /// The body: each entry of the step under its kind, with thinking when shown; the exit
    /// after the last step.
    fn body(&self) -> Text<'static> {
        let Some(step) = self.steps.get(self.step) else {
            return Text::from("No trajectory loaded or empty trajectory");
        };
        let mut lines: Vec<Line<'static>> = Vec::new();
        let plain = |text: &str, lines: &mut Vec<Line<'static>>| {
            lines.extend(text.replace('\0', "").lines().map(|line| Line::from(line.to_owned())));
        };
        let note = |text: String| Line::from(Span::styled(text, Style::default().fg(Color::DarkGray)));
        for entry in step {
            lines.push(Line::from(Span::styled(
                format!(" {} ", kind_title(entry).to_uppercase()),
                kind_style(entry),
            )));
            let message = entry.message();
            if self.show_reasoning
                && let Some(thinking) = thinking_text(message)
            {
                lines.push(Line::from(Span::styled(
                    " THINKING ",
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                )));
                lines.extend(
                    thinking
                        .replace('\0', "")
                        .lines()
                        .map(|line| Line::from(Span::styled(line.to_owned(), Style::default().fg(Color::Gray)))),
                );
            }
            plain(&message_text(message), &mut lines);
            match entry {
                Entry::Answer {
                    message: Message::Assistant(answer),
                    cost,
                    ..
                } => {
                    lines.push(note(format!(
                        "${cost:.4}, {} input + {} output tokens, {}",
                        answer.usage.input,
                        answer.usage.output,
                        serde_json::to_value(answer.stop_reason)
                            .ok()
                            .and_then(|reason| reason.as_str().map(str::to_owned))
                            .unwrap_or_default()
                    )));
                }
                Entry::Observation { output, .. } => {
                    let mut text = format!("return code {}", output.returncode);
                    if !output.exception_info.is_empty() {
                        text.push_str(", ");
                        text.push_str(&output.exception_info);
                    }
                    lines.push(note(text));
                }
                Entry::FormatError { error, .. } => lines.push(note(error.clone())),
                Entry::Interrupt { reason, .. } => lines.push(note(
                    serde_json::to_value(reason)
                        .ok()
                        .and_then(|reason| reason.as_str().map(str::to_owned))
                        .unwrap_or_default(),
                )),
                _ => {}
            }
            lines.push(Line::default());
        }
        if self.step + 1 == self.steps.len()
            && let Some(exit) = self.loaded.as_ref().and_then(|trajectory| trajectory.exit.as_ref())
        {
            lines.push(Line::from(Span::styled(
                format!(" EXIT: {} ", exit.status),
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            )));
            if let Some(error) = &exit.error {
                plain(error, &mut lines);
            }
            plain(&exit.submission, &mut lines);
        }
        Text::from(lines)
    }

    /// Draws the inspector: title, summary, body, key hints.
    pub fn render(&mut self, frame: &mut Frame<'_>) {
        let [header, summary, body, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .areas(frame.area());
        frame.render_widget(
            Paragraph::new(self.title()).alignment(Alignment::Center).style(
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            header,
        );
        frame.render_widget(
            Paragraph::new(self.summary()).style(Style::default().fg(Color::Cyan)),
            summary,
        );
        let text = self.body();
        self.body_height = body.height;
        self.body_lines = wrapped_lines(&text, body);
        self.scroll = self.scroll.min(self.body_lines.saturating_sub(self.body_height));
        frame.render_widget(
            Paragraph::new(text).wrap(Wrap { trim: false }).scroll((self.scroll, 0)),
            body,
        );
        let hints = "l/→ Step++  h/← Step--  0 Step=0  $ Step=-1  j/↓ k/↑ scroll  L Traj++  H Traj--  e Jless  E Jless (all)  r Thinking  R Reload  q Quit";
        let footer_text = match &self.status {
            Some(status) => Line::from(Span::styled(status.clone(), Style::default().fg(Color::Yellow))),
            None => Line::from(Span::styled(hints, Style::default().fg(Color::DarkGray))),
        };
        frame.render_widget(Paragraph::new(footer_text), footer);
    }
}

fn kind_style(entry: &Entry) -> Style {
    let color = match entry {
        Entry::Answer { .. } => Color::Red,
        Entry::Task { .. } | Entry::Human { .. } => Color::Green,
        Entry::Observation { .. } => Color::Blue,
        Entry::System { .. } => Color::Yellow,
        Entry::FormatError { .. } | Entry::Interrupt { .. } => Color::LightRed,
    };
    Style::default().fg(Color::Black).bg(color).add_modifier(Modifier::BOLD)
}

/// About how many rows `text` takes wrapped to `area`'s width.
fn wrapped_lines(text: &Text<'_>, area: Rect) -> u16 {
    let width = usize::from(area.width.max(1));
    let rows: usize = text.lines.iter().map(|line| line.width().max(1).div_ceil(width)).sum();
    u16::try_from(rows).unwrap_or(u16::MAX)
}

/// Runs the inspector on the terminal until `q`.
pub fn run(files: Vec<PathBuf>, show_reasoning: bool) -> std::io::Result<()> {
    let mut inspector = Inspector::new(files, show_reasoning);
    let mut terminal = ratatui::init();
    let result = (|| -> std::io::Result<()> {
        loop {
            terminal.draw(|frame| inspector.render(frame))?;
            let Event::Key(key) = event::read()? else { continue };
            match inspector.handle_key(key) {
                Command::None => {}
                Command::Quit => return Ok(()),
                Command::Jless { path, temporary } => {
                    ratatui::restore();
                    let status = std::process::Command::new("jless").arg(&path).status();
                    if temporary {
                        let _ = std::fs::remove_file(&path);
                    }
                    terminal = ratatui::init();
                    terminal.clear()?;
                    if let Err(error) = status {
                        inspector.status = Some(if error.kind() == std::io::ErrorKind::NotFound {
                            "jless not found. Install with: `brew install jless` or `cargo install jless`".to_owned()
                        } else {
                            format!("jless failed: {error}")
                        });
                    }
                }
            }
        }
    })();
    ratatui::restore();
    let _ = std::io::stdout().flush();
    result
}

#[cfg(test)]
mod tests {
    use delta_ai::providers::faux::{faux_assistant_message, faux_text, faux_thinking, faux_tool_call};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use serde_json::json;

    use super::*;
    use crate::env::ExecOutput;
    use crate::trajectory::{Action, Exit, ExitStatus, InterruptReason, system_message, tool_result, user_message};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn answer(text: &str) -> Entry {
        Entry::Answer {
            message: Message::Assistant(faux_assistant_message(vec![faux_text(text)])),
            actions: Vec::new(),
            cost: 0.0,
        }
    }

    fn save(path: &Path, entries: Vec<Entry>, exit: Option<Exit>) {
        let trajectory = Trajectory {
            entries,
            exit,
            model: json!({ "provider": "faux", "id": "scripted" }),
            ..Trajectory::default()
        };
        trajectory.save(path).unwrap();
    }

    #[test]
    fn steps_start_at_answers_and_human_commands() {
        let entries = vec![
            Entry::System {
                message: system_message("s"),
            },
            Entry::Task {
                message: user_message("task"),
            },
            answer("a1"),
            Entry::Interrupt {
                message: user_message("no"),
                reason: InterruptReason::UserRejection,
            },
            Entry::Human {
                message: user_message("User command"),
                command: "ls".to_owned(),
            },
            Entry::Observation {
                message: user_message("o2"),
                output: ExecOutput::default(),
            },
            answer("a2"),
        ];
        let sizes: Vec<usize> = entries_to_steps(&entries).iter().map(Vec::len).collect();
        assert_eq!(sizes, [2, 2, 2, 1]);
    }

    #[test]
    fn keys_move_through_steps_and_files_and_the_screen_shows_them() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("a.traj.json");
        let call = faux_tool_call("bash", json!({ "command": "ls -la" }));
        let delta_ai::types::AssistantBlock::ToolCall(tool_call) = &call else {
            panic!("a tool call");
        };
        let id = tool_call.id.clone();
        save(
            &first,
            vec![
                Entry::System {
                    message: system_message("be brief"),
                },
                Entry::Answer {
                    message: Message::Assistant(faux_assistant_message(vec![
                        faux_thinking("hmm"),
                        faux_text("looking"),
                        call,
                    ])),
                    actions: vec![Action {
                        command: "ls -la".to_owned(),
                        tool_call_id: Some(id.clone()),
                    }],
                    cost: 0.25,
                },
                Entry::Observation {
                    message: tool_result(&id, "<returncode>0</returncode>"),
                    output: ExecOutput::default(),
                },
            ],
            Some(Exit {
                status: ExitStatus::Submitted,
                submission: "the patch".to_owned(),
                error: None,
            }),
        );
        let second = dir.path().join("b.traj.json");
        save(
            &second,
            vec![Entry::Task {
                message: user_message("other"),
            }],
            None,
        );
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let third = dir.path().join("sub/c.traj.json");
        std::fs::write(&third, json!({ "messages": [] }).to_string()).unwrap();
        assert_eq!(trajectory_files(dir.path()).unwrap(), [first, second, third]);

        let mut inspector = Inspector::new(trajectory_files(dir.path()).unwrap(), true);
        assert_eq!(inspector.title(), "Trajectory 1/3 - a.traj.json - Step 1/2");
        assert_eq!(inspector.summary(), "faux/scripted  submitted  0 calls  $0.0000  0s");
        assert_eq!(inspector.handle_key(key(KeyCode::Char('l'))), Command::None);
        assert_eq!(inspector.step(), 1);
        inspector.handle_key(key(KeyCode::Char('l')));
        assert_eq!(inspector.step(), 1, "clamped at the last step");

        let mut terminal = Terminal::new(TestBackend::new(60, 24)).unwrap();
        terminal.draw(|frame| inspector.render(frame)).unwrap();
        let screen = format!("{:?}", terminal.backend().buffer());
        for expected in [
            "Step 2/2",
            "ANSWER",
            "ls -la",
            "THINKING",
            "hmm",
            "$0.2500",
            "OBSERVATION",
            "<returncode>",
            "EXIT: submitted",
            "the patch",
        ] {
            assert!(screen.contains(expected), "{expected} in {screen}");
        }
        inspector.handle_key(key(KeyCode::Char('r')));
        terminal.draw(|frame| inspector.render(frame)).unwrap();
        assert!(!format!("{:?}", terminal.backend().buffer()).contains("THINKING"));

        inspector.handle_key(key(KeyCode::Char('0')));
        assert_eq!(inspector.step(), 0);
        inspector.handle_key(key(KeyCode::Char('$')));
        assert_eq!(inspector.step(), 1);
        inspector.handle_key(key(KeyCode::Char('L')));
        assert_eq!(inspector.title(), "Trajectory 2/3 - b.traj.json - Step 1/1");
        assert!(inspector.summary().contains("running"));
        inspector.handle_key(key(KeyCode::Char('L')));
        assert_eq!(inspector.title(), "Trajectory Inspector - No Data");
        let status = inspector.status.clone().unwrap();
        assert!(status.starts_with("Error loading c.traj.json"), "{status}");
        assert!(status.contains("not a delta-mini/1 trajectory"), "{status}");
        inspector.handle_key(key(KeyCode::Char('H')));
        inspector.handle_key(key(KeyCode::Char('H')));
        assert_eq!(inspector.trajectory(), 0);

        assert!(matches!(
            inspector.handle_key(key(KeyCode::Char('e'))),
            Command::Jless { temporary: true, .. }
        ));
        assert!(matches!(
            inspector.handle_key(key(KeyCode::Char('E'))),
            Command::Jless { temporary: false, .. }
        ));
        assert_eq!(inspector.handle_key(key(KeyCode::Char('q'))), Command::Quit);
    }

    #[test]
    fn scrolling_stays_within_the_text_and_reload_keeps_the_step() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("long.traj.json");
        let long = (0..100)
            .map(|line| format!("line {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        save(
            &path,
            vec![
                Entry::System {
                    message: system_message("first"),
                },
                answer(&long),
            ],
            None,
        );
        let mut inspector = Inspector::new(vec![path], true);
        inspector.handle_key(key(KeyCode::Char('l')));
        let mut terminal = Terminal::new(TestBackend::new(40, 11)).unwrap();
        terminal.draw(|frame| inspector.render(frame)).unwrap();
        for _ in 0..20 {
            inspector.handle_key(key(KeyCode::Char('j')));
        }
        // The heading, 100 lines, the cost note and a blank line in an 8-row body.
        assert_eq!(inspector.scroll(), 103 - 8);
        inspector.handle_key(key(KeyCode::Char('k')));
        assert_eq!(inspector.scroll(), 103 - 8 - 15);
        inspector.handle_key(key(KeyCode::Char('R')));
        assert_eq!((inspector.step(), inspector.status.as_deref()), (1, Some("Reloaded")));
    }
}
