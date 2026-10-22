//! The trajectory inspector (upstream's `run/utilities/inspector.py`): browse trajectory files
//! step by step in the terminal.
//!
//! A step is an assistant message (or a message carrying actions) with the messages after it,
//! as upstream groups them. Keys: `l`/→ and `h`/← change the step, `0` and `$` jump to the
//! first and last, `j`/↓ and `k`/↑ scroll (PgDn/PgUp by a page), `L` and `H` change the
//! trajectory, `e` opens the step and `E` the whole file in `jless`, `r` shows or hides
//! reasoning, `R` reloads the file and `q` quits.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Paragraph, Wrap};
use serde_json::Value as Json;

use crate::trajectory::content_string;

/// Groups messages into the pages the inspector shows (upstream's `_messages_to_steps`):
/// a new step starts at each assistant message and each message with actions.
pub fn messages_to_steps(messages: &[Json]) -> Vec<Vec<Json>> {
    let mut steps = Vec::new();
    let mut current = Vec::new();
    for message in messages {
        let has_actions = message["extra"]["actions"]
            .as_array()
            .is_some_and(|actions| !actions.is_empty());
        if has_actions || message["role"] == "assistant" {
            steps.push(std::mem::take(&mut current));
            current.push(message.clone());
        } else {
            current.push(message.clone());
        }
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
    steps: Vec<Vec<Json>>,
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

    /// Loads the current file: a trajectory document, or a bare list of messages.
    fn load(&mut self) {
        self.steps = Vec::new();
        self.step = 0;
        self.scroll = 0;
        let Some(path) = self.files.get(self.trajectory) else {
            return;
        };
        let loaded = std::fs::read_to_string(path)
            .map_err(|error| error.to_string())
            .and_then(|text| serde_json::from_str::<Json>(&text).map_err(|error| error.to_string()))
            .and_then(|data| match data {
                Json::Array(messages) => Ok(messages),
                Json::Object(mut object) => match object.remove("messages") {
                    Some(Json::Array(messages)) => Ok(messages),
                    _ => Err("Unrecognized trajectory format".to_owned()),
                },
                _ => Err("Unrecognized trajectory format".to_owned()),
            });
        match loaded {
            Ok(messages) => self.steps = messages_to_steps(&messages),
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

    /// The title upstream shows.
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

    /// The body: each message of the step under its role, with reasoning when shown.
    fn body(&self) -> Text<'static> {
        let Some(step) = self.steps.get(self.step) else {
            return Text::from("No trajectory loaded or empty trajectory");
        };
        let mut lines: Vec<Line<'static>> = Vec::new();
        for message in step {
            let role = message["role"]
                .as_str()
                .or_else(|| message["type"].as_str())
                .unwrap_or("unknown")
                .to_owned();
            lines.push(Line::from(Span::styled(
                format!(" {} ", role.to_uppercase()),
                role_style(&role),
            )));
            let content = content_string(message).replace('\0', "");
            lines.extend(content.lines().map(|line| Line::from(line.to_owned())));
            if self.show_reasoning
                && role.eq_ignore_ascii_case("assistant")
                && let Some(reasoning) = message["reasoning_content"].as_str().filter(|text| !text.is_empty())
            {
                lines.push(Line::from(Span::styled(
                    " REASONING ",
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                )));
                lines.extend(
                    reasoning
                        .replace('\0', "")
                        .lines()
                        .map(|line| Line::from(Span::styled(line.to_owned(), Style::default().fg(Color::Gray)))),
                );
            }
            lines.push(Line::default());
        }
        Text::from(lines)
    }

    /// Draws the inspector: title, body, key hints.
    pub fn render(&mut self, frame: &mut Frame<'_>) {
        let [header, body, footer] =
            Layout::vertical([Constraint::Length(1), Constraint::Min(1), Constraint::Length(1)]).areas(frame.area());
        frame.render_widget(
            Paragraph::new(self.title()).alignment(Alignment::Center).style(
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            header,
        );
        let text = self.body();
        self.body_height = body.height;
        self.body_lines = wrapped_lines(&text, body);
        self.scroll = self.scroll.min(self.body_lines.saturating_sub(self.body_height));
        frame.render_widget(
            Paragraph::new(text).wrap(Wrap { trim: false }).scroll((self.scroll, 0)),
            body,
        );
        let hints = "l/→ Step++  h/← Step--  0 Step=0  $ Step=-1  j/↓ k/↑ scroll  L Traj++  H Traj--  e Jless  E Jless (all)  r Reasoning  R Reload  q Quit";
        let footer_text = match &self.status {
            Some(status) => Line::from(Span::styled(status.clone(), Style::default().fg(Color::Yellow))),
            None => Line::from(Span::styled(hints, Style::default().fg(Color::DarkGray))),
        };
        frame.render_widget(Paragraph::new(footer_text), footer);
    }
}

fn role_style(role: &str) -> Style {
    let color = match role {
        "assistant" => Color::Red,
        "user" => Color::Green,
        "tool" => Color::Blue,
        "system" => Color::Yellow,
        "exit" => Color::Magenta,
        _ => Color::White,
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
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use serde_json::json;

    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn steps_start_at_assistant_messages_and_actions() {
        let messages = vec![
            json!({ "role": "system", "content": "s" }),
            json!({ "role": "user", "content": "task" }),
            json!({ "role": "assistant", "content": "a1" }),
            json!({ "role": "tool", "content": "o1" }),
            json!({ "role": "user", "content": "User command", "extra": { "actions": [{ "command": "ls" }] } }),
            json!({ "role": "user", "content": "o2" }),
            json!({ "role": "exit", "content": "" }),
        ];
        let steps = messages_to_steps(&messages);
        let sizes: Vec<usize> = steps.iter().map(Vec::len).collect();
        assert_eq!(sizes, [2, 2, 3]);
    }

    #[test]
    fn keys_move_through_steps_and_files_and_the_screen_shows_them() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, messages: Json| {
            let path = dir.path().join(name);
            std::fs::write(&path, json!({ "messages": messages }).to_string()).unwrap();
            path
        };
        let first = write(
            "a.traj.json",
            json!([
                { "role": "system", "content": "be brief" },
                { "role": "assistant", "content": "thinking", "reasoning_content": "hmm",
                  "tool_calls": [{ "function": { "arguments": "{\"command\": \"ls -la\"}" } }] },
                { "role": "tool", "content": "{\"returncode\": 0, \"output\": \"file\"}" },
            ]),
        );
        let second = write("b.traj.json", json!([{ "role": "user", "content": "other" }]));
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let third = dir.path().join("sub/c.traj.json");
        std::fs::write(&third, "not json").unwrap();
        assert_eq!(trajectory_files(dir.path()).unwrap(), [first, second, third]);

        let mut inspector = Inspector::new(trajectory_files(dir.path()).unwrap(), true);
        assert_eq!(inspector.title(), "Trajectory 1/3 - a.traj.json - Step 1/2");
        assert_eq!(inspector.handle_key(key(KeyCode::Char('l'))), Command::None);
        assert_eq!(inspector.step(), 1);
        inspector.handle_key(key(KeyCode::Char('l')));
        assert_eq!(inspector.step(), 1, "clamped at the last step");

        let mut terminal = Terminal::new(TestBackend::new(60, 20)).unwrap();
        terminal.draw(|frame| inspector.render(frame)).unwrap();
        let screen = format!("{:?}", terminal.backend().buffer());
        for expected in [
            "Step 2/2",
            "ASSISTANT",
            "ls -la",
            "REASONING",
            "hmm",
            "TOOL",
            "<returncode>",
        ] {
            assert!(screen.contains(expected), "{expected} in {screen}");
        }
        inspector.handle_key(key(KeyCode::Char('r')));
        terminal.draw(|frame| inspector.render(frame)).unwrap();
        assert!(!format!("{:?}", terminal.backend().buffer()).contains("REASONING"));

        inspector.handle_key(key(KeyCode::Char('0')));
        assert_eq!(inspector.step(), 0);
        inspector.handle_key(key(KeyCode::Char('$')));
        assert_eq!(inspector.step(), 1);
        inspector.handle_key(key(KeyCode::Char('L')));
        assert_eq!(inspector.title(), "Trajectory 2/3 - b.traj.json - Step 1/1");
        inspector.handle_key(key(KeyCode::Char('L')));
        assert_eq!(inspector.title(), "Trajectory Inspector - No Data");
        assert!(
            inspector
                .status
                .as_deref()
                .unwrap()
                .starts_with("Error loading c.traj.json")
        );
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
            .join("\n")
            + "\n";
        let messages = json!([
            { "role": "system", "content": "first" },
            { "role": "assistant", "content": long },
        ]);
        std::fs::write(&path, json!({ "messages": messages }).to_string()).unwrap();
        let mut inspector = Inspector::new(vec![path], true);
        inspector.handle_key(key(KeyCode::Char('l')));
        let mut terminal = Terminal::new(TestBackend::new(40, 10)).unwrap();
        terminal.draw(|frame| inspector.render(frame)).unwrap();
        for _ in 0..20 {
            inspector.handle_key(key(KeyCode::Char('j')));
        }
        assert_eq!(inspector.scroll(), 102 - 8);
        inspector.handle_key(key(KeyCode::Char('k')));
        assert_eq!(inspector.scroll(), 102 - 8 - 15);
        inspector.handle_key(key(KeyCode::Char('R')));
        assert_eq!((inspector.step(), inspector.status.as_deref()), (1, Some("Reloaded")));
    }
}
