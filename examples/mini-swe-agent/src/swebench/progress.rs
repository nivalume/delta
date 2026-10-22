//! The batch runner's progress (upstream's `RunBatchProgressManager`): an overall bar with cost
//! and ETA, the exit statuses so far, and each running instance's status, redrawn in place on
//! a terminal; a YAML report of the exit statuses after each instance.

use std::io::{IsTerminal as _, Write as _};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{Map, Value as Json};

use crate::model::GlobalModelStats;

#[derive(Debug, Default)]
struct Inner {
    /// Running instances: id, status, start.
    running: Vec<(String, String, Instant)>,
    /// Finished instances by exit status, in the order statuses first appeared.
    by_status: Map<String, Json>,
    /// Lines drawn last time, to draw over them.
    drawn: usize,
    spinner: usize,
}

/// Progress of a batch run.
#[derive(Debug)]
pub struct BatchProgress {
    total: usize,
    started: Instant,
    report: Option<PathBuf>,
    /// Redraw in place (stderr is a terminal); otherwise one line per finished instance.
    live: bool,
    inner: Mutex<Inner>,
}

fn shorten(text: &str, max: usize, left: bool) -> String {
    let count = text.chars().count();
    let shortened = if count <= max {
        text.to_owned()
    } else if left {
        let tail: String = text.chars().skip(count - (max - 3)).collect();
        format!("...{tail}")
    } else {
        let head: String = text.chars().take(max - 3).collect();
        format!("{head}...")
    };
    format!("{shortened:<max$}")
}

fn clock(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    format!("{}:{:02}:{:02}", seconds / 3600, seconds / 60 % 60, seconds % 60)
}

impl BatchProgress {
    pub fn new(total: usize, report: Option<PathBuf>) -> Self {
        Self {
            total,
            started: Instant::now(),
            report,
            live: std::io::stderr().is_terminal(),
            inner: Mutex::new(Inner::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn completed(&self) -> usize {
        self.lock()
            .by_status
            .values()
            .map(|ids| ids.as_array().map_or(0, Vec::len))
            .sum()
    }

    pub fn on_instance_start(&self, id: &str) {
        self.lock()
            .running
            .push((id.to_owned(), "Task initialized".to_owned(), Instant::now()));
    }

    pub fn update_instance_status(&self, id: &str, status: &str) {
        let mut inner = self.lock();
        if let Some(entry) = inner.running.iter_mut().find(|(running, _, _)| running == id) {
            status.clone_into(&mut entry.1);
        }
    }

    pub fn on_instance_end(&self, id: &str, exit_status: &str) {
        {
            let mut inner = self.lock();
            inner.running.retain(|(running, _, _)| running != id);
            let ids = inner
                .by_status
                .entry(exit_status.to_owned())
                .or_insert_with(|| Json::Array(Vec::new()));
            if let Json::Array(ids) = ids {
                ids.push(Json::String(id.to_owned()));
            }
        }
        if !self.live {
            eprintln!("{}/{} {id}: {exit_status}", self.completed(), self.total);
        }
        self.save_report();
    }

    /// The exit statuses as upstream's report holds them.
    pub fn overview(&self) -> Json {
        let mut overview = Map::new();
        overview.insert(
            "instances_by_exit_status".to_owned(),
            Json::Object(self.lock().by_status.clone()),
        );
        Json::Object(overview)
    }

    fn save_report(&self) {
        let Some(path) = &self.report else { return };
        match serde_saphyr::to_string(&self.overview()) {
            Ok(text) => {
                if let Err(error) = std::fs::write(path, text) {
                    eprintln!("Could not write {}: {error}", path.display());
                }
            }
            Err(error) => eprintln!("Could not write the report: {error}"),
        }
    }

    fn eta(&self, completed: usize) -> String {
        if completed == 0 {
            return String::new();
        }
        let remaining = self.started.elapsed().as_secs_f64() / completed as f64 * (self.total - completed) as f64;
        format!("eta: {}", clock(Duration::from_secs_f64(remaining)))
    }

    /// The lines of the live view.
    pub fn lines(&self, width: usize) -> Vec<String> {
        const SPINNER: [char; 8] = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'];
        let completed = self.completed();
        let (cost, _) = GlobalModelStats::get().totals();
        let mut inner = self.lock();
        inner.spinner = (inner.spinner + 1) % SPINNER.len();
        let spinner = SPINNER[inner.spinner];
        let percent = (completed * 100).checked_div(self.total).unwrap_or(100);
        let bar_width = width.saturating_sub(70).clamp(10, 40);
        let filled = (bar_width * completed).checked_div(self.total).unwrap_or(bar_width);
        let mut lines = vec![format!(
            "{spinner} Overall Progress (${cost:.2}) {}{} {completed}/{} {percent:>3}% {} {}",
            "━".repeat(filled),
            "╺".repeat(bar_width - filled),
            self.total,
            clock(self.started.elapsed()),
            self.eta(completed),
        )];
        if !inner.by_status.is_empty() {
            let mut statuses: Vec<(&String, &Json)> = inner.by_status.iter().collect();
            statuses.sort_by_key(|(_, ids)| std::cmp::Reverse(ids.as_array().map_or(0, Vec::len)));
            lines.push(format!("{:<24} {:>5}  Most recent instances", "Exit Status", "Count"));
            for (status, ids) in statuses {
                let ids = ids.as_array().cloned().unwrap_or_default();
                let recent: Vec<&str> = ids.iter().rev().filter_map(Json::as_str).collect();
                lines.push(format!(
                    "{:<24} {:>5}  {}",
                    shorten(status, 24, false),
                    ids.len(),
                    shorten(&recent.join(", "), 55, false).trim_end()
                ));
            }
        }
        for (id, status, started) in &inner.running {
            lines.push(format!(
                "{spinner} {} {} {}",
                shorten(id, 35, true),
                shorten(status, 30, false),
                clock(started.elapsed())
            ));
        }
        lines
    }

    /// Redraws the live view over the last one.
    pub fn draw(&self) {
        if !self.live {
            return;
        }
        let width = crossterm::terminal::size().map_or(100, |(columns, _)| usize::from(columns));
        let lines = self.lines(width);
        let previous = std::mem::replace(&mut self.lock().drawn, lines.len());
        let mut stderr = std::io::stderr().lock();
        if previous > 0 {
            let _ = crossterm::execute!(
                stderr,
                crossterm::cursor::MoveToPreviousLine(u16::try_from(previous).unwrap_or(u16::MAX)),
                crossterm::terminal::Clear(crossterm::terminal::ClearType::FromCursorDown)
            );
        }
        for line in lines {
            let line: String = line.chars().take(width.saturating_sub(1)).collect();
            let _ = writeln!(stderr, "{line}");
        }
        let _ = stderr.flush();
    }

    /// Every exit status and its instances.
    pub fn print_report(&self) {
        for (status, ids) in &self.lock().by_status {
            let ids = ids.as_array().cloned().unwrap_or_default();
            eprintln!("{status}: {}", ids.len());
            for id in ids.iter().filter_map(Json::as_str) {
                eprintln!("  {id}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_are_counted_and_reported() {
        let dir = tempfile::tempdir().unwrap();
        let report = dir.path().join("exit_statuses.yaml");
        let progress = BatchProgress::new(3, Some(report.clone()));
        progress.on_instance_start("a__1");
        progress.on_instance_start("b__2");
        progress.update_instance_status("b__2", "Step   2 ($0.10)");
        let lines = progress.lines(120);
        assert!(lines[0].contains("0/3"));
        assert!(
            lines
                .iter()
                .any(|line| line.contains("b__2") && line.contains("Step   2 ($0.10)"))
        );
        progress.on_instance_end("a__1", "Submitted");
        progress.on_instance_end("b__2", "LimitsExceeded");
        assert_eq!(progress.completed(), 2);
        let lines = progress.lines(120);
        assert!(lines[0].contains("2/3"));
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Submitted") && line.contains("a__1"))
        );
        let saved: Json = serde_saphyr::from_str(&std::fs::read_to_string(report).unwrap()).unwrap();
        assert_eq!(
            saved,
            serde_json::json!({ "instances_by_exit_status": { "Submitted": ["a__1"], "LimitsExceeded": ["b__2"] } })
        );
        assert_eq!(shorten("abcdefghij", 6, true), "...hij");
        assert_eq!(shorten("abcdefghij", 6, false), "abc...");
    }
}
