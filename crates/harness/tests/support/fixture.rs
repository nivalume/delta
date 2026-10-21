//! A Harness over one durable storage with a fake model, opened as often as a case needs.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use delta_chord::context::Context;
use delta_chord::json;
use delta_harness::agent::{AgentChange, Change, HarnessSettings, SettingsSource};
use delta_harness::compaction::{Compactor, TaskCompactor};
use delta_harness::documents::DocAddress;
use delta_harness::generation::{LiveDoc, LiveState};
use delta_harness::harness::{Conversation, EntryRange, Harness, HarnessOptions, RootOptions, Submission};
use delta_harness::model::ModelRef;
use delta_harness::records::{EntryRecord, ScanOrder, SubmissionRecord};
use delta_harness::registry::Registry;
use delta_harness::session::ReportInfo;
use delta_harness::storage::Storage;

use super::model::{FakeModel, PROVIDER};
use super::{Backend, Durable, TempDir};

/// The model every chat uses.
pub const MODEL_ID: &str = "model-1";
/// Its context window.
pub const CONTEXT_WINDOW: u64 = 200_000;
/// The base of the test clock, an arbitrary epoch ms.
pub const EPOCH: i64 = 1_700_000_000_000;
/// How long a case waits for a condition before failing.
pub const PATIENCE: Duration = Duration::from_secs(10);

/// Reported errors with what they came from.
pub type Reports = Arc<Mutex<Vec<(String, ReportInfo)>>>;

pub fn ctx() -> Context {
    Context::background()
}

/// One storage with the dependencies of every Harness opened over it.
pub struct Fixture {
    pub storage: Arc<dyn Storage>,
    pub model: FakeModel,
    pub registry: Registry,
    pub settings: Arc<Mutex<HarnessSettings>>,
    pub reports: Reports,
    pub compactor: Arc<dyn Compactor>,
    /// Epoch ms on tokio's clock, so paused time moves it.
    start: tokio::time::Instant,
    _dir: Option<TempDir>,
}

impl Fixture {
    pub async fn new(backend: Backend) -> Self {
        let fixture = backend.open().await;
        let settings = HarnessSettings {
            close_grace: Some(Duration::from_secs(2)),
            ..HarnessSettings::default()
        };
        Self {
            storage: Arc::new(Durable(fixture.storage)),
            model: FakeModel::new(MODEL_ID, CONTEXT_WINDOW),
            registry: Registry::new(),
            settings: Arc::new(Mutex::new(settings)),
            reports: Reports::default(),
            compactor: Arc::new(TaskCompactor),
            start: tokio::time::Instant::now(),
            _dir: fixture.dir,
        }
    }

    /// The test clock now.
    pub fn now(&self) -> i64 {
        clock_at(self.start)
    }

    /// Changes the live settings.
    pub fn settings(&self, change: impl FnOnce(&mut HarnessSettings)) {
        change(&mut self.settings.lock().unwrap());
    }

    pub fn options(&self) -> HarnessOptions {
        let mut options = HarnessOptions::new(Arc::new(self.model.clone()));
        options.registry = self.registry.clone();
        let settings = self.settings.clone();
        options.settings = SettingsSource::Dynamic(Arc::new(move || settings.lock().unwrap().clone()));
        options.compactor = self.compactor.clone();
        let start = self.start;
        options.clock = Arc::new(move || clock_at(start));
        let reports = self.reports.clone();
        options.on_report = Some(Arc::new(move |error, info| {
            reports.lock().unwrap().push((error.to_string(), info.clone()));
        }));
        options
    }

    /// Opens a Harness, paused.
    pub async fn open(&self) -> Harness {
        Harness::open(self.storage.clone(), self.options(), &ctx())
            .await
            .unwrap()
    }

    pub fn reported(&self) -> Vec<(String, ReportInfo)> {
        self.reports.lock().unwrap().clone()
    }
}

fn clock_at(start: tokio::time::Instant) -> i64 {
    EPOCH + i64::try_from(start.elapsed().as_millis()).unwrap()
}

/// The instructions of every chat: its only prompt section.
pub const INSTRUCTIONS: &str = "Be helpful.";

/// The fake model's reference.
pub fn model_ref() -> ModelRef {
    ModelRef::new(PROVIDER, MODEL_ID)
}

/// The root conversation, configured with the fake model and [`INSTRUCTIONS`].
pub async fn chat(harness: &Harness) -> Conversation {
    let options = RootOptions {
        agent: Some(AgentChange {
            model: Change::Set(model_ref()),
            instructions: Change::Set(INSTRUCTIONS.to_owned()),
            ..AgentChange::default()
        }),
        init: None,
    };
    harness.root(options, &ctx()).await.unwrap()
}

/// The submission's settled record.
pub async fn settled(submission: &Submission) -> SubmissionRecord {
    tokio::time::timeout(PATIENCE, submission.wait(&ctx()))
        .await
        .expect("the submission settles")
        .unwrap()
}

/// Waits until the conversation is idle.
pub async fn idle(conversation: &Conversation) {
    tokio::time::timeout(PATIENCE, conversation.wait_for_idle(&ctx()))
        .await
        .expect("the conversation goes idle")
        .unwrap();
}

/// Every entry visible from the conversation, oldest first.
pub async fn entries(conversation: &Conversation) -> Vec<EntryRecord> {
    let range = EntryRange {
        order: Some(ScanOrder::Ascending),
        ..EntryRange::default()
    };
    let mut all = Vec::new();
    let mut cursor = None;
    loop {
        let page = conversation.entries(range, 100, cursor, &ctx()).await.unwrap();
        all.extend(page.items);
        match page.next {
            Some(next) => cursor = Some(next),
            None => return all,
        }
    }
}

/// The entries' kinds.
pub fn kinds(entries: &[EntryRecord]) -> Vec<&str> {
    entries.iter().map(|entry| entry.kind.as_str()).collect()
}

/// The committed `delta.live` of the conversation.
pub async fn live(harness: &Harness, conversation: &Conversation) -> LiveState {
    let value = harness
        .snapshot(DocAddress::<LiveDoc>::new(conversation.id()), &ctx())
        .await
        .unwrap()
        .unwrap();
    json::from_json(&value).unwrap()
}

/// Polls `check` until it holds.
pub async fn eventually<F: Future<Output = bool>>(mut check: impl FnMut() -> F) {
    let deadline = tokio::time::Instant::now() + PATIENCE;
    while !check().await {
        assert!(tokio::time::Instant::now() < deadline, "condition was not reached");
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}
