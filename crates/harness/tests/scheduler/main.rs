//! The scheduler: phases and checkpoints, recovery and close, ownership and abort cascades,
//! structured waits, sleeps and inspection.
//!
//! Ported from upstream `test/harness-tasks*.test.ts`, `harness-ownership`,
//! `harness-structured`, `harness-cancellation-barrier`, `harness-sleep`, `harness-blocked`,
//! `harness-lifecycle` and `harness-inspect` where they apply to the kernel. Every case runs
//! against the memory backend and SQLite in a temp file; a crash is simulated by dropping
//! the Session and scheduler without closing them and reopening the same storage.

#![allow(clippy::unwrap_used, clippy::too_many_lines)] // Case functions are not `#[test]` functions.

#[path = "../support/mod.rs"]
mod support;

/// Runs each case against memory and SQLite in a file, on a multi-threaded runtime so
/// invocations really run beside the commit line.
macro_rules! cases {
    ($($case:ident),* $(,)?) => {
        mod memory {
            $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)] async fn $case() { super::$case(crate::Backend::Memory).await; })*
        }
        mod sqlite {
            $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)] async fn $case() { super::$case(crate::Backend::SqliteFile).await; })*
        }
    };
}

/// [`cases!`] with tokio's clock paused: timers fire when time is advanced. Memory only:
/// SQLite runs on its own thread, and a paused clock auto-advances while the runtime waits
/// for it, which would make sleeps due early.
macro_rules! paused_cases {
    ($($case:ident),* $(,)?) => {
        mod memory {
            $(#[tokio::test(start_paused = true)] async fn $case() { super::$case(crate::Backend::Memory).await; })*
        }
    };
}

mod inspect;
mod ownership;
mod recovery;
mod short_sleeps;
mod sleep;
mod tasks;
mod waits;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use delta_chord::context::Context;
use delta_chord::json::Json;
use delta_harness::documents::MigrateError;
use delta_harness::records::{ConversationId, ConversationOwnership, StorageWrite, TaskId, TaskRecord};
use delta_harness::scheduler::{
    AnyTask, MemoryCatalog, Next, Scheduler, SchedulerOptions, SchedulerSettings, SettledTask, TaskDefinition,
    TaskError, TaskRuntime, Wake,
};
use delta_harness::session::{ReportInfo, Session, SessionOptions, TaskOptions};
use delta_harness::storage::Storage;
use futures::future::BoxFuture;
use tokio::sync::watch;

pub(crate) use support::Backend;
use support::{Durable, TempDir};

pub(crate) fn ctx() -> Context {
    Context::background()
}

/// How long a case waits for a condition before failing.
const PATIENCE: Duration = Duration::from_secs(10);

/// Polls `check` until it holds.
pub(crate) async fn eventually(mut check: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + PATIENCE;
    while !check() {
        assert!(tokio::time::Instant::now() < deadline, "condition was not reached");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

// ─── Environment ────────────────────────────────────────────────────────────

/// Reported errors with what they came from.
pub(crate) type Reports = Arc<Mutex<Vec<(String, ReportInfo)>>>;

/// One storage, its catalog and settings, opened as often as a case needs.
pub(crate) struct Env {
    pub(crate) storage: Arc<dyn Storage>,
    pub(crate) catalog: Arc<MemoryCatalog>,
    pub(crate) reports: Reports,
    pub(crate) settings: Arc<Mutex<SchedulerSettings>>,
    pub(crate) wakes: Option<Arc<Mutex<Vec<Wake>>>>,
    /// Epoch ms on tokio's clock, so paused time moves it.
    start: tokio::time::Instant,
    _dir: Option<TempDir>,
}

/// The base of the test clock, an arbitrary epoch ms.
pub(crate) const EPOCH: i64 = 1_700_000_000_000;

impl Env {
    pub(crate) async fn new(backend: Backend, tasks: impl IntoIterator<Item = Arc<dyn AnyTask>>) -> Self {
        let fixture = backend.open().await;
        Self {
            storage: Arc::new(Durable(fixture.storage)),
            catalog: Arc::new(MemoryCatalog::new(tasks)),
            reports: Reports::default(),
            settings: Arc::new(Mutex::new(SchedulerSettings::default())),
            wakes: None,
            start: tokio::time::Instant::now(),
            _dir: fixture.dir,
        }
    }

    /// The test clock now.
    pub(crate) fn now(&self) -> i64 {
        clock_at(self.start)
    }

    pub(crate) fn set_settings(&self, settings: SchedulerSettings) {
        *self.settings.lock().unwrap() = settings;
    }

    /// Opens a Session and its scheduler (paused).
    pub(crate) async fn open(&self) -> Harness {
        let reports = self.reports.clone();
        let start = self.start;
        let options = SessionOptions {
            now: Arc::new(move || clock_at(start)),
            reporter: Some(Arc::new(move |error, info| {
                reports.lock().unwrap().push((error.to_string(), info.clone()));
            })),
            ..SessionOptions::default()
        };
        let session = Session::open(self.storage.clone(), options);
        let settings = self.settings.clone();
        let mut options = SchedulerOptions::new(self.catalog.clone());
        options.settings = Arc::new(move || *settings.lock().unwrap());
        if let Some(wakes) = &self.wakes {
            let wakes = wakes.clone();
            options.on_wake = Some(Arc::new(move |wake| wakes.lock().unwrap().push(wake)));
        }
        let scheduler = Scheduler::open(session.clone(), options, &ctx()).await.unwrap();
        Harness { session, scheduler }
    }

    /// Opens and resumes.
    pub(crate) async fn run(&self) -> Harness {
        let harness = self.open().await;
        harness.scheduler.resume().unwrap();
        harness
    }

    pub(crate) fn reported(&self) -> Vec<(String, ReportInfo)> {
        self.reports.lock().unwrap().clone()
    }

    /// The stored record of `id`.
    pub(crate) async fn stored(&self, id: TaskId) -> TaskRecord {
        self.storage.task(id, &ctx()).await.unwrap().unwrap()
    }

    /// Writes `record` straight to storage (states a crash leaves behind).
    pub(crate) async fn write(&self, record: TaskRecord) {
        self.storage
            .commit(vec![StorageWrite::Task(record)], &ctx())
            .await
            .unwrap();
    }
}

fn clock_at(start: tokio::time::Instant) -> i64 {
    EPOCH + i64::try_from(start.elapsed().as_millis()).unwrap()
}

/// An open Session with its scheduler.
pub(crate) struct Harness {
    pub(crate) session: Session,
    pub(crate) scheduler: Scheduler,
}

impl Harness {
    /// A new ownerless conversation.
    pub(crate) async fn conversation(&self) -> ConversationId {
        self.session
            .commit(&ctx(), |tx| {
                Box::pin(async move { Ok(tx.create_conversation(ConversationOwnership::Ownerless).await?.id) })
            })
            .await
            .unwrap()
    }

    /// A task of `definition` with `input`.
    pub(crate) async fn create<T: TaskDefinition>(
        &self,
        definition: &Arc<T>,
        input: T::Input,
        options: TaskOptions,
    ) -> TaskId {
        let definition = definition.clone();
        self.session
            .commit(&ctx(), move |tx| {
                Box::pin(async move { Ok(tx.create_task(&*definition, input, options).await?.id()) })
            })
            .await
            .unwrap()
    }

    /// A conversation-owned probe task of `conversation` with a null input.
    pub(crate) async fn start(&self, probe: &Arc<Probe>, conversation: ConversationId) -> TaskId {
        self.create(probe, Json::Null, TaskOptions::conversation(conversation))
            .await
    }

    /// The task's terminal record.
    pub(crate) async fn settled(&self, id: TaskId) -> SettledTask {
        let background = ctx();
        let wait = self.scheduler.wait_for_task(id, &background);
        tokio::time::timeout(PATIENCE, wait)
            .await
            .expect("the task settles")
            .unwrap()
    }

    /// The committed record of `id`.
    pub(crate) async fn record(&self, id: TaskId) -> TaskRecord {
        self.session
            .commit(&ctx(), move |tx| Box::pin(async move { tx.task(id).await }))
            .await
            .unwrap()
            .unwrap()
    }

    /// The committed record of `id` once `check` holds for it.
    pub(crate) async fn until(&self, id: TaskId, check: impl Fn(&TaskRecord) -> bool) -> TaskRecord {
        let deadline = tokio::time::Instant::now() + PATIENCE;
        loop {
            let record = self.record(id).await;
            if check(&record) {
                return record;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "task {id} never reached the state: {record:?}"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// Drops the Session and scheduler without closing them, as a crash: invocations stop
    /// where they are. Returns once the old Session's line ended.
    pub(crate) async fn crash(self) {
        let closed = self.session.closed();
        drop(self);
        tokio::time::timeout(PATIENCE, closed)
            .await
            .expect("the old Session ends");
    }

    pub(crate) async fn close(self) {
        self.scheduler.close(&ctx()).await.unwrap();
    }
}

// ─── Probe tasks ────────────────────────────────────────────────────────────

/// A phase or abort handler of a [`Probe`].
pub(crate) type Script =
    Arc<dyn for<'a> Fn(&'a TaskRuntime<Probe>, Json) -> BoxFuture<'a, Result<Next<Probe>, TaskError>> + Send + Sync>;

type Migration = Arc<dyn Fn(Json, Json, u32) -> Result<(Json, Json), MigrateError> + Send + Sync>;

/// A task with `Json` input and checkpoints whose handlers a case scripts. It starts at
/// `{"phase":"start"}`; its default phase completes with `null` and its default abort
/// handler ends `aborted` with reason `test`.
#[derive(Clone)]
pub(crate) struct Probe {
    name: String,
    version: u32,
    run: Script,
    abort: Script,
    migrate: Option<Migration>,
}

pub(crate) fn script<F>(f: F) -> Script
where
    F: for<'a> Fn(&'a TaskRuntime<Probe>, Json) -> BoxFuture<'a, Result<Next<Probe>, TaskError>>
        + Send
        + Sync
        + 'static,
{
    Arc::new(f)
}

impl Probe {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            version: 1,
            run: script(|_, _| Box::pin(async { Ok(Next::complete(Json::Null)) })),
            abort: script(|_, _| Box::pin(async { Ok(Next::abort(Some("test".to_owned()))) })),
            migrate: None,
        }
    }

    pub(crate) fn run<F>(mut self, f: F) -> Self
    where
        F: for<'a> Fn(&'a TaskRuntime<Probe>, Json) -> BoxFuture<'a, Result<Next<Probe>, TaskError>>
            + Send
            + Sync
            + 'static,
    {
        self.run = Arc::new(f);
        self
    }

    pub(crate) fn on_abort<F>(mut self, f: F) -> Self
    where
        F: for<'a> Fn(&'a TaskRuntime<Probe>, Json) -> BoxFuture<'a, Result<Next<Probe>, TaskError>>
            + Send
            + Sync
            + 'static,
    {
        self.abort = Arc::new(f);
        self
    }

    pub(crate) fn version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    pub(crate) fn migrate(
        mut self,
        f: impl Fn(Json, Json, u32) -> Result<(Json, Json), MigrateError> + Send + Sync + 'static,
    ) -> Self {
        self.migrate = Some(Arc::new(f));
        self
    }

    pub(crate) fn arc(self) -> Arc<Self> {
        Arc::new(self)
    }
}

impl TaskDefinition for Probe {
    type Input = Json;
    type Checkpoint = Json;
    type Result = Json;

    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> u32 {
        self.version
    }

    fn initial(&self, _input: &Json) -> Json {
        delta_chord::json!({ "phase": "start" })
    }

    fn run<'a>(
        &'a self,
        phase: Json,
        rt: &'a TaskRuntime<Self>,
        _ctx: &'a Context,
    ) -> impl Future<Output = Result<Next<Self>, TaskError>> + Send + 'a {
        (self.run)(rt, phase)
    }

    fn abort<'a>(
        &'a self,
        phase: Json,
        rt: &'a TaskRuntime<Self>,
        _ctx: &'a Context,
    ) -> impl Future<Output = Result<Next<Self>, TaskError>> + Send + 'a {
        (self.abort)(rt, phase)
    }

    fn can_migrate(&self) -> bool {
        self.migrate.is_some()
    }

    fn migrate(&self, input: Json, checkpoint: Json, from: u32) -> Result<(Json, Json), MigrateError> {
        match &self.migrate {
            Some(migrate) => migrate(input, checkpoint, from),
            None => Err(MigrateError::Unsupported { from }),
        }
    }
}

/// A probe as the catalog holds it.
pub(crate) fn erased(probe: &Arc<Probe>) -> Arc<dyn AnyTask> {
    probe.clone()
}

/// The checkpoint `{"phase": phase}`.
pub(crate) fn phase(name: &str) -> Json {
    delta_chord::json!({ "phase": name })
}

// ─── Coordination ───────────────────────────────────────────────────────────

/// A gate handlers wait at until a case opens it.
#[derive(Clone)]
pub(crate) struct Gate(Arc<watch::Sender<bool>>);

impl Gate {
    pub(crate) fn new() -> Self {
        Self(Arc::new(watch::Sender::new(false)))
    }

    pub(crate) fn open(&self) {
        self.0.send_replace(true);
    }

    pub(crate) async fn wait(&self) {
        let mut open = self.0.subscribe();
        let _ = open.wait_for(|open| *open).await;
    }
}

/// Counts calls.
#[derive(Clone, Default)]
pub(crate) struct Count(Arc<AtomicUsize>);

impl Count {
    pub(crate) fn hit(&self) -> usize {
        self.0.fetch_add(1, Ordering::SeqCst) + 1
    }

    pub(crate) fn get(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

/// An ordered log of what handlers did.
#[derive(Clone, Default)]
pub(crate) struct Log(Arc<Mutex<Vec<String>>>);

impl Log {
    pub(crate) fn push(&self, line: impl Into<String>) {
        self.0.lock().unwrap().push(line.into());
    }

    pub(crate) fn lines(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
}
