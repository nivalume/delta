//! `mini-extra swebench`: every selected instance in its own container, several at once;
//! each trajectory in `<output>/<id>/<id>.traj.json`, the patches in `<output>/preds.json`.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Map, Value as Json, json};

use super::{BatchProgress, filter_instances, instance_environment, load_dataset};
use crate::agent::{AgentKind, StepCallback};
use crate::cli::{BuildOptions, build_agent, credential_store, load_config, with_model};
use crate::config::Overrides;
use crate::model::{BatchStats, MiniModelClient, ModelSettings};
use crate::trajectory::{ExitStatus, now_millis};

/// Run delta-mini on SWE-bench instances in batch mode.
#[derive(Clone, Debug, clap::Args)]
pub struct SwebenchArgs {
    /// SWE-bench subset to use or path to a dataset.
    #[arg(long, default_value = "lite")]
    pub subset: String,
    /// Dataset split.
    #[arg(long, default_value = "dev")]
    pub split: String,
    /// Slice specification (e.g., '0:5' for first 5 instances).
    #[arg(long = "slice", default_value = "")]
    pub slice: String,
    /// Filter instance IDs by regex.
    #[arg(long = "filter", default_value = "")]
    pub filter: String,
    /// Shuffle instances.
    #[arg(long)]
    pub shuffle: bool,
    /// Output directory.
    #[arg(short = 'o', long, default_value = "")]
    pub output: String,
    /// Number of instances processed at once.
    #[arg(short = 'w', long, default_value_t = 1)]
    pub workers: usize,
    /// Model as provider/model-id, or a model id one provider has.
    #[arg(short = 'm', long)]
    pub model: Option<String>,
    /// Redo existing instances.
    #[arg(long = "redo-existing")]
    pub redo_existing: bool,
    /// Config files and key.path=value pairs; default `swebench.yaml`. Setting it replaces the
    /// default, so name it too: `-c swebench.yaml -c key=value`.
    #[arg(short = 'c', long = "config")]
    pub config: Vec<String>,
    /// Stop starting model calls once the batch has spent this many USD; 0: no limit.
    #[arg(long = "total-cost-limit", default_value_t = 0.0)]
    pub total_cost_limit: f64,
    /// Stop starting model calls once the batch has made this many; 0: no limit.
    #[arg(long = "total-call-limit", default_value_t = 0)]
    pub total_call_limit: u64,
    /// The Hugging Face datasets server.
    #[arg(long = "datasets-server")]
    pub datasets_server: Option<String>,
}

/// What a batch run did.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchSummary {
    /// Instances run (after filtering and skipping).
    pub instances: usize,
    /// `instances_by_exit_status`.
    pub overview: Json,
}

static PREDS_LOCK: Mutex<()> = Mutex::new(());

fn read_preds(path: &Path) -> Map<String, Json> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<Json>(&text).ok())
        .and_then(|data| match data {
            Json::Object(object) => Some(object),
            _ => None,
        })
        .unwrap_or_default()
}

/// Records an instance's patch in `preds.json`, in the shape SWE-bench's evaluation reads.
pub fn preds_update(path: &Path, instance_id: &str, model_name: &str, patch: &str) -> std::io::Result<()> {
    let _guard = PREDS_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut preds = read_preds(path);
    preds.insert(
        instance_id.to_owned(),
        json!({ "model_name_or_path": model_name, "instance_id": instance_id, "model_patch": patch }),
    );
    crate::trajectory::save_json(path, &Json::Object(preds))
}

/// Removes an instance from `preds.json`.
pub fn preds_remove(path: &Path, instance_id: &str) -> std::io::Result<()> {
    let _guard = PREDS_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if !path.exists() {
        return Ok(());
    }
    let mut preds = read_preds(path);
    if preds.remove(instance_id).is_some() {
        crate::trajectory::save_json(path, &Json::Object(preds))?;
    }
    Ok(())
}

/// Appends timestamped lines to `delta-mini.log`.
#[derive(Debug)]
struct Log(Mutex<Option<std::fs::File>>);

impl Log {
    fn open(path: &Path) -> Self {
        let file = std::fs::OpenOptions::new().create(true).append(true).open(path).ok();
        Self(Mutex::new(file))
    }

    fn line(&self, level: &str, message: &str) {
        let mut file = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(file) = file.as_mut() {
            let _ = writeln!(file, "{:.3} {level} {message}", now_millis() as f64 / 1000.0);
        }
    }
}

/// Runs the batch.
pub async fn run_batch(args: SwebenchArgs) -> Result<BatchSummary, String> {
    let output = PathBuf::from(if args.output.is_empty() { "." } else { &args.output });
    std::fs::create_dir_all(&output).map_err(|error| format!("Could not create {}: {error}", output.display()))?;
    eprintln!("Results will be saved to {}", output.display());
    let log = Arc::new(Log::open(&output.join("delta-mini.log")));

    eprintln!("Loading dataset {}, split {}...", args.subset, args.split);
    let instances = load_dataset(&args.subset, &args.split, args.datasets_server.as_deref())
        .await
        .map_err(|error| error.to_string())?;
    let mut instances = filter_instances(instances, &args.filter, &args.slice, args.shuffle)?;
    let preds_path = output.join("preds.json");
    if !args.redo_existing && preds_path.exists() {
        let existing = read_preds(&preds_path);
        eprintln!("Skipping {} existing instances", existing.len());
        instances.retain(|instance| !existing.contains_key(instance["instance_id"].as_str().unwrap_or_default()));
    }
    eprintln!("Running on {} instances...", instances.len());

    let mut config = load_config(&args.config, "swebench.yaml")?;
    with_model(Overrides::default(), args.model.as_deref()).apply(&mut config);
    // The model must resolve before any container starts.
    let settings = ModelSettings::new(&config["model"]).map_err(|error| error.to_string())?;
    let (client, _) = MiniModelClient::new(&settings, &config["providers"], Some(credential_store()))
        .map_err(|error| error.to_string())?;
    let model_name = format!("{}/{}", client.resolved().provider, client.resolved().id);

    let report = output.join(format!("exit_statuses_{}.yaml", now_millis()));
    let stats = Arc::new(BatchStats::new(args.total_cost_limit, args.total_call_limit));
    let progress = Arc::new(BatchProgress::new(instances.len(), Some(report), Arc::clone(&stats)));
    let cancelled = Arc::new(AtomicBool::new(false));
    let interrupts = {
        let cancelled = Arc::clone(&cancelled);
        tokio::spawn(async move {
            while tokio::signal::ctrl_c().await.is_ok() {
                if cancelled.swap(true, Ordering::SeqCst) {
                    std::process::exit(130);
                }
                eprintln!("Cancelling all pending jobs. Press ^C again to exit immediately.");
            }
        })
    };
    let drawing = {
        let progress = Arc::clone(&progress);
        tokio::spawn(async move {
            loop {
                progress.draw();
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
        })
    };

    let workers = Arc::new(tokio::sync::Semaphore::new(args.workers.max(1)));
    let config = Arc::new(config);
    let count = instances.len();
    let mut tasks = tokio::task::JoinSet::new();
    for instance in instances {
        let workers = Arc::clone(&workers);
        let cancelled = Arc::clone(&cancelled);
        let progress = Arc::clone(&progress);
        let config = Arc::clone(&config);
        let log = Arc::clone(&log);
        let output = output.clone();
        let model_name = model_name.clone();
        let stats = Arc::clone(&stats);
        tasks.spawn(async move {
            let Ok(_permit) = workers.acquire_owned().await else {
                return;
            };
            if cancelled.load(Ordering::SeqCst) {
                return;
            }
            let batch = Batch {
                output: &output,
                config: &config,
                model_name: &model_name,
                progress: &progress,
                log: &log,
                stats: &stats,
            };
            process_instance(&instance, &batch).await;
        });
    }
    while let Some(joined) = tasks.join_next().await {
        if let Err(error) = joined {
            log.line("ERROR", &format!("An instance's task failed: {error}"));
        }
    }
    drawing.abort();
    interrupts.abort();
    progress.draw();
    Ok(BatchSummary {
        instances: count,
        overview: progress.overview(),
    })
}

/// What every instance of a batch shares.
struct Batch<'a> {
    output: &'a Path,
    config: &'a Json,
    model_name: &'a str,
    progress: &'a Arc<BatchProgress>,
    log: &'a Log,
    stats: &'a Arc<BatchStats>,
}

/// One instance: its container, the agent, the trajectory and the prediction.
async fn process_instance(instance: &Json, batch: &Batch<'_>) {
    let Batch {
        output,
        config,
        model_name,
        progress,
        log,
        stats,
    } = batch;
    let id = instance["instance_id"].as_str().unwrap_or_default().to_owned();
    let instance_dir = output.join(&id);
    let trajectory_path = instance_dir.join(format!("{id}.traj.json"));
    let preds = output.join("preds.json");
    // No leftovers of an earlier run if this one fails early.
    let _ = preds_remove(&preds, &id);
    let _ = std::fs::remove_file(&trajectory_path);
    progress.on_instance_start(&id);
    progress.update_instance_status(&id, "Pulling/starting environment");

    let (exit_status, submission) = match instance_environment(config, instance).await {
        Err(error) => {
            log.line("ERROR", &format!("Error processing instance {id}: {error}"));
            (ExitStatus::Error, String::new())
        }
        Ok(env) => {
            let on_step: StepCallback = {
                let progress = Arc::clone(progress);
                let id = id.clone();
                Arc::new(move |step, cost| {
                    progress.update_instance_status(&id, &format!("Step {step:3} (${cost:.2})"));
                })
            };
            let mut extra = Map::new();
            extra.insert("instance_id".to_owned(), json!(id));
            let result = match build_agent(
                config,
                BuildOptions {
                    kind: AgentKind::Default,
                    default_environment: "docker",
                    env: Some(Arc::clone(&env)),
                    output: Some(trajectory_path.clone()),
                    extra,
                    on_step: Some(on_step),
                    batch: Some(Arc::clone(stats)),
                    ..BuildOptions::default()
                },
            )
            .await
            {
                Err(error) => {
                    log.line("ERROR", &format!("Error processing instance {id}: {error}"));
                    (ExitStatus::Error, String::new())
                }
                Ok(built) => {
                    let task = instance["problem_statement"].as_str().unwrap_or_default();
                    match built.agent.run(task, Map::new()).await {
                        Ok(output) => {
                            log.line("INFO", &format!("Saved trajectory to '{}'", trajectory_path.display()));
                            if let Some(error) = &output.exit.error {
                                log.line("ERROR", &format!("Instance {id}: {error}"));
                            }
                            (output.exit.status, output.exit.submission)
                        }
                        Err(error) => {
                            log.line("ERROR", &format!("Error processing instance {id}: {error}"));
                            (ExitStatus::Error, String::new())
                        }
                    }
                }
            };
            env.cleanup().await;
            result
        }
    };
    if let Err(error) = preds_update(&preds, &id, model_name, &submission) {
        log.line("ERROR", &format!("Could not update {}: {error}", preds.display()));
    }
    progress.on_instance_end(&id, exit_status.as_str());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preds_hold_one_entry_per_instance() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("preds.json");
        preds_update(&path, "a__1", "m", "diff a").unwrap();
        preds_update(&path, "b__2", "m", "").unwrap();
        preds_update(&path, "a__1", "m", "diff a2").unwrap();
        let preds = read_preds(&path);
        assert_eq!(preds.len(), 2);
        assert_eq!(
            preds["a__1"],
            json!({ "model_name_or_path": "m", "instance_id": "a__1", "model_patch": "diff a2" })
        );
        preds_remove(&path, "a__1").unwrap();
        assert_eq!(read_preds(&path).keys().collect::<Vec<_>>(), ["b__2"]);
    }
}
