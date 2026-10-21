//! Harness settings: what a host passes ([`HarnessSettings`], every field optional, read
//! live and never stored) and what readers get ([`Settings`], defaults filled in, object
//! fields merged field by field).

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use delta_ai::retry::RetryPolicy;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::model::{ConversationStreamOptions, ModelRef};
use crate::tool::ToolExecutionMode;

/// How many queued items of one mode a boundary places.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueMode {
    #[default]
    OneAtATime,
    All,
}

/// What a failed run (`model_error`, `no_model`) does with the items queued behind it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AfterFailedRun {
    /// Leave them for the next input or queued write.
    #[default]
    Keep,
    /// Apply the final boundary, so the next run starts with them.
    Continue,
    /// Settle queued inputs `unanswered` / `run_failed`; queued writes stay.
    Withdraw,
}

/// When generation starts a compaction, and how much it keeps (compaction.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactionPolicy {
    pub enabled: bool,
    /// Room kept free for the answer: above `context_window - reserve_tokens` a generation
    /// compacts before its request.
    pub reserve_tokens: u64,
    /// Tokens of recent history a compaction keeps verbatim.
    pub keep_recent_tokens: u64,
    /// Below the blocking threshold by this much, a background compaction starts; `0`
    /// disables it.
    pub background_tokens: u64,
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
            background_tokens: 32_768,
        }
    }
}

/// [`CompactionPolicy`] with every field optional.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PartialCompactionPolicy {
    pub enabled: Option<bool>,
    pub reserve_tokens: Option<u64>,
    pub keep_recent_tokens: Option<u64>,
    pub background_tokens: Option<u64>,
}

impl PartialCompactionPolicy {
    /// These fields over `base`.
    pub fn over(&self, base: CompactionPolicy) -> CompactionPolicy {
        CompactionPolicy {
            enabled: self.enabled.unwrap_or(base.enabled),
            reserve_tokens: self.reserve_tokens.unwrap_or(base.reserve_tokens),
            keep_recent_tokens: self.keep_recent_tokens.unwrap_or(base.keep_recent_tokens),
            background_tokens: self.background_tokens.unwrap_or(base.background_tokens),
        }
    }
}

/// The compaction settings: the policy and per-model overrides keyed `provider/model_id`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompactionSettings {
    pub policy: PartialCompactionPolicy,
    pub model_overrides: IndexMap<String, PartialCompactionPolicy>,
}

/// The resolved compaction settings.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResolvedCompaction {
    pub policy: CompactionPolicy,
    pub model_overrides: IndexMap<String, PartialCompactionPolicy>,
}

impl ResolvedCompaction {
    /// The policy for `model`: its override, field by field, over the policy.
    pub fn policy_for(&self, model: &ModelRef) -> CompactionPolicy {
        match self.model_overrides.get(&model.key()) {
            Some(overrides) => overrides.over(self.policy),
            None => self.policy,
        }
    }
}

/// How often progress commits (partials, tool output).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgressPolicy {
    /// At most one partial-response commit per interval.
    pub partial_interval: Duration,
    /// The pace of tool output commits.
    pub output_interval: Duration,
}

impl Default for ProgressPolicy {
    fn default() -> Self {
        Self {
            partial_interval: Duration::from_millis(100),
            output_interval: Duration::from_millis(100),
        }
    }
}

/// [`RetryPolicy`] with every field optional.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PartialRetryPolicy {
    pub enabled: Option<bool>,
    pub max_retries: Option<u32>,
    pub base_delay: Option<Duration>,
    pub max_delay: Option<Duration>,
}

/// [`ProgressPolicy`] with every field optional.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PartialProgressPolicy {
    pub partial_interval: Option<Duration>,
    pub output_interval: Option<Duration>,
}

/// How long an approval request waits by default: 15 minutes.
pub const DEFAULT_APPROVAL_TIMEOUT: Duration = Duration::from_mins(15);

/// The default retry policy: enabled, 3 retries, 2 s doubling, at most 60 s.
pub const DEFAULT_RETRY_POLICY: RetryPolicy = RetryPolicy {
    enabled: true,
    max_retries: 3,
    base_delay: Duration::from_secs(2),
    max_delay: Some(Duration::from_mins(1)),
};

/// The default restart limit.
pub const DEFAULT_RESTART_LIMIT: u32 = 5;

/// What a host passes. Every field is optional; settings are read live at each use and
/// never stored, and a change makes no commit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HarnessSettings {
    /// The default extension selection; `None`: every installed one, in install order.
    pub extensions: Option<Vec<String>>,
    pub stream: Option<ConversationStreamOptions>,
    pub retry: Option<PartialRetryPolicy>,
    pub compaction: Option<CompactionSettings>,
    pub progress: Option<PartialProgressPolicy>,
    pub tool_execution: Option<ToolExecutionMode>,
    pub steering_mode: Option<QueueMode>,
    pub follow_up_mode: Option<QueueMode>,
    /// How long an idle conversation keeps its context range in memory.
    pub context_retention: Option<Duration>,
    /// How long an unobserved document stays cached; read once, at open.
    pub document_retention: Option<Duration>,
    pub after_failed_run: Option<AfterFailedRun>,
    /// `Some(None)`: never park.
    pub restart_limit: Option<Option<u32>>,
    pub close_grace: Option<Duration>,
    /// How long a tool call waits for an approval decision when the policy names no
    /// timeout.
    pub approval_timeout: Option<Duration>,
}

/// The resolved settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub extensions: Option<Vec<String>>,
    pub stream: ConversationStreamOptions,
    pub retry: RetryPolicy,
    pub compaction: ResolvedCompaction,
    pub progress: ProgressPolicy,
    pub tool_execution: ToolExecutionMode,
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
    pub context_retention: Duration,
    pub document_retention: Duration,
    pub after_failed_run: AfterFailedRun,
    pub restart_limit: Option<u32>,
    pub close_grace: Duration,
    pub approval_timeout: Duration,
}

impl Default for Settings {
    fn default() -> Self {
        HarnessSettings::default().resolve()
    }
}

impl HarnessSettings {
    /// Every field over its built-in default; object fields merged field by field.
    pub fn resolve(&self) -> Settings {
        let retry = self.retry.unwrap_or_default();
        let progress = self.progress.unwrap_or_default();
        let defaults = ProgressPolicy::default();
        let compaction = self.compaction.clone().unwrap_or_default();
        Settings {
            extensions: self.extensions.clone(),
            stream: self.stream.clone().unwrap_or_default(),
            retry: RetryPolicy {
                enabled: retry.enabled.unwrap_or(DEFAULT_RETRY_POLICY.enabled),
                max_retries: retry.max_retries.unwrap_or(DEFAULT_RETRY_POLICY.max_retries),
                base_delay: retry.base_delay.unwrap_or(DEFAULT_RETRY_POLICY.base_delay),
                max_delay: retry.max_delay.or(DEFAULT_RETRY_POLICY.max_delay),
            },
            compaction: ResolvedCompaction {
                policy: compaction.policy.over(CompactionPolicy::default()),
                model_overrides: compaction.model_overrides,
            },
            progress: ProgressPolicy {
                partial_interval: progress.partial_interval.unwrap_or(defaults.partial_interval),
                output_interval: progress.output_interval.unwrap_or(defaults.output_interval),
            },
            tool_execution: self.tool_execution.unwrap_or_default(),
            steering_mode: self.steering_mode.unwrap_or_default(),
            follow_up_mode: self.follow_up_mode.unwrap_or_default(),
            context_retention: self.context_retention.unwrap_or(Duration::from_mins(10)),
            document_retention: self.document_retention.unwrap_or(Duration::from_mins(10)),
            after_failed_run: self.after_failed_run.unwrap_or_default(),
            restart_limit: self.restart_limit.unwrap_or(Some(DEFAULT_RESTART_LIMIT)),
            close_grace: self.close_grace.unwrap_or(Duration::from_secs(30)),
            approval_timeout: self.approval_timeout.unwrap_or(DEFAULT_APPROVAL_TIMEOUT),
        }
    }
}

/// Where the Harness reads its settings: one value, or a closure called at each use.
#[derive(Clone)]
pub enum SettingsSource {
    Fixed(HarnessSettings),
    Dynamic(Arc<dyn Fn() -> HarnessSettings + Send + Sync>),
}

impl SettingsSource {
    /// The resolved settings now.
    pub fn resolve(&self) -> Settings {
        match self {
            Self::Fixed(settings) => settings.resolve(),
            Self::Dynamic(settings) => settings().resolve(),
        }
    }
}

impl Default for SettingsSource {
    fn default() -> Self {
        Self::Fixed(HarnessSettings::default())
    }
}

impl fmt::Debug for SettingsSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fixed(settings) => f.debug_tuple("Fixed").field(settings).finish(),
            Self::Dynamic(_) => f.write_str("Dynamic"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_fill_every_field() {
        let settings = Settings::default();
        assert_eq!(settings.retry, DEFAULT_RETRY_POLICY);
        assert_eq!(settings.progress.partial_interval, Duration::from_millis(100));
        assert_eq!(settings.restart_limit, Some(5));
        assert_eq!(settings.steering_mode, QueueMode::OneAtATime);
        assert_eq!(settings.compaction.policy, CompactionPolicy::default());
    }

    #[test]
    fn object_fields_merge_field_by_field() {
        let mut overrides = IndexMap::new();
        overrides.insert(
            "p/m".to_owned(),
            PartialCompactionPolicy {
                reserve_tokens: Some(1),
                ..PartialCompactionPolicy::default()
            },
        );
        let settings = HarnessSettings {
            retry: Some(PartialRetryPolicy {
                max_retries: Some(1),
                ..PartialRetryPolicy::default()
            }),
            compaction: Some(CompactionSettings {
                policy: PartialCompactionPolicy {
                    enabled: Some(false),
                    ..PartialCompactionPolicy::default()
                },
                model_overrides: overrides,
            }),
            restart_limit: Some(None),
            ..HarnessSettings::default()
        }
        .resolve();
        assert_eq!(settings.retry.max_retries, 1);
        assert_eq!(settings.retry.base_delay, Duration::from_secs(2));
        assert_eq!(settings.restart_limit, None);
        let policy = settings.compaction.policy_for(&ModelRef::new("p", "m"));
        assert!(!policy.enabled);
        assert_eq!(policy.reserve_tokens, 1);
        assert_eq!(policy.keep_recent_tokens, 20_000);
    }
}
