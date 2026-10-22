//! [`ClaudeHooks`]: the configured handlers as the harness's tool, generation and compaction
//! hooks, and a [`PermissionRequest`](ClaudeHooks::permission_policy) policy decorator.

use std::fmt;
use std::sync::Arc;

use delta_ai::types::{Message, UserBlock, UserContent};
use delta_chord::context::Context;
use delta_harness::approval::{ApprovalCall, ApprovalPolicy, PermissionsDoc, PolicyDecision, PolicyError};
use delta_harness::compaction::{COMPACTION_TASK, CompactionReason};
use delta_harness::documents::DocAddress;
use delta_harness::generation::GENERATION_TASK;
use delta_harness::records::ConversationId;
use delta_harness::registry::{
    BeforeCompact, BeforeTool, CompactDecision, CompactionHooks, Extension, GenerationHooks, HookApi, HookError,
    HookPermission, HookResult, InputDecision, RunInput, ToolHooks, Yielded,
};
use delta_harness::tool::{Severity, TOOL_TASK, ToolExecutionResult, ToolHookCall};
use futures::future::BoxFuture;
use serde_json::{Value, json};

use crate::config::HooksConfig;
use crate::protocol::{HookOutput, arguments_from, claude_permission_mode, claude_tool_input, claude_tool_name};
use crate::runner::{Outcome, run_handlers};

/// The name of the extension [`ClaudeHooks::extension`] makes.
pub const CLAUDE_HOOKS: &str = "claude-hooks";

/// Gives the transcript path handlers receive for a conversation.
pub type TranscriptPath = Arc<dyn Fn(ConversationId) -> String + Send + Sync>;
/// Receives each non-blocking handler failure: the event and the error.
pub type OnError = Arc<dyn Fn(&str, &str) + Send + Sync>;

/// How [`ClaudeHooks`] runs handlers.
#[derive(Clone, Default)]
pub struct ClaudeHooksOptions {
    /// For HTTP handlers; a default client when unset.
    pub client: Option<reqwest::Client>,
    /// `transcript_path` of the input; empty when unset (delta keeps no transcript file).
    pub transcript_path: Option<TranscriptPath>,
    pub on_error: Option<OnError>,
}

impl fmt::Debug for ClaudeHooksOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClaudeHooksOptions")
            .field("client", &self.client.is_some())
            .field("transcript_path", &self.transcript_path.is_some())
            .field("on_error", &self.on_error.is_some())
            .finish()
    }
}

/// Claude Code's command and HTTP hooks over the harness's hook seams.
pub struct ClaudeHooks {
    config: HooksConfig,
    client: reqwest::Client,
    transcript_path: Option<TranscriptPath>,
    on_error: Option<OnError>,
}

impl fmt::Debug for ClaudeHooks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClaudeHooks")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl ClaudeHooks {
    pub fn new(config: HooksConfig, options: ClaudeHooksOptions) -> Arc<Self> {
        Arc::new(Self {
            config,
            client: options.client.unwrap_or_default(),
            transcript_path: options.transcript_path,
            on_error: options.on_error,
        })
    }

    /// The extension holding the tool, generation and compaction hooks.
    pub fn extension(self: &Arc<Self>) -> Extension {
        let tool: Arc<dyn ToolHooks> = self.clone();
        let generation: Arc<dyn GenerationHooks> = self.clone();
        let compaction: Arc<dyn CompactionHooks> = self.clone();
        Extension::new(CLAUDE_HOOKS)
            .with_hooks(TOOL_TASK, tool)
            .with_hooks(GENERATION_TASK, generation)
            .with_hooks(COMPACTION_TASK, compaction)
    }

    /// `inner` with the `PermissionRequest` handlers deciding what it would ask: `allow`
    /// allows, `deny` (or exit 2) denies, anything else keeps the request.
    pub fn permission_policy(self: &Arc<Self>, inner: Arc<dyn ApprovalPolicy>) -> Arc<dyn ApprovalPolicy> {
        Arc::new(PermissionRequestPolicy {
            hooks: self.clone(),
            inner,
        })
    }

    /// The input of `event`: the common fields over `fields`.
    async fn input(&self, api: &HookApi<'_>, event: &str, fields: Value, ctx: &Context) -> Value {
        let cwd = api.cwd(ctx).await.ok().flatten().unwrap_or_default();
        let mode = match api
            .snapshot(DocAddress::<PermissionsDoc>::new(api.conversation_id), ctx)
            .await
        {
            Ok(Some(value)) => value.get("mode").and_then(|mode| mode.as_str().map(str::to_owned)),
            _ => None,
        };
        let transcript = self
            .transcript_path
            .as_ref()
            .map(|path| path(api.conversation_id))
            .unwrap_or_default();
        let mut input = json!({
            "session_id": api.conversation_id.to_string(),
            "transcript_path": transcript,
            "cwd": cwd,
            "permission_mode": claude_permission_mode(mode.as_deref()),
            "hook_event_name": event,
        });
        if let (Some(input), Value::Object(fields)) = (input.as_object_mut(), fields) {
            input.extend(fields);
        }
        input
    }

    /// Runs `event`'s handlers matching `values` (none: every group) and reports the
    /// failures of a fresh run.
    async fn run(
        &self,
        api: &HookApi<'_>,
        event: &str,
        values: &[&str],
        fields: Value,
        ctx: &Context,
    ) -> Result<Vec<Outcome>, HookError> {
        let handlers: Vec<_> = self
            .config
            .handlers(event, values)
            .into_iter()
            .map(|(_, handler)| handler)
            .collect();
        if handlers.is_empty() && api.memo::<Vec<Outcome>>(&format!("claude-hooks:{event}"))?.is_none() {
            return Ok(Vec::new());
        }
        let input = self.input(api, event, fields, ctx).await;
        let (outcomes, fresh) = run_handlers(&self.client, api, event, &handlers, &input, ctx).await?;
        if fresh && let Some(on_error) = &self.on_error {
            for outcome in &outcomes {
                if let Outcome::Failed { error } = outcome {
                    on_error(event, error);
                }
            }
        }
        Ok(outcomes)
    }

    fn report(&self, event: &str, error: &str) {
        if let Some(on_error) = &self.on_error {
            on_error(event, error);
        }
    }
}

/// The JSON outputs of `outcomes`.
fn outputs(outcomes: &[Outcome]) -> impl Iterator<Item = &HookOutput> {
    outcomes.iter().filter_map(|outcome| match outcome {
        Outcome::Output { output, .. } => Some(output),
        _ => None,
    })
}

/// The first exit-2 reason.
fn blocked(outcomes: &[Outcome]) -> Option<String> {
    outcomes.iter().find_map(|outcome| match outcome {
        Outcome::Blocked { reason } => Some(reason.clone()),
        _ => None,
    })
}

/// Every `additionalContext`, joined.
fn contexts(outcomes: &[Outcome]) -> Option<String> {
    let texts: Vec<&str> = outputs(outcomes)
        .filter_map(|output| output.specific_str("additionalContext"))
        .collect();
    (!texts.is_empty()).then(|| texts.join("\n\n"))
}

/// The text items of a result's output.
fn output_text(result: &ToolExecutionResult) -> String {
    result
        .output
        .iter()
        .flatten()
        .filter_map(|block| match block {
            UserBlock::Text(text) => Some(text.text.as_str()),
            UserBlock::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

impl ToolHooks for ClaudeHooks {
    /// `PreToolUse`: deny (or exit 2, or a block) blocks; `ask` and `allow` reach the
    /// policy; `updatedInput` replaces the arguments; `additionalContext` follows the
    /// result. Several handlers: deny over defer over ask over allow.
    fn before_tool<'a>(
        &'a self,
        call: &'a ToolHookCall,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<BeforeTool>> {
        Box::pin(async move {
            let name = call.call.name.as_str();
            let claude = claude_tool_name(name);
            let fields = json!({
                "tool_name": claude,
                "tool_input": claude_tool_input(&call.call.arguments),
                "tool_use_id": call.call.id,
            });
            let outcomes = self.run(api, "PreToolUse", &[&claude, name], fields, ctx).await?;
            if outcomes.is_empty() {
                return Ok(None);
            }
            if let Some(reason) = blocked(&outcomes) {
                return Ok(Some(BeforeTool {
                    block: Some(reason),
                    ..BeforeTool::default()
                }));
            }
            let mut result = BeforeTool::default();
            let mut decision = 0;
            for output in outputs(&outcomes) {
                if let Some(reason) = output.block() {
                    return Ok(Some(BeforeTool {
                        block: Some(reason),
                        ..BeforeTool::default()
                    }));
                }
                let reason = output.specific_str("permissionDecisionReason").map(str::to_owned);
                // deny > defer > ask > allow
                let (rank, permission) = match output.specific_str("permissionDecision") {
                    Some("deny") => {
                        let reason = reason.unwrap_or_else(|| "Denied by PreToolUse hook".to_owned());
                        return Ok(Some(BeforeTool {
                            block: Some(reason),
                            ..BeforeTool::default()
                        }));
                    }
                    Some("defer") => (3, None),
                    Some("ask") => (2, Some(HookPermission::Ask { reason })),
                    Some("allow") => (1, Some(HookPermission::Allow)),
                    _ => (0, None),
                };
                if rank > decision {
                    decision = rank;
                    result.permission = permission;
                }
                if result.arguments.is_none()
                    && let Some(updated) = output.specific("updatedInput")
                {
                    result.arguments = arguments_from(updated, &call.call.arguments);
                }
            }
            result.context = contexts(&outcomes);
            Ok((result != BeforeTool::default()).then_some(result))
        })
    }

    /// `PostToolUse` / `PostToolUseFailure`: `updatedToolOutput` replaces the output; a
    /// block's reason, exit 2's stderr and `additionalContext` follow it.
    fn after_tool<'a>(
        &'a self,
        call: &'a ToolHookCall,
        result: &'a ToolExecutionResult,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<ToolExecutionResult>> {
        Box::pin(async move {
            let name = call.call.name.as_str();
            let claude = claude_tool_name(name);
            let (event, extra) = if result.is_error {
                let error = result
                    .diagnostics
                    .iter()
                    .find(|diagnostic| diagnostic.severity == Severity::Error)
                    .map_or_else(|| output_text(result), |diagnostic| diagnostic.message.clone());
                ("PostToolUseFailure", json!({ "error": error }))
            } else {
                let response = json!({ "output": output_text(result), "structured": result.structured_output });
                ("PostToolUse", json!({ "tool_response": response }))
            };
            let mut fields = json!({
                "tool_name": claude,
                "tool_input": claude_tool_input(&call.call.arguments),
                "tool_use_id": call.call.id,
            });
            if let (Some(fields), Value::Object(extra)) = (fields.as_object_mut(), extra) {
                fields.extend(extra);
            }
            let outcomes = self.run(api, event, &[&claude, name], fields, ctx).await?;
            let mut changed = result.clone();
            let mut touched = false;
            if let Some(updated) = outputs(&outcomes).find_map(|output| output.specific_str("updatedToolOutput")) {
                changed.output = Some(vec![UserBlock::text(updated)]);
                touched = true;
            }
            let notes: Vec<String> = outcomes
                .iter()
                .filter_map(|outcome| match outcome {
                    Outcome::Blocked { reason } => Some(reason.clone()),
                    Outcome::Output { output, .. } => output.block(),
                    Outcome::Failed { .. } => None,
                })
                .chain(contexts(&outcomes))
                .collect();
            if !notes.is_empty() {
                let mut output = changed.output.take().unwrap_or_default();
                output.push(UserBlock::text(notes.join("\n\n")));
                changed.output = Some(output);
                touched = true;
            }
            Ok(touched.then_some(changed))
        })
    }
}

impl GenerationHooks for ClaudeHooks {
    /// `UserPromptSubmit`: a block (or exit 2) blocks the run; `additionalContext` and plain
    /// stdout become context.
    fn on_input<'a>(
        &'a self,
        input: &'a RunInput,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<InputDecision>> {
        Box::pin(async move {
            let prompt: Vec<String> = input
                .entries
                .iter()
                .flat_map(|entry| entry.model.iter().flatten())
                .filter_map(|message| match message {
                    Message::User(user) => Some(user.content.text()),
                    _ => None,
                })
                .collect();
            let fields = json!({ "prompt": prompt.join("\n\n") });
            let outcomes = self.run(api, "UserPromptSubmit", &[], fields, ctx).await?;
            if let Some(reason) = blocked(&outcomes).or_else(|| outputs(&outcomes).find_map(HookOutput::block)) {
                return Ok(Some(InputDecision::Block { reason }));
            }
            let mut texts: Vec<String> = outcomes
                .iter()
                .filter_map(|outcome| match outcome {
                    Outcome::Output { text: Some(text), .. } => Some(text.clone()),
                    _ => None,
                })
                .collect();
            texts.extend(contexts(&outcomes));
            Ok((!texts.is_empty()).then(|| InputDecision::Context(texts.join("\n\n"))))
        })
    }

    /// `Stop`, or `SubagentStop` in a conversation a task owns: `decision: "block"` (or
    /// exit 2) continues the run with the reason.
    fn on_yield<'a>(
        &'a self,
        yielded: &'a Yielded,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<UserContent>> {
        Box::pin(async move {
            let owned = api
                .conversation(ctx)
                .await?
                .is_some_and(|record| record.owner.is_some());
            let event = if owned { "SubagentStop" } else { "Stop" };
            let fields = json!({
                "stop_hook_active": yielded.continuations > 0,
                "last_assistant_message": yielded.message.text(),
            });
            let outcomes = self.run(api, event, &[], fields, ctx).await?;
            let reason = blocked(&outcomes).or_else(|| {
                outputs(&outcomes)
                    .filter(|output| output.decision.as_deref() == Some("block"))
                    .find_map(|output| output.reason.clone())
            });
            Ok(reason.map(UserContent::from))
        })
    }
}

impl CompactionHooks for ClaudeHooks {
    /// `PreCompact` (matcher: `manual` or `auto`): a block (or exit 2) declines the
    /// compaction.
    fn before_compact<'a>(
        &'a self,
        compaction: &'a BeforeCompact,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<CompactDecision>> {
        Box::pin(async move {
            let trigger = match compaction.reason {
                CompactionReason::Manual => "manual",
                CompactionReason::Threshold | CompactionReason::Overflow => "auto",
            };
            let fields = json!({
                "trigger": trigger,
                "custom_instructions": compaction.instructions.clone().unwrap_or_default(),
            });
            let outcomes = self.run(api, "PreCompact", &[trigger], fields, ctx).await?;
            let declined = blocked(&outcomes).is_some() || outputs(&outcomes).any(|output| output.block().is_some());
            Ok(declined.then_some(CompactDecision::Decline))
        })
    }
}

/// [`ClaudeHooks::permission_policy`].
struct PermissionRequestPolicy {
    hooks: Arc<ClaudeHooks>,
    inner: Arc<dyn ApprovalPolicy>,
}

impl ApprovalPolicy for PermissionRequestPolicy {
    fn decide<'a>(
        &'a self,
        call: &'a ApprovalCall<'a>,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<PolicyDecision, PolicyError>> {
        Box::pin(async move {
            let decision = self.inner.decide(call, api, ctx).await?;
            let PolicyDecision::Ask { proposed_rule, .. } = &decision else {
                return Ok(decision);
            };
            let name = call.call.call.name.as_str();
            let claude = claude_tool_name(name);
            let fields = json!({
                "tool_name": claude,
                "tool_input": claude_tool_input(&call.call.call.arguments),
                "permission_suggestions": proposed_rule.iter().collect::<Vec<_>>(),
            });
            let outcomes = self
                .hooks
                .run(api, "PermissionRequest", &[&claude, name], fields, ctx)
                .await?;
            if let Some(reason) = blocked(&outcomes) {
                return Ok(PolicyDecision::Deny { reason });
            }
            for output in outputs(&outcomes) {
                let Some(answer) = output.specific("decision") else {
                    continue;
                };
                if answer.get("updatedInput").is_some() {
                    self.hooks.report(
                        "PermissionRequest",
                        "updatedInput is not supported; the call runs as requested",
                    );
                }
                match answer.get("behavior").and_then(Value::as_str) {
                    Some("allow") => return Ok(PolicyDecision::Allow),
                    Some("deny") => {
                        let reason = answer
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("Denied by PermissionRequest hook")
                            .to_owned();
                        return Ok(PolicyDecision::Deny { reason });
                    }
                    _ => {}
                }
            }
            Ok(decision)
        })
    }
}
