//! One connection: newline framing, `initialize` first, requests answered in order, client
//! answers to approval requests turned into decisions.

use std::collections::HashMap;
use std::sync::Arc;

use delta_chord::context::Context;
use delta_harness::agent::{AgentChange, Change};
use delta_harness::approval::{PermissionRules, PermissionsChange, PermissionsState, ReviewDecision};
use delta_harness::harness::{
    AbortOptions, Conversation, ConversationCreateOptions, EntryRange, Harness, HarnessError,
};
use delta_harness::inbox::{SubmissionDraft, WhenBusy};
use delta_harness::model::ModelRef;
use delta_harness::records::{ConversationId, ConversationQuery, Cursor, ScanOrder, SubmissionId, SubmissionState};
use delta_protocol::delta::{
    PermissionMode, PermissionRulesResponse, Permissions, PermissionsResponse, PermissionsUpdateParams,
    ToolApprovalDecision, ToolRequestApprovalResponse,
};
use delta_protocol::v2::{
    CommandExecutionApprovalDecision, CommandExecutionRequestApprovalResponse, FileChangeApprovalDecision,
    FileChangeRequestApprovalResponse, InitializeResponse, ThreadListResponse, ThreadReadResponse,
    ThreadStartedNotification, Turn, TurnInterruptResponse, TurnStartResponse, TurnStatus, TurnSteerResponse,
    UserInput,
};
use delta_protocol::{
    CLIENT_REQUEST_METHODS, ClientRequest, INTERNAL_ERROR, INVALID_PARAMS, INVALID_REQUEST, JsonRpcMessage,
    METHOD_NOT_FOUND, PARSE_ERROR, RequestId, ServerNotification,
};
use serde::Serialize;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

use crate::items::{ApprovalKind, user_content};
use crate::threads::{self, Attached, Followed, Outgoing, Shared};
use crate::{AppServerOptions, ServeError};

/// Why a request failed: a JSON-RPC error code and message.
struct Failure {
    code: i64,
    message: String,
}

impl Failure {
    fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl From<HarnessError> for Failure {
    fn from(error: HarnessError) -> Self {
        let code = match error {
            HarnessError::UnknownConversation(_) | HarnessError::UnknownSubmission(_) | HarnessError::Approval(_) => {
                INVALID_PARAMS
            }
            _ => INTERNAL_ERROR,
        };
        Self::new(code, error.to_string())
    }
}

/// What to do once a request's response is written.
enum Then {
    Follow(ConversationId, Box<Attached>),
    Resend(ConversationId),
    Notify(ServerNotification),
}

type Handled = Result<(Value, Vec<Then>), Failure>;

fn reply(result: &impl Serialize) -> Handled {
    let value = serde_json::to_value(result).map_err(|error| Failure::new(INTERNAL_ERROR, error.to_string()))?;
    Ok((value, Vec::new()))
}

struct Connection {
    harness: Harness,
    options: Arc<AppServerOptions>,
    shared: Arc<Shared>,
    initialized: bool,
    followed: HashMap<ConversationId, Followed>,
    ctx: Context,
}

pub(crate) async fn serve<R, W>(
    harness: Harness,
    options: Arc<AppServerOptions>,
    reader: R,
    writer: W,
    ctx: &Context,
) -> Result<(), ServeError>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let (lines, mut queued) = mpsc::unbounded_channel::<String>();
    let writer = tokio::spawn(async move {
        let mut writer = writer;
        while let Some(line) = queued.recv().await {
            writer.write_all(line.as_bytes()).await?;
            writer.write_all(b"\n").await?;
            writer.flush().await?;
        }
        Ok::<(), std::io::Error>(())
    });
    let mut connection = Connection {
        harness,
        options,
        shared: Arc::new(Shared::new(Outgoing::new(lines))),
        initialized: false,
        followed: HashMap::new(),
        ctx: ctx.clone(),
    };
    let mut reader = BufReader::new(reader).lines();
    let read = loop {
        let line = tokio::select! {
            line = reader.next_line() => line,
            () = ctx.cancelled() => break Ok(()),
        };
        match line {
            Ok(Some(line)) if line.trim().is_empty() => {}
            Ok(Some(line)) => connection.line(&line).await,
            Ok(None) => break Ok(()),
            Err(error) => break Err(ServeError::Io(error)),
        }
    };
    for (_, followed) in connection.followed.drain() {
        followed.stop().await;
    }
    drop(connection);
    let written = match writer.await {
        // The client closed its side first: nothing left to deliver it.
        Ok(Err(error))
            if matches!(
                error.kind(),
                std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset
            ) =>
        {
            Ok(())
        }
        Ok(result) => result.map_err(ServeError::Io),
        Err(error) => Err(ServeError::Io(std::io::Error::other(error))),
    };
    read.and(written)
}

fn parse_thread(id: &str) -> Result<ConversationId, Failure> {
    id.parse()
        .map(ConversationId)
        .map_err(|_| Failure::new(INVALID_PARAMS, format!("Unknown thread {id}")))
}

fn parse_turn(id: &str) -> Result<SubmissionId, Failure> {
    id.parse()
        .map(SubmissionId)
        .map_err(|_| Failure::new(INVALID_PARAMS, format!("Unknown turn {id}")))
}

fn protocol_mode(mode: delta_harness::approval::PermissionMode) -> PermissionMode {
    use delta_harness::approval::PermissionMode as Harness;
    match mode {
        Harness::Default => PermissionMode::Default,
        Harness::AcceptEdits => PermissionMode::AcceptEdits,
        Harness::Plan => PermissionMode::Plan,
        Harness::DontAsk => PermissionMode::DontAsk,
        Harness::Bypass => PermissionMode::Bypass,
    }
}

fn harness_mode(mode: PermissionMode) -> delta_harness::approval::PermissionMode {
    use delta_harness::approval::PermissionMode as Harness;
    match mode {
        PermissionMode::Default => Harness::Default,
        PermissionMode::AcceptEdits => Harness::AcceptEdits,
        PermissionMode::Plan => Harness::Plan,
        PermissionMode::DontAsk => Harness::DontAsk,
        PermissionMode::Bypass => Harness::Bypass,
    }
}

fn protocol_rules(rules: PermissionRules) -> delta_protocol::delta::PermissionRules {
    delta_protocol::delta::PermissionRules {
        allow: rules.allow,
        ask: rules.ask,
        deny: rules.deny,
    }
}

fn harness_rules(rules: delta_protocol::delta::PermissionRules) -> PermissionRules {
    PermissionRules {
        allow: rules.allow,
        ask: rules.ask,
        deny: rules.deny,
    }
}

/// A client's answer to an approval request of `kind`.
fn decision(kind: ApprovalKind, result: Value) -> Result<ReviewDecision, serde_json::Error> {
    Ok(match kind {
        ApprovalKind::CommandExecution => {
            match serde_json::from_value::<CommandExecutionRequestApprovalResponse>(result)?.decision {
                CommandExecutionApprovalDecision::Accept => ReviewDecision::Approved,
                CommandExecutionApprovalDecision::AcceptForSession => ReviewDecision::ApprovedForSession,
                CommandExecutionApprovalDecision::AcceptWithExecpolicyAmendment { execpolicy_amendment } => {
                    ReviewDecision::ApprovedWithRule {
                        rule: format!("Bash({}:*)", execpolicy_amendment.command.join(" ")),
                    }
                }
                CommandExecutionApprovalDecision::Decline => ReviewDecision::Denied { rejection: None },
                CommandExecutionApprovalDecision::Cancel => ReviewDecision::Abort,
            }
        }
        ApprovalKind::FileChange => match serde_json::from_value::<FileChangeRequestApprovalResponse>(result)?.decision
        {
            FileChangeApprovalDecision::Accept => ReviewDecision::Approved,
            FileChangeApprovalDecision::AcceptForSession => ReviewDecision::ApprovedForSession,
            FileChangeApprovalDecision::Decline => ReviewDecision::Denied { rejection: None },
            FileChangeApprovalDecision::Cancel => ReviewDecision::Abort,
        },
        ApprovalKind::Tool => match serde_json::from_value::<ToolRequestApprovalResponse>(result)?.decision {
            ToolApprovalDecision::Accept => ReviewDecision::Approved,
            ToolApprovalDecision::AcceptForSession => ReviewDecision::ApprovedForSession,
            ToolApprovalDecision::AcceptWithRule { rule } => ReviewDecision::ApprovedWithRule { rule },
            ToolApprovalDecision::Decline => ReviewDecision::Denied { rejection: None },
            ToolApprovalDecision::Cancel => ReviewDecision::Abort,
        },
    })
}

fn in_progress(id: String) -> Turn {
    Turn {
        id,
        items: Vec::new(),
        status: TurnStatus::InProgress,
        error: None,
        started_at: None,
        completed_at: None,
        duration_ms: None,
    }
}

impl Connection {
    async fn line(&mut self, line: &str) {
        let value: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(error) => {
                self.shared
                    .out
                    .fail(None, PARSE_ERROR, format!("Invalid JSON: {error}"));
                return;
            }
        };
        let Ok(message) = serde_json::from_value::<JsonRpcMessage>(value.clone()) else {
            let id: Option<RequestId> = value.get("id").cloned().and_then(|id| serde_json::from_value(id).ok());
            self.shared
                .out
                .fail(id.as_ref(), INVALID_REQUEST, "Not a JSON-RPC message".to_owned());
            return;
        };
        match message {
            JsonRpcMessage::Request(request) => self.request(request.id, &request.method, value).await,
            // `initialized` and anything else the client announces.
            JsonRpcMessage::Notification(_) => {}
            JsonRpcMessage::Response(response) => self.answer(&response.id, Some(response.result)).await,
            JsonRpcMessage::Error(error) => self.answer(&error.id, None).await,
        }
    }

    async fn request(&mut self, id: RequestId, method: &str, value: Value) {
        if !CLIENT_REQUEST_METHODS.contains(&method) {
            self.shared
                .out
                .fail(Some(&id), METHOD_NOT_FOUND, format!("Unknown method {method}"));
            return;
        }
        let request = match serde_json::from_value::<ClientRequest>(value) {
            Ok(request) => request,
            Err(error) => {
                self.shared
                    .out
                    .fail(Some(&id), INVALID_PARAMS, format!("Invalid {method} params: {error}"));
                return;
            }
        };
        if !self.initialized && !matches!(request, ClientRequest::Initialize { .. }) {
            self.shared
                .out
                .fail(Some(&id), INVALID_REQUEST, "Send initialize first".to_owned());
            return;
        }
        match self.handle(request).await {
            Ok((result, then)) => {
                self.shared.out.respond(id, result);
                for then in then {
                    self.then(then);
                }
            }
            Err(failure) => self.shared.out.fail(Some(&id), failure.code, failure.message),
        }
    }

    fn then(&mut self, then: Then) {
        match then {
            Then::Follow(id, attached) => {
                let followed = Followed::start(*attached, self.harness.clone(), self.shared.clone(), self.ctx.clone());
                self.followed.insert(id, followed);
            }
            Then::Resend(id) => {
                if let Some(followed) = self.followed.get(&id) {
                    followed.resend(&self.shared, id);
                }
            }
            Then::Notify(notification) => self.shared.out.notify(notification),
        }
    }

    /// The client answered approval request `id`; `None` when it answered with an error,
    /// which leaves the approval pending.
    async fn answer(&mut self, id: &RequestId, result: Option<Value>) {
        let Some(pending) = self.shared.take_request(id) else {
            return;
        };
        let Some(result) = result else {
            return;
        };
        let Ok(decision) = decision(pending.kind, result) else {
            return;
        };
        if let Ok(Some(conversation)) = self.harness.conversation(pending.conversation, &self.ctx).await {
            // Decided meanwhile (another client, a timeout): the request resolves anyway.
            let _ = conversation.decide(pending.approval, decision, &self.ctx).await;
        }
    }

    async fn handle(&mut self, request: ClientRequest) -> Handled {
        match request {
            ClientRequest::Initialize { params, .. } => {
                if self.initialized {
                    return Err(Failure::new(INVALID_REQUEST, "Already initialized"));
                }
                self.initialized = true;
                let opt_out = params
                    .capabilities
                    .and_then(|capabilities| capabilities.opt_out_notification_methods)
                    .unwrap_or_default();
                self.shared.out.opt_out(opt_out);
                reply(&InitializeResponse {
                    user_agent: self.options.user_agent.clone(),
                    codex_home: self.options.home.clone(),
                    platform_family: std::env::consts::FAMILY.to_owned(),
                    platform_os: std::env::consts::OS.to_owned(),
                })
            }
            ClientRequest::ThreadStart { params, .. } => {
                let instructions: Vec<String> = [params.base_instructions, params.developer_instructions]
                    .into_iter()
                    .flatten()
                    .collect();
                let mut agent = self.overrides(params.model.as_deref(), params.cwd.as_deref())?;
                if agent.model == Change::Keep
                    && let Some(model) = &self.options.model
                {
                    agent.model = Change::Set(model.clone());
                }
                if agent.cwd == Change::Keep {
                    agent.cwd = Change::Set(self.options.cwd.clone());
                }
                if !instructions.is_empty() {
                    agent.instructions = Change::Set(instructions.join("\n\n"));
                }
                let options = ConversationCreateOptions::ownerless().with_agent(agent);
                let conversation = self.harness.create_conversation(options, &self.ctx).await?;
                self.started(&conversation, false).await
            }
            ClientRequest::ThreadResume { params, .. } => {
                let conversation = self.conversation(&params.thread_id).await?;
                let agent = self.overrides(params.model.as_deref(), params.cwd.as_deref())?;
                if agent != AgentChange::default() {
                    conversation.configure(agent, &self.ctx).await?;
                }
                let id = conversation.id();
                let (status, then) = match self.followed.get(&id) {
                    Some(followed) => (followed.status(), Then::Resend(id)),
                    None => {
                        let attached = self.attach(&conversation).await?;
                        (attached.status(), Then::Follow(id, Box::new(attached)))
                    }
                };
                let (thread, info) =
                    threads::thread(&self.harness, &conversation, status, &self.options.cwd, true, &self.ctx).await?;
                let (value, _) = reply(&threads::session(thread, info))?;
                Ok((value, vec![then]))
            }
            ClientRequest::ThreadFork { params, .. } => {
                let conversation = self.conversation(&params.thread_id).await?;
                let at = match params.last_turn_id.as_deref() {
                    Some(turn) => {
                        let submission = self
                            .harness
                            .submission(parse_turn(turn)?, &self.ctx)
                            .await?
                            .ok_or_else(|| Failure::new(INVALID_PARAMS, format!("Unknown turn {turn}")))?;
                        match submission.status(&self.ctx).await?.state {
                            SubmissionState::Done {
                                answer: Some(answer), ..
                            } => answer,
                            _ => return Err(Failure::new(INVALID_PARAMS, format!("Turn {turn} has no answer"))),
                        }
                    }
                    None => conversation
                        .entries(EntryRange::default(), 1, None, &self.ctx)
                        .await?
                        .items
                        .first()
                        .map(|entry| entry.id)
                        .ok_or_else(|| Failure::new(INVALID_PARAMS, "The thread has no entries to fork"))?,
                };
                let agent = self.overrides(params.model.as_deref(), params.cwd.as_deref())?;
                let options = ConversationCreateOptions::ownerless().with_agent(agent);
                let fork = conversation.fork(at, options, &self.ctx).await?;
                self.started(&fork, true).await
            }
            ClientRequest::ThreadList { params, .. } => {
                let cursor = params
                    .cursor
                    .map(|cursor| serde_json::from_str::<Cursor>(&cursor))
                    .transpose()
                    .map_err(|_| Failure::new(INVALID_PARAMS, "Invalid cursor"))?;
                let limit = params.limit.map_or(50, |limit| limit.clamp(1, 200)) as usize;
                let query = ConversationQuery {
                    order: Some(ScanOrder::Descending),
                    ..ConversationQuery::default()
                };
                let page = self.harness.conversations(query, limit, cursor, &self.ctx).await?;
                let mut data = Vec::with_capacity(page.items.len());
                for record in page.items {
                    let Some(conversation) = self.harness.conversation(record.id, &self.ctx).await? else {
                        continue;
                    };
                    let status = self.status(record.id);
                    let (thread, _) = threads::thread(
                        &self.harness,
                        &conversation,
                        status,
                        &self.options.cwd,
                        false,
                        &self.ctx,
                    )
                    .await?;
                    data.push(thread);
                }
                reply(&ThreadListResponse {
                    data,
                    next_cursor: page.next.and_then(|cursor| serde_json::to_string(&cursor).ok()),
                    backwards_cursor: None,
                })
            }
            ClientRequest::ThreadRead { params, .. } => {
                let conversation = self.conversation(&params.thread_id).await?;
                let status = self.status(conversation.id());
                let (thread, _) = threads::thread(
                    &self.harness,
                    &conversation,
                    status,
                    &self.options.cwd,
                    params.include_turns,
                    &self.ctx,
                )
                .await?;
                reply(&ThreadReadResponse { thread })
            }
            ClientRequest::TurnStart { params, .. } => {
                let conversation = self.conversation(&params.thread_id).await?;
                let content = self.content(&params.input).await?;
                let agent = self.overrides(params.model.as_deref(), params.cwd.as_deref())?;
                if agent != AgentChange::default() {
                    conversation.configure(agent, &self.ctx).await?;
                }
                let then = self.follow(&conversation).await?;
                let submission = conversation.submit(SubmissionDraft::input(content), &self.ctx).await?;
                let state = match &then {
                    Some(Then::Follow(_, attached)) => Some(attached.state().clone()),
                    _ => self
                        .followed
                        .get(&conversation.id())
                        .map(|followed| followed.state.clone()),
                };
                if let Some(state) = state {
                    threads::lock(&state).translator.requested(submission.id());
                }
                let (value, _) = reply(&TurnStartResponse {
                    turn: in_progress(submission.id().to_string()),
                })?;
                Ok((value, then.into_iter().collect()))
            }
            ClientRequest::TurnSteer { params, .. } => {
                let conversation = self.conversation(&params.thread_id).await?;
                let then = self.follow(&conversation).await?;
                let current = match &then {
                    Some(Then::Follow(_, attached)) => threads::lock(attached.state()).translator.current_turn(),
                    _ => self.followed.get(&conversation.id()).and_then(Followed::current_turn),
                };
                if current.as_deref() != Some(params.expected_turn_id.as_str()) {
                    return Err(Failure::new(
                        INVALID_REQUEST,
                        format!("Turn {} is not running", params.expected_turn_id),
                    ));
                }
                let content = self.content(&params.input).await?;
                let draft = SubmissionDraft::input(content).when_busy(WhenBusy::Steer);
                conversation.submit(draft, &self.ctx).await?;
                let (value, _) = reply(&TurnSteerResponse {
                    turn_id: params.expected_turn_id,
                })?;
                Ok((value, then.into_iter().collect()))
            }
            ClientRequest::TurnInterrupt { params, .. } => {
                let conversation = self.conversation(&params.thread_id).await?;
                let running = self.followed.get(&conversation.id()).and_then(Followed::current_turn);
                if running.as_deref() == Some(params.turn_id.as_str()) {
                    conversation.abort(AbortOptions::default(), &self.ctx).await?;
                } else if let Some(submission) =
                    self.harness.submission(parse_turn(&params.turn_id)?, &self.ctx).await?
                {
                    // A turn not started yet: withdraw its queued input.
                    submission.abort(&self.ctx).await?;
                }
                reply(&TurnInterruptResponse {})
            }
            ClientRequest::PermissionsRead { params, .. } => {
                let conversation = self.conversation(&params.thread_id).await?;
                let state = conversation.permissions(&self.ctx).await?;
                reply(&self.permissions(state))
            }
            ClientRequest::PermissionsUpdate { params, .. } => {
                let PermissionsUpdateParams {
                    thread_id,
                    mode,
                    clear_mode,
                    add,
                    remove,
                    directories,
                } = params;
                let conversation = self.conversation(&thread_id).await?;
                let change = PermissionsChange {
                    mode: match (mode, clear_mode) {
                        (Some(mode), _) => Change::Set(harness_mode(mode)),
                        (None, true) => Change::Clear,
                        (None, false) => Change::Keep,
                    },
                    add: harness_rules(add),
                    remove: harness_rules(remove),
                    directories: directories.map_or(Change::Keep, Change::Set),
                };
                let state = conversation.update_permissions(change, &self.ctx).await?;
                reply(&self.permissions(state))
            }
            ClientRequest::PermissionRulesRead { .. } => {
                let rules = self.harness.permission_rules(&self.ctx).await?;
                reply(&PermissionRulesResponse {
                    rules: protocol_rules(rules),
                })
            }
            ClientRequest::PermissionRulesRemove { params, .. } => {
                self.harness.remove_permission_rule(&params.rule, &self.ctx).await?;
                let rules = self.harness.permission_rules(&self.ctx).await?;
                reply(&PermissionRulesResponse {
                    rules: protocol_rules(rules),
                })
            }
        }
    }

    fn permissions(&self, state: PermissionsState) -> PermissionsResponse {
        PermissionsResponse {
            permissions: Permissions {
                mode: state.mode.map(protocol_mode),
                effective_mode: protocol_mode(state.mode.unwrap_or(self.options.permission_mode)),
                rules: protocol_rules(state.rules),
                directories: state.directories,
            },
        }
    }

    async fn conversation(&self, thread_id: &str) -> Result<Conversation, Failure> {
        self.harness
            .conversation(parse_thread(thread_id)?, &self.ctx)
            .await?
            .ok_or_else(|| Failure::new(INVALID_PARAMS, format!("Unknown thread {thread_id}")))
    }

    fn status(&self, id: ConversationId) -> delta_protocol::v2::ThreadStatus {
        self.followed
            .get(&id)
            .map_or(delta_protocol::v2::ThreadStatus::NotLoaded, Followed::status)
    }

    async fn content(&self, input: &[UserInput]) -> Result<delta_ai::types::UserContent, Failure> {
        user_content(input)
            .await
            .map_err(|error| Failure::new(INVALID_PARAMS, error.0))
    }

    /// The model a request names: `<provider>/<model id>`, or a model id of the default
    /// model's provider.
    fn model(&self, model: &str) -> Result<ModelRef, Failure> {
        if let Some((provider, id)) = model.split_once('/') {
            return Ok(ModelRef::new(provider, id));
        }
        match &self.options.model {
            Some(default) => Ok(ModelRef::new(default.provider.clone(), model)),
            None => Err(Failure::new(
                INVALID_PARAMS,
                format!("Name the provider of model {model} as <provider>/{model}"),
            )),
        }
    }

    fn overrides(&self, model: Option<&str>, cwd: Option<&str>) -> Result<AgentChange, Failure> {
        Ok(AgentChange {
            model: match model {
                Some(model) => Change::Set(self.model(model)?),
                None => Change::Keep,
            },
            cwd: cwd.map_or(Change::Keep, |cwd| Change::Set(cwd.to_owned())),
            ..AgentChange::default()
        })
    }

    async fn attach(&self, conversation: &Conversation) -> Result<Attached, Failure> {
        let cwd = threads::thread_cwd(conversation, &self.options.cwd, &self.ctx).await?;
        Ok(threads::attach(&self.harness, conversation.clone(), cwd, &self.ctx).await?)
    }

    /// Attaches to a thread the connection does not follow yet.
    async fn follow(&self, conversation: &Conversation) -> Result<Option<Then>, Failure> {
        if self.followed.contains_key(&conversation.id()) {
            return Ok(None);
        }
        let attached = self.attach(conversation).await?;
        Ok(Some(Then::Follow(conversation.id(), Box::new(attached))))
    }

    /// The answer to `thread/start` or `thread/fork`, then `thread/started`.
    async fn started(&self, conversation: &Conversation, turns: bool) -> Handled {
        let attached = self.attach(conversation).await?;
        let (thread, info) = threads::thread(
            &self.harness,
            conversation,
            attached.status(),
            &self.options.cwd,
            turns,
            &self.ctx,
        )
        .await?;
        let notification = ServerNotification::ThreadStarted(ThreadStartedNotification { thread: thread.clone() });
        let (value, _) = reply(&threads::session(thread, info))?;
        Ok((
            value,
            vec![
                Then::Follow(conversation.id(), Box::new(attached)),
                Then::Notify(notification),
            ],
        ))
    }
}
