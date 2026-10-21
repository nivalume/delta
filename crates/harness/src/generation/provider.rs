//! `delta.provider`: the stable provider-facing identity of one conversation, sent as the
//! request's `session_id` so providers keep their prompt caches.

use delta_chord::context::Context;
use delta_chord::delta::Op;
use delta_chord::json::Json;

use crate::documents::{CheckpointInfo, ConversationScope, DocAddress, DocDefinition, ForkInitial, Latest};
use crate::scheduler::{RuntimeError, TaskDefinition, TaskRuntime};
use crate::session::TxError;

/// The `delta.provider` document: conversation scope, `latest`, a fork starts with a fresh
/// identity, a complete base on every change; `{ "session_id": <UUIDv7> }` initially.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProviderDoc;

impl DocDefinition for ProviderDoc {
    const KIND: &'static str = "delta.provider";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Latest, ForkInitial>;
    type Seed = ();

    fn initial((): &()) -> Json {
        delta_chord::json!({ "session_id": uuid::Uuid::now_v7().to_string() })
    }

    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        true
    }
}

fn session_id(value: &Json) -> Option<String> {
    value.get("session_id").and_then(Json::as_str).map(str::to_owned)
}

/// The conversation's persisted provider session id, without writing in the normal path; a
/// conversation without `delta.provider` gets one in a commit before the request starts.
pub(crate) async fn ensure_session_id<T: TaskDefinition>(
    rt: &TaskRuntime<T>,
    ctx: &Context,
) -> Result<String, RuntimeError> {
    let address = DocAddress::<ProviderDoc>::new(rt.conversation_id());
    if let Some(id) = rt.snapshot(address.clone(), ctx).await?.as_ref().and_then(session_id) {
        return Ok(id);
    }
    rt.commit(ctx, move |tx, _| {
        Box::pin(async move {
            let doc = tx.doc(address).await?;
            let id = session_id(doc.value()).ok_or_else(|| TxError::callback("delta.provider has no session_id"))?;
            Ok((id, None))
        })
    })
    .await
}
