//! Explicit invocation contexts: cancellation with a reason, and typed values.
//!
//! A [`Context`] is passed to every operation of a run. Deriving one never changes its parent.
//! Cancellation is built on [`CancellationToken`], as in `delta_ai`, so [`Context::token`] can be
//! handed to a model request. See `docs/design/context.md`.

use std::any::Any;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, OnceLock};

use futures::future::select_all;
use tokio_util::sync::{CancellationToken, DropGuard};

/// Typed identity of one value carried by a [`Context`]. Declare keys as statics; two keys are
/// different keys even when their types are the same.
///
/// ```
/// use delta_chord::context::{Context, ContextKey};
///
/// static USER: ContextKey<String> = ContextKey::new("user");
///
/// let context = Context::background().with_value(&USER, "ann".to_owned());
/// assert_eq!(context.value(&USER).map(String::as_str), Some("ann"));
/// ```
pub struct ContextKey<T> {
    name: &'static str,
    marker: PhantomData<fn() -> T>,
}

impl<T> ContextKey<T> {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            marker: PhantomData,
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    fn id(&'static self) -> usize {
        std::ptr::from_ref(self) as usize
    }
}

impl<T> fmt::Debug for ContextKey<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ContextKey").field(&self.name).finish()
    }
}

/// Why a context was cancelled.
#[derive(Clone)]
pub struct CancelReason(Arc<dyn Error + Send + Sync>);

impl CancelReason {
    pub fn new(error: impl Error + Send + Sync + 'static) -> Self {
        Self(Arc::new(error))
    }

    /// A reason that is only a message.
    pub fn message(message: impl Into<String>) -> Self {
        Self(Arc::from(Box::<dyn Error + Send + Sync>::from(message.into())))
    }

    /// The reason of a cancellation that gave none (a bare token, or [`Canceller::cancel`]).
    pub fn aborted() -> Self {
        Self::new(Aborted)
    }

    pub fn error(&self) -> &(dyn Error + Send + Sync + 'static) {
        &*self.0
    }

    pub fn downcast_ref<E: Error + 'static>(&self) -> Option<&E> {
        self.0.downcast_ref()
    }
}

impl fmt::Debug for CancelReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Display for CancelReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// The reason of a cancellation that gave none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("The operation was aborted")]
pub struct Aborted;

/// What [`Context::run`] returns when the context is cancelled first.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{reason}")]
pub struct Cancelled {
    pub reason: CancelReason,
}

/// Cancels the context made with it by [`Context::with_cancel`], and the contexts derived from
/// that one.
#[derive(Debug, Clone)]
pub struct Canceller {
    token: CancellationToken,
    reason: Arc<OnceLock<CancelReason>>,
}

impl Canceller {
    /// Cancel without a specific reason ([`Aborted`]).
    pub fn cancel(&self) {
        self.cancel_with(CancelReason::aborted());
    }

    /// Cancel with `reason`. A second cancellation keeps the first reason.
    pub fn cancel_with(&self, reason: CancelReason) {
        let _ = self.reason.set(reason);
        self.token.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }
}

/// An immutable, cheaply cloned value passed explicitly through operations.
#[derive(Clone, Default)]
pub struct Context {
    values: Option<Arc<ValueNode>>,
    cancel: Option<Arc<CancelState>>,
}

struct ValueNode {
    key: usize,
    name: &'static str,
    value: Box<dyn Any + Send + Sync>,
    parent: Option<Arc<ValueNode>>,
}

struct CancelState {
    /// The context is cancelled when any of these is. Usually one: contexts made by
    /// `with_cancel` use child tokens, which their parent cancels synchronously.
    tokens: Vec<CancellationToken>,
    /// Canceller tokens with their reasons, closest first.
    reasons: Vec<(CancellationToken, Arc<OnceLock<CancelReason>>)>,
    /// One token that is cancelled when any of `tokens` is; joined by a task on first use when
    /// there are several.
    merged: OnceLock<CancellationToken>,
    /// Ends the task that feeds `merged` when the last context sharing this state is dropped.
    watcher: Mutex<Option<DropGuard>>,
    /// Keeps the parent's task alive while a context whose token is a child of `merged` lives.
    _parent: Option<Arc<CancelState>>,
}

impl CancelState {
    fn token(&self) -> CancellationToken {
        if let [token] = &self.tokens[..] {
            return token.clone();
        }
        self.merged
            .get_or_init(|| {
                let merged = CancellationToken::new();
                let alive = CancellationToken::new();
                let sources = self.tokens.clone();
                let target = merged.clone();
                let stop = alive.clone();
                tokio::spawn(async move {
                    let any = select_all(sources.iter().map(|token| Box::pin(token.cancelled())));
                    tokio::select! {
                        _ = any => target.cancel(),
                        () = stop.cancelled() => {}
                    }
                });
                *self.watcher.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(alive.drop_guard());
                merged
            })
            .clone()
    }
}

impl Context {
    /// A context that is never cancelled and carries no values.
    pub fn background() -> Self {
        Self::default()
    }

    /// Derive a context carrying `value` under `key`, replacing an earlier value of the key.
    #[must_use]
    pub fn with_value<T: Send + Sync + 'static>(&self, key: &'static ContextKey<T>, value: T) -> Self {
        let node = ValueNode {
            key: key.id(),
            name: key.name,
            value: Box::new(value),
            parent: self.values.clone(),
        };
        Self {
            values: Some(Arc::new(node)),
            cancel: self.cancel.clone(),
        }
    }

    /// The value of `key`, from this context or the closest ancestor that set it.
    pub fn value<T: Send + Sync + 'static>(&self, key: &'static ContextKey<T>) -> Option<&T> {
        let id = key.id();
        let mut node = self.values.as_deref();
        while let Some(current) = node {
            if current.key == id {
                return current.value.downcast_ref();
            }
            node = current.parent.as_deref();
        }
        None
    }

    /// Derive a context that is cancelled with this one or by the returned canceller.
    ///
    /// Needs a tokio runtime when this context was joined with an outside token
    /// ([`with_token`](Self::with_token)), for the same reason as [`token`](Self::token).
    #[must_use]
    pub fn with_cancel(&self) -> (Self, Canceller) {
        let token = match &self.cancel {
            None => CancellationToken::new(),
            Some(state) => state.token().child_token(),
        };
        let reason = Arc::new(OnceLock::new());
        let mut tokens = vec![token.clone()];
        let mut reasons = vec![(token.clone(), reason.clone())];
        let merged = OnceLock::new();
        if let Some(state) = &self.cancel {
            reasons.extend(state.reasons.iter().cloned());
            if state.tokens.len() > 1 {
                // The child token follows the parent's joining task; checking the parent's own
                // tokens too keeps `is_cancelled` exact before that task has run.
                tokens.extend(state.tokens.iter().cloned());
                let _ = merged.set(token.clone());
            }
        }
        let state = CancelState {
            tokens,
            reasons,
            merged,
            watcher: Mutex::new(None),
            _parent: self.cancel.clone(),
        };
        (
            Self {
                values: self.values.clone(),
                cancel: Some(Arc::new(state)),
            },
            Canceller { token, reason },
        )
    }

    /// Derive a context that is cancelled with this one or by `token`.
    ///
    /// [`token`](Self::token) on the result needs a tokio runtime when this context could
    /// already be cancelled: it spawns a small task that joins both, which ends when the last
    /// context sharing it is dropped.
    #[must_use]
    pub fn with_token(&self, token: CancellationToken) -> Self {
        let mut tokens = Vec::new();
        let mut reasons = Vec::new();
        if let Some(state) = &self.cancel {
            tokens.extend(state.tokens.iter().cloned());
            reasons.extend(state.reasons.iter().cloned());
        }
        tokens.push(token);
        let state = CancelState {
            tokens,
            reasons,
            merged: OnceLock::new(),
            watcher: Mutex::new(None),
            _parent: None,
        };
        Self {
            values: self.values.clone(),
            cancel: Some(Arc::new(state)),
        }
    }

    /// Derive a context with the same values that is never cancelled. For work that must
    /// finish, such as cleanup and rollback.
    #[must_use]
    pub fn without_cancel(&self) -> Self {
        Self {
            values: self.values.clone(),
            cancel: None,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(|state| state.tokens.iter().any(CancellationToken::is_cancelled))
    }

    /// Why the context was cancelled: the reason of the closest canceller that gave one, or
    /// [`Aborted`]. `None` while it is not cancelled.
    pub fn reason(&self) -> Option<CancelReason> {
        if !self.is_cancelled() {
            return None;
        }
        let state = self.cancel.as_ref()?;
        let given = state
            .reasons
            .iter()
            .filter(|(token, _)| token.is_cancelled())
            .find_map(|(_, reason)| reason.get().cloned());
        Some(given.unwrap_or_else(CancelReason::aborted))
    }

    /// Resolves when the context is cancelled; never for a context that cannot be.
    pub async fn cancelled(&self) {
        let Some(state) = &self.cancel else {
            return std::future::pending().await;
        };
        if let [token] = &state.tokens[..] {
            token.cancelled().await;
        } else {
            select_all(state.tokens.iter().map(|token| Box::pin(token.cancelled()))).await;
        }
    }

    /// Run `future` until it completes or the context is cancelled. On cancellation the future
    /// is dropped, which cancels it.
    pub async fn run<F: Future>(&self, future: F) -> Result<F::Output, Cancelled> {
        if let Some(reason) = self.reason() {
            return Err(Cancelled { reason });
        }
        tokio::select! {
            biased;
            () = self.cancelled() => Err(Cancelled { reason: self.reason().unwrap_or_else(CancelReason::aborted) }),
            output = future => Ok(output),
        }
    }

    /// A token that is cancelled when this context is, for APIs that take a
    /// [`CancellationToken`] (`delta_ai` request options, `delta_agent` runs).
    pub fn token(&self) -> CancellationToken {
        self.cancel
            .as_ref()
            .map_or_else(CancellationToken::new, |state| state.token())
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut names = Vec::new();
        let mut node = self.values.as_deref();
        while let Some(current) = node {
            names.push(current.name);
            node = current.parent.as_deref();
        }
        names.reverse();
        f.debug_struct("Context")
            .field("values", &names)
            .field("cancellable", &self.cancel.is_some())
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}
