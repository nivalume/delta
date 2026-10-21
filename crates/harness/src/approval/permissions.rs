//! Permission modes and rules: the `delta.permissions` and `delta.permission_rules`
//! documents, and [`RulesPolicy`], the [`ApprovalPolicy`] over them
//! (`docs/design/permissions.md`).

use delta_chord::context::Context;
use delta_chord::delta::Op;
use delta_chord::json::{self, Json};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

use super::policy::{ApprovalCall, ApprovalPolicy, PolicyDecision, PolicyError};
use super::rules::{Anchors, Rule, RuleCall, RuleError, allowed_by, is_within, propose};
use crate::agent::Change;
use crate::documents::{
    CheckpointInfo, ConversationScope, DocAddress, DocDefinition, ForkAsOf, Rewindable, SessionScope,
};
use crate::records::ConversationRecord;
use crate::registry::HookApi;
use crate::session::{Tx, TxError};
use crate::tool::{PathAccess, PermissionSubject};

/// How much a conversation's calls may do without asking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    /// Reads inside the working directories run; anything else not allowed by a rule asks.
    #[default]
    Default,
    /// Also writes inside the working directories run.
    AcceptEdits,
    /// Commands and writes are denied; reads as in `Default`.
    Plan,
    /// What would ask is denied.
    DontAsk,
    /// Everything runs, except what a deny or ask rule names.
    Bypass,
}

/// Rules as written, by what they decide.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionRules {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ask: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deny: Vec<String>,
}

impl PermissionRules {
    /// Every rule parses.
    pub fn validate(&self) -> Result<(), RuleError> {
        for rule in self.allow.iter().chain(&self.ask).chain(&self.deny) {
            rule.parse::<Rule>()?;
        }
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.allow.is_empty() && self.ask.is_empty() && self.deny.is_empty()
    }
}

/// The value of a `delta.permissions` document.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionsState {
    /// `None`: the policy's default mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<PermissionMode>,
    #[serde(default, skip_serializing_if = "PermissionRules::is_empty")]
    pub rules: PermissionRules,
    /// More directories reads (and `accept_edits` writes) reach: absolute, or relative to
    /// the call's directory.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub directories: Vec<String>,
}

/// A change of a conversation's permissions; rules are added and removed by their text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PermissionsChange {
    pub mode: Change<PermissionMode>,
    pub add: PermissionRules,
    pub remove: PermissionRules,
    pub directories: Change<Vec<String>>,
}

impl PermissionsChange {
    /// The change setting the mode only.
    pub fn mode(mode: PermissionMode) -> Self {
        Self {
            mode: Change::Set(mode),
            ..Self::default()
        }
    }

    /// Every added rule parses.
    pub fn validate(&self) -> Result<(), RuleError> {
        self.add.validate()
    }

    /// Applies the change: removals first, then additions not already present.
    pub fn apply(self, state: &mut PermissionsState) {
        self.mode.apply(&mut state.mode);
        let mut directories = Some(std::mem::take(&mut state.directories));
        self.directories.apply(&mut directories);
        state.directories = directories.unwrap_or_default();
        let lists = [
            (&mut state.rules.allow, self.remove.allow, self.add.allow),
            (&mut state.rules.ask, self.remove.ask, self.add.ask),
            (&mut state.rules.deny, self.remove.deny, self.add.deny),
        ];
        for (list, remove, add) in lists {
            list.retain(|rule| !remove.iter().any(|removed| removed.trim() == rule.trim()));
            for rule in add {
                let rule = rule.trim().to_owned();
                if !list.contains(&rule) {
                    list.push(rule);
                }
            }
        }
    }
}

/// The `delta.permissions` document: conversation scope, rewindable, a fork copies it as of
/// the fork entry, a complete base on every change, `{}` initially.
#[derive(Clone, Copy, Debug, Default)]
pub struct PermissionsDoc;

impl DocDefinition for PermissionsDoc {
    const KIND: &'static str = "delta.permissions";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Rewindable, ForkAsOf>;
    type Seed = ();

    fn initial((): &()) -> Json {
        delta_chord::json!({})
    }

    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        true
    }
}

/// The `delta.permission_rules` document: session scope, the rules approvals added
/// (`ReviewDecision::ApprovedWithRule`), a complete base on every change.
#[derive(Clone, Copy, Debug, Default)]
pub struct PermissionRulesDoc;

impl DocDefinition for PermissionRulesDoc {
    const KIND: &'static str = "delta.permission_rules";
    const VERSION: u32 = 1;
    type Scope = SessionScope;
    type Seed = ();

    fn initial((): &()) -> Json {
        delta_chord::json!({})
    }

    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        true
    }
}

/// Edits `conversation`'s `delta.permissions` as a typed state.
pub async fn edit_permissions<R>(
    tx: &mut Tx<'_>,
    conversation: crate::records::ConversationId,
    edit: impl FnOnce(&mut PermissionsState) -> R,
) -> Result<R, TxError> {
    let mut doc = tx.doc(DocAddress::<PermissionsDoc>::new(conversation)).await?;
    doc.update_as::<PermissionsState, R>(edit).map_err(TxError::callback)
}

/// Edits the session's `delta.permission_rules` as typed rules.
pub async fn edit_permission_rules<R>(
    tx: &mut Tx<'_>,
    edit: impl FnOnce(&mut PermissionRules) -> R,
) -> Result<R, TxError> {
    let mut doc = tx.doc(DocAddress::<PermissionRulesDoc>::new(())).await?;
    doc.update_as::<PermissionRules, R>(edit).map_err(TxError::callback)
}

/// A conversation created with an owner starts with a copy of the owner conversation's
/// `delta.permissions`, when it has one; a fork gets its copy from the fork policy.
pub async fn create_permissions(tx: &mut Tx<'_>, record: &ConversationRecord) -> Result<(), TxError> {
    let Some(owner) = record.owner.filter(|_| record.parent.is_none()) else {
        return Ok(());
    };
    let Some((_, value)) = tx
        .peek_doc(DocAddress::<PermissionsDoc>::new(owner.conversation_id))
        .await?
    else {
        return Ok(());
    };
    tx.doc(DocAddress::<PermissionsDoc>::new(record.id)).await?.set(value);
    Ok(())
}

/// What [`RulesPolicy`] starts from: the mode of a conversation that sets none, rules for
/// every conversation, and the anchors of `/path` and `~/path` rules.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PermissionDefaults {
    pub mode: PermissionMode,
    pub rules: PermissionRules,
    pub directories: Vec<String>,
    /// The directory `/path` rules are relative to.
    pub root: Option<String>,
    /// The directory `~/path` rules are relative to; [`RulesPolicy::new`] takes the
    /// process's home directory when unset.
    pub home: Option<String>,
}

/// The memo a decision is kept in, so a rerun before the intent commit decides the same.
const DECISION_MEMO: &str = "delta.permissions:decision";

/// A decision as memoized.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Memoized {
    Allow,
    Deny {
        reason: String,
    },
    Ask {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        proposed_rule: Option<String>,
    },
}

impl From<Memoized> for PolicyDecision {
    fn from(decision: Memoized) -> Self {
        match decision {
            Memoized::Allow => Self::Allow,
            Memoized::Deny { reason } => Self::Deny { reason },
            Memoized::Ask { prompt, proposed_rule } => Self::Ask {
                timeout: None,
                prompt,
                proposed_rule,
            },
        }
    }
}

/// Claude Code's rules and permission modes as an [`ApprovalPolicy`]: deny rules, then ask
/// rules, then the mode, allow rules and the working directories decide; what remains asks
/// with a proposed rule (or is denied in `dont_ask`).
#[derive(Debug)]
pub struct RulesPolicy {
    defaults: PermissionDefaults,
    rules: ParsedRules,
}

/// Rules parsed, by what they decide.
#[derive(Clone, Debug, Default)]
struct ParsedRules {
    allow: Vec<Rule>,
    ask: Vec<Rule>,
    deny: Vec<Rule>,
}

impl ParsedRules {
    /// The rules that parse; `strict` fails on the first that does not.
    fn parse(rules: &PermissionRules, strict: bool) -> Result<Self, RuleError> {
        let list = |texts: &[String]| -> Result<Vec<Rule>, RuleError> {
            let mut parsed = Vec::new();
            for text in texts {
                match text.parse::<Rule>() {
                    Ok(rule) => parsed.push(rule),
                    Err(error) if strict => return Err(error),
                    Err(_) => {}
                }
            }
            Ok(parsed)
        };
        Ok(Self {
            allow: list(&rules.allow)?,
            ask: list(&rules.ask)?,
            deny: list(&rules.deny)?,
        })
    }
}

impl RulesPolicy {
    /// A policy over `defaults`; fails when a default rule does not parse or needs a root
    /// that is not set.
    pub fn new(mut defaults: PermissionDefaults) -> Result<Self, RuleError> {
        let rules = ParsedRules::parse(&defaults.rules, true)?;
        if defaults.root.is_none()
            && let Some(rule) = rules
                .allow
                .iter()
                .chain(&rules.ask)
                .chain(&rules.deny)
                .find(|rule| rule.needs_root())
        {
            return Err(RuleError::NoRoot(rule.to_string()));
        }
        if defaults.home.is_none() {
            defaults.home = std::env::home_dir().and_then(|home| home.to_str().map(str::to_owned));
        }
        Ok(Self { defaults, rules })
    }

    async fn evaluate(
        &self,
        call: &ApprovalCall<'_>,
        api: &HookApi<'_>,
        ctx: &Context,
    ) -> Result<Memoized, PolicyError> {
        let state: PermissionsState = match api
            .snapshot(DocAddress::<PermissionsDoc>::new(call.conversation_id), ctx)
            .await?
        {
            Some(value) => json::from_json(&value)?,
            None => PermissionsState::default(),
        };
        let session: PermissionRules = match api.snapshot(DocAddress::<PermissionRulesDoc>::new(()), ctx).await? {
            Some(value) => json::from_json(&value)?,
            None => PermissionRules::default(),
        };
        let conversation = ParsedRules::parse(&state.rules, false)?;
        let session = ParsedRules::parse(&session, false)?;
        let mode = state.mode.unwrap_or(self.defaults.mode);
        Ok(self.decide_with(call, mode, &[&self.rules, &session, &conversation], &state.directories))
    }

    /// The decision for `call` under `mode` with the rule sets `sources` and the
    /// conversation's extra `directories`.
    fn decide_with(
        &self,
        call: &ApprovalCall<'_>,
        mode: PermissionMode,
        sources: &[&ParsedRules],
        directories: &[String],
    ) -> Memoized {
        let tool = call.call.call.name.as_str();
        let subject = call.subject.as_ref();
        let rule_call = RuleCall {
            tool,
            subject,
            anchors: Anchors {
                cwd: call.cwd,
                home: self.defaults.home.as_deref(),
                root: self.defaults.root.as_deref(),
            },
        };
        if let Some(rule) = sources
            .iter()
            .flat_map(|rules| &rules.deny)
            .find(|rule| rule.restricts(&rule_call))
        {
            return Memoized::Deny {
                reason: format!("denied by rule {rule}"),
            };
        }
        if let Some(rule) = sources
            .iter()
            .flat_map(|rules| &rules.ask)
            .find(|rule| rule.restricts(&rule_call))
        {
            return Memoized::Ask {
                prompt: Some(format!("{rule} requires approval")),
                proposed_rule: None,
            };
        }
        let changes = matches!(
            subject,
            Some(
                PermissionSubject::Command { .. }
                    | PermissionSubject::Path {
                        access: PathAccess::Write,
                        ..
                    }
            )
        );
        match mode {
            PermissionMode::Bypass => return Memoized::Allow,
            PermissionMode::Plan if changes => {
                return Memoized::Deny {
                    reason: format!("plan mode: {tool} may not run"),
                };
            }
            _ => {}
        }
        if allowed_by(sources.iter().flat_map(|rules| &rules.allow), &rule_call) {
            return Memoized::Allow;
        }
        if let Some(PermissionSubject::Path { path, access }) = subject {
            let reachable = *access == PathAccess::Read || mode == PermissionMode::AcceptEdits;
            let inside = call.cwd.is_some_and(|cwd| is_within(path, cwd, None))
                || self
                    .defaults
                    .directories
                    .iter()
                    .chain(directories)
                    .any(|directory| is_within(path, directory, call.cwd));
            if reachable && inside {
                return Memoized::Allow;
            }
        }
        if mode == PermissionMode::DontAsk {
            return Memoized::Deny {
                reason: format!("{tool} would need approval (dont_ask mode)"),
            };
        }
        Memoized::Ask {
            prompt: None,
            proposed_rule: propose(tool, subject),
        }
    }
}

impl ApprovalPolicy for RulesPolicy {
    fn decide<'a>(
        &'a self,
        call: &'a ApprovalCall<'a>,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<PolicyDecision, PolicyError>> {
        Box::pin(async move {
            if let Some(decided) = api.memo::<Memoized>(DECISION_MEMO)? {
                return Ok(decided.into());
            }
            let decided = self.evaluate(call, api, ctx).await?;
            Ok(api.memo_or_insert(DECISION_MEMO, decided, ctx).await?.into())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changes_apply_removals_then_additions() {
        let mut state = PermissionsState {
            mode: Some(PermissionMode::Plan),
            rules: PermissionRules {
                allow: vec!["Bash(ls)".to_owned(), "Read".to_owned()],
                ..PermissionRules::default()
            },
            directories: vec!["/tmp".to_owned()],
        };
        let change = PermissionsChange {
            mode: Change::Clear,
            add: PermissionRules {
                allow: vec![" Bash(pwd) ".to_owned(), "Read".to_owned()],
                deny: vec!["Read(./.env)".to_owned()],
                ..PermissionRules::default()
            },
            remove: PermissionRules {
                allow: vec!["Bash(ls)".to_owned()],
                ..PermissionRules::default()
            },
            directories: Change::Set(vec!["/data".to_owned()]),
        };
        change.validate().unwrap();
        change.apply(&mut state);
        assert_eq!(state.mode, None);
        assert_eq!(state.rules.allow, ["Read", "Bash(pwd)"]);
        assert_eq!(state.rules.deny, ["Read(./.env)"]);
        assert_eq!(state.directories, ["/data"]);
        let invalid = PermissionsChange {
            add: PermissionRules {
                ask: vec!["Bash(".to_owned()],
                ..PermissionRules::default()
            },
            ..PermissionsChange::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn states_serialize_compactly() {
        let state = PermissionsState {
            mode: Some(PermissionMode::AcceptEdits),
            ..PermissionsState::default()
        };
        assert_eq!(
            serde_json::to_value(&state).unwrap(),
            serde_json::json!({ "mode": "accept_edits" })
        );
        let empty: PermissionsState = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(empty, PermissionsState::default());
    }

    #[test]
    fn defaults_are_validated() {
        let defaults = |deny: &str| PermissionDefaults {
            rules: PermissionRules {
                deny: vec![deny.to_owned()],
                ..PermissionRules::default()
            },
            ..PermissionDefaults::default()
        };
        assert!(RulesPolicy::new(defaults("Bash(rm -rf *)")).is_ok());
        assert_eq!(
            RulesPolicy::new(defaults("Read(/secrets)")).unwrap_err(),
            RuleError::NoRoot("Read(/secrets)".to_owned())
        );
        assert!(RulesPolicy::new(defaults("x(y)")).is_err());
    }
}
