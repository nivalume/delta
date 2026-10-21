//! Permissions on the surface: a conversation's mode and rules, and the rules approvals
//! added harness-wide (`docs/design/permissions.md`).

use delta_chord::context::Context;
use delta_chord::json;

use super::{Conversation, Harness, HarnessError};
use crate::approval::{
    ApprovalError, PermissionMode, PermissionRules, PermissionRulesDoc, PermissionsChange, PermissionsDoc,
    PermissionsState, edit_permission_rules, edit_permissions,
};
use crate::documents::DocAddress;

impl Conversation {
    /// The conversation's permissions; the empty state when none were set.
    pub async fn permissions(&self, ctx: &Context) -> Result<PermissionsState, HarnessError> {
        let address = DocAddress::<PermissionsDoc>::new(self.id);
        match self.inner.session.snapshot(address, ctx).await? {
            Some(value) => json::from_json(&value).map_err(HarnessError::Document),
            None => Ok(PermissionsState::default()),
        }
    }

    /// Changes the conversation's mode, rules or directories in one commit; returns the new
    /// state. An added rule that does not parse is [`ApprovalError::InvalidRule`].
    pub async fn update_permissions(
        &self,
        change: PermissionsChange,
        ctx: &Context,
    ) -> Result<PermissionsState, HarnessError> {
        change
            .validate()
            .map_err(|error| HarnessError::Approval(ApprovalError::InvalidRule(error)))?;
        let conversation = self.id;
        Ok(self
            .inner
            .session
            .commit(ctx, move |tx| {
                Box::pin(async move {
                    edit_permissions(tx, conversation, |state| {
                        change.apply(state);
                        state.clone()
                    })
                    .await
                })
            })
            .await?)
    }

    /// Sets the conversation's permission mode.
    pub async fn set_permission_mode(
        &self,
        mode: PermissionMode,
        ctx: &Context,
    ) -> Result<PermissionsState, HarnessError> {
        self.update_permissions(PermissionsChange::mode(mode), ctx).await
    }
}

impl Harness {
    /// The allow rules approvals added (`ApprovedWithRule`), for every conversation.
    pub async fn permission_rules(&self, ctx: &Context) -> Result<PermissionRules, HarnessError> {
        let address = DocAddress::<PermissionRulesDoc>::new(());
        match self.inner.session.snapshot(address, ctx).await? {
            Some(value) => json::from_json(&value).map_err(HarnessError::Document),
            None => Ok(PermissionRules::default()),
        }
    }

    /// Removes `rule` from the rules approvals added; whether it was there.
    pub async fn remove_permission_rule(&self, rule: &str, ctx: &Context) -> Result<bool, HarnessError> {
        let rule = rule.trim().to_owned();
        Ok(self
            .inner
            .session
            .commit(ctx, move |tx| {
                Box::pin(async move {
                    edit_permission_rules(tx, |rules| {
                        let before = rules.allow.len();
                        rules.allow.retain(|allowed| *allowed != rule);
                        rules.allow.len() != before
                    })
                    .await
                })
            })
            .await?)
    }
}
