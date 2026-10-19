//! Cloudflare endpoint templates (upstream `api/cloudflare.ts`).
//!
//! `{CLOUDFLARE_ACCOUNT_ID}` and `{CLOUDFLARE_GATEWAY_ID}` are placeholders. The Cloudflare
//! providers fill them in from the resolved provider env before dispatch, see
//! [`crate::providers::cloudflare::resolve_cloudflare_base_url`].

/// Workers AI direct endpoint.
pub const CLOUDFLARE_WORKERS_AI_BASE_URL: &str =
    "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1";

/// Workers AI REST endpoint root; model runs are posted to `run`.
pub const CLOUDFLARE_WORKERS_AI_REST_BASE_URL: &str =
    "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai";

/// AI Gateway Unified API. <https://developers.cloudflare.com/ai-gateway/usage/unified-api/>
pub const CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL: &str =
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat";

/// AI Gateway → OpenAI passthrough. Used until `/compat` supports `/v1/responses`.
pub const CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL: &str =
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai";

/// AI Gateway → Anthropic passthrough.
pub const CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL: &str =
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/anthropic";
