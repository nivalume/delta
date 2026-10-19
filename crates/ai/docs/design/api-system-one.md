# API: typesafe-system-one and cloudflare-workers-ai-system-one

## Purpose

TypeSafe's System One protocol classifies structured state: the client sends `{ state, questions }` and gets `{ answers, usage }` back. Two classifier APIs speak it:

- `typesafe-system-one`: the native protocol, `POST <baseUrl>/systemone` with `{ model, state, questions }`. TypeSafe, OpenRouter (`/api/v1/systemone`), OpenCode Zen (`/inference/v1/systemone`), Vercel AI Gateway (`/typesafe/v1/systemone`) and llama.cpp (`/v1/systemone`) serve it, so their providers use this API with their own base URLs.
- `cloudflare-workers-ai-system-one`: System One models on the Workers AI REST endpoint, `POST /accounts/{account}/ai/run` with `{ model, input: { state, questions } }` inside Cloudflare's API envelope.

Both are thin transports over one shared implementation. The same module set also holds the HTTP plumbing that every classifier API uses (`api::classifier_shared`).

Code:

- `crates/ai/src/api/classifier_shared.rs`: JSON POST with retries and per-attempt timeouts, usage parsing, field checks, error messages.
- `crates/ai/src/api/system_one_shared.rs`: `SystemOneTransport` and `classify_system_one`.
- `crates/ai/src/api/typesafe_system_one.rs`: `TypeSafeSystemOneApi` (implements `ProviderClassifier`), `classify(model, context, options)`, `API`.
- `crates/ai/src/api/cloudflare_workers_ai_system_one.rs`: `CloudflareWorkersAiSystemOneApi`, `classify`, `API`.

## Upstream source files

- `packages/ai/src/api/classifier-shared.ts` → `api::classifier_shared`.
- `packages/ai/src/api/system-one-shared.ts` → `api::system_one_shared`.
- `packages/ai/src/api/typesafe-system-one.ts` (+ `.lazy.ts`) → `api::typesafe_system_one`.
- `packages/ai/src/api/cloudflare-workers-ai-system-one.ts` (+ `.lazy.ts`) → `api::cloudflare_workers_ai_system_one`.
- `packages/ai/src/providers/typesafe.ts` → `providers::builtin::typesafe_provider()`.
- The `classifiers` maps of `providers/openrouter.ts`, `opencode.ts`, `vercel-ai-gateway.ts`, `cloudflare-workers-ai.ts` → `providers::builtin::with_operations` and `providers::cloudflare::cloudflare_workers_ai_provider()`.
- Tests: `test/typesafe-system-one.test.ts` → `tests/typesafe_system_one.rs`; `test/cloudflare-workers-ai-system-one.test.ts` → `tests/cloudflare_workers_ai_system_one.rs`; the TypeSafe-compatible endpoint cases of `test/classifier-models.test.ts` → `tests/classifier_models.rs`.

## Shared classifier plumbing (`classifier_shared`)

`post_classifier_request(label, url, model, body, options, no_retry_statuses)` (upstream `postClassifierRequest`):

1. Fails with `No API key for provider: <provider>` when `options.base.api_key` is missing or empty.
2. Runs `on_payload`; a returned payload replaces the body.
3. Headers: `content-type: application/json`, `authorization: Bearer <key>`, then `model.headers`, then `options.base.headers`. Names merge case-insensitively and a `None` value deletes the header (upstream `providerHeadersToRecord`).
4. `post_json` sends the request (below), then `on_response` gets the status and headers of the successful response.

`post_json(label, url, headers, payload, options, no_retry_statuses)` is the request loop of upstream's `retryProviderRequest` around `fetch`:

- Each attempt gets its own `options.timeout`, covering the request and reading the body. A timed-out attempt fails with `Request timed out after <ms>ms`. Without `timeout` there is no time limit, as upstream.
- A non-2xx response fails with `Error::Status { message: "<label> returned <status>", body }` (body trimmed and capped at 4000 characters).
- Retries: up to `options.max_retries` (default 2). HTTP errors are retried with the vendor SDK policy (`x-should-retry`, then 408, 409, 429 and 5xx); timeouts are retried too (upstream: an error without status counts as retryable). `no_retry_statuses` fail at once. Network errors and invalid JSON are not retried: upstream retries only errors that carry `status` and `headers` fields.
- The delay is `retry-after-ms` / `retry-after`, else exponential backoff with jitter. A server delay above `max_retry_delay` (default 60 s, zero disables the cap) fails with `Server requested <n>s retry delay (max: <m>s). <label> returned <status>`.
- Cancellation (`options.cancel`) ends the request, the body read or the backoff sleep with `Error::Aborted`.

`parse_classifier_usage(value, model)` (upstream `parseClassifierUsage`): `None` unless `usage` is an object with `input_tokens` or `output_tokens`. A count that is not a positive finite number counts as 0. `total_tokens` is input + output, and the cost comes from `calculate_cost` with the catalog price (tiers included).

`format_classifier_error(error, prefix)` reproduces `formatProviderError(normalizeProviderError(error), prefix)` for the errors these APIs raise:

| Error | Message |
| --- | --- |
| HTTP error with a body the message does not contain | `<prefix> (<status>): <body>` |
| HTTP error without a body | `<prefix> (<status>): <label> returned <status>` |
| Cancellation | `Request aborted` (upstream's `createAbortError`) |
| Anything else | the error's message, unchanged |

`fail_result` clears the answers, sets stop reason `aborted` when the cancellation token fired (else `error`) and keeps any usage already set.

## System One mapping (`system_one_shared`)

`SystemOneTransport` holds what differs per service, as fn pointers: `api`, `label` (used in error messages), `url(model)`, `payload(model, request)` and `output(body)`.

`classify_system_one(transport, model, context, options)`:

1. `model.api` must equal `transport.api`, else `Unsupported classifier API: <api>`.
2. Images are rejected before any request: `<label> does not support image input`.
3. The wire request is `{ state, questions }`. Each question is serialized as in the public contract, except that `bool` questions get `type: "noul"` (TypeSafe's wire name). `options.temperature` is ignored: System One has no such field.
4. `post_classifier_request` with no extra no-retry statuses.
5. `transport.output(body)` extracts `{ answers, usage }`.
6. Usage is parsed and stored before the answers, so a billed request with malformed answers keeps its usage.
7. Answers are read in question order. Every question needs an object answer, else `<label> did not return an answer for <id>`:

| Question | Wire answer | Result |
| --- | --- | --- |
| `choice` | `{ type: "choice", choice: string, probabilities: { key: number }, confidence: number }` | `Choice { choice, probabilities, confidence }` |
| `score` | `{ type: "score", score: number, confidence: number }` | `Score { score, confidence }` |
| `bool` | `{ type: "noul", noul: number }` | `Bool { probability: noul }` |

Wrong types fail with `<label> did not return a <choice|score|bool> answer for <id>`; non-object probabilities with `<label> returned invalid probabilities for <id>`; numbers that are missing or not finite with `<label> returned an invalid <field>` (`probability for <id>.<key>`, `confidence for <id>`, `score for <id>`, `probability for <id>`).

Errors become an error result with prefix `<label> error` (see the table above).

## Transports

| | `typesafe-system-one` | `cloudflare-workers-ai-system-one` |
| --- | --- | --- |
| Label | `System One API` | `Cloudflare Workers AI` |
| URL | `<baseUrl without trailing slashes>/systemone` | `<baseUrl without trailing slashes>/run` |
| Payload | `{ model, state, questions }` | `{ model, input: { state, questions } }` |
| Output | the body, which must be an object | see below |

URLs are resolved like upstream's `new URL(path, base + "/")` (`endpoint_url`, with the `url` crate); an unparsable base URL fails with `Invalid URL`.

The Cloudflare output unwraps the REST envelope:

1. The body must be an object, else `Cloudflare Workers AI returned an unexpected response`.
2. `success: false` fails with `Cloudflare Workers AI error: <messages of errors[], joined with "; ">`, or `Cloudflare Workers AI request failed` when no error has a string message.
3. `result` must be an object. When it has `answers`, it is the output (Cloudflare-hosted models such as `@cf/cloudflare/clef`).
4. Otherwise it is a run record (third-party models such as `typesafe/jev`): `state` must be `"Completed"`, else `Cloudflare Workers AI run did not complete (state: <String(state)>)`; its `result` object is the output.

## Providers

| Provider | Classifier APIs | Base URL source |
| --- | --- | --- |
| `typesafe` (new, `TYPESAFE_API_KEY`, no chat API) | `typesafe-system-one` | catalog (`https://api.typesafe.ai/v1/`) |
| `openrouter` | `typesafe-system-one` (next to `openrouter-images`) | catalog (`https://openrouter.ai/api/v1`) |
| `opencode` | `typesafe-system-one` | catalog (`https://opencode.ai/inference/v1`) |
| `vercel-ai-gateway` | `typesafe-system-one` | catalog (`https://ai-gateway.vercel.sh/typesafe/v1`) |
| `cloudflare-workers-ai` | `cloudflare-workers-ai-system-one`, wrapped in `CloudflareClassifier` | catalog, with `{CLOUDFLARE_ACCOUNT_ID}` filled in from the resolved env |

The table-driven providers get their image and classifier implementations from `with_operations(id, options)` in `providers::builtin`, applied in `catalog_provider`. `typesafe_provider()` uses `ProviderApi::none()`. See [cloudflare.md](cloudflare.md) for `CloudflareClassifier`.

## Rust-specific differences

- `fetch` is replaced by `reqwest` (`options.base.http_client` or the shared client). The per-attempt `AbortSignal.timeout` becomes `tokio::time::timeout` around sending the request and reading the body.
- Messages of network and JSON errors are reqwest's and serde_json's text, where upstream shows the `fetch` / `SyntaxError` text.
- Token counts are `u64`: a fractional count is truncated.
- `SystemOneTransport` is a struct of fn pointers instead of an object with methods.
- Questions and answers keep insertion order (`IndexMap`). JavaScript objects list integer-like keys first, so a request with question ids such as `"1"` is sent in a different order upstream. The answers are keyed by id either way. Ids such as `__proto__` need no special care in Rust; the upstream tests for them are ported and pass.
- Upstream resolves `retry-after-ms` with `Number.parseFloat` and accepts negative values (they sleep 0 ms). `http::retry_after` ignores negative values and falls back to `retry-after` or backoff.

## Improvements pending confirmation

- Network errors (connection refused, reset) are not retried, as upstream: its retry helper only retries errors with `status` and `headers` fields, and `fetch` errors have neither. The chat APIs retry connection errors (the vendor SDKs do). Retrying them here too would make the classifier APIs consistent.
