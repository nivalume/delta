# API: openrouter-images

## Purpose

`openrouter-images` generates images through OpenRouter's chat completions endpoint: one non-streaming `POST <baseUrl>/chat/completions` whose `modalities` ask for images. It is the only image API upstream ships; the built-in `openrouter` provider registers it for its image models.

Upstream sends the request with the OpenAI SDK. delta has no official OpenAI Rust SDK, so the module sends the same request on `reqwest` + `delta_ai::http` and reproduces the SDK's error messages.

Code: `crates/ai/src/api/openrouter_images.rs` with `OpenRouterImagesApi` (implements `ProviderImages`), `generate_images(model, context, options)` and `API`.

## Upstream source files

- `packages/ai/src/api/openrouter-images.ts` (+ `.lazy.ts`).
- The `images` map of `packages/ai/src/providers/openrouter.ts` → `providers::builtin::with_operations`.
- Tests: `test/openrouter-images.test.ts` (it mocks the OpenAI SDK; the port uses a mock HTTP server) and `test/images.test.ts` (live end to end, ported as a local-server test of the built-in provider) → `tests/openrouter_images.rs`; the `provider.generateImages` check of `test/images-models.test.ts` → `tests/images_models.rs`.

## Request mapping

1. `options.base.api_key` is required: `No API key for provider: <provider>`.
2. Body (upstream `buildParams`):

```json
{
  "model": "<model.id>",
  "messages": [{ "role": "user", "content": [
    { "type": "text", "text": "..." },
    { "type": "image_url", "image_url": { "url": "data:<mime>;base64,<data>" } }
  ] }],
  "stream": false,
  "modalities": ["image", "text"]
}
```

`modalities` is `["image", "text"]` when `model.output` includes text, else `["image"]`. `on_payload` may replace the body.

3. Headers: `Accept` and `Content-Type: application/json`, `Authorization: Bearer <key>`, `User-Agent`, then the custom headers, which override the defaults case-insensitively. As upstream (`{...model.headers, ...options.headers}` handed to the SDK as default headers), the custom headers are `model.headers` overlaid by `options.base.headers` with case-sensitive names, and `None` values are dropped: a `None` removes a model header of the same spelling but cannot remove a default such as `Authorization`.
4. `http::send_with_retry`. Upstream runs the SDK with `maxRetries: 0` inside `retryProviderRequest`, whose default is 0 retries, so a request is retried only when `max_retries` is set. The timeout is `options.base.timeout`, else the SDK's 10 minutes.
5. `on_response` with the status and headers.

## Response mapping

- `id` → `response_id`.
- `usage` (when it is an object) → `usage`, priced at the model's base rates (upstream `parseUsage` does not apply cost tiers): `cache_write` = `prompt_tokens_details.cache_write_tokens`; `cache_read` = `cached_tokens`, minus the cache writes when there are any; `input` = `prompt_tokens` − cache read − cache write (at least 0); `output` = `completion_tokens`.
- The first choice's `message.content`, when a non-empty string, becomes a text block.
- Each `message.images[].image_url` (a string or `{ url }`) that is a `data:<mime>;base64,<data>` URL becomes an image block. Other URLs are skipped.
- No choices: an empty output with stop reason `stop`.

## Errors

Every failure returns a result with stop reason `error`, or `aborted` when `options.base.cancel` fired; cancellation reads `Request aborted`, as upstream's retry helper reports it. HTTP errors are formatted as upstream's `formatProviderError(normalizeProviderError(error))` on an OpenAI SDK `APIError`: the SDK message (`<status> <error.message>`), or `<status>: <JSON of the body's error object>` when the message lacks it.

## Rust-specific differences

- No OpenAI SDK: the request is sent directly. The SDK's own headers (`X-Stainless-*`) are not sent.
- Connection errors and timeouts carry reqwest's message, where the SDK says `Connection error.` / `Request timed out.`. Both are retried when `max_retries` is set, as the SDK errors are.
- A response without a `choices` array, or a first choice without a `message` object, fails with `OpenRouter returned a response without choices` / `OpenRouter returned a choice without a message`; upstream fails with a JavaScript `TypeError`. A `message.images` value that is not an array is ignored (upstream throws).
- `sanitizeSurrogates` has no counterpart: Rust strings cannot hold lone surrogates.
- Token counts are `u64`: fractional counts are truncated and negative counts read as 0.
- `ImagesOptions::metadata` is not sent, as upstream ignores it for this API.

## Improvements pending confirmation

- The API retries only when `max_retries` is set (upstream default 0), while the chat APIs default to 2 retries. Defaulting to 2 here too would make image generation as robust as chat.
