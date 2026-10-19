# Built-in model catalog and faux provider (`providers::catalog`, `providers::faux`)

## Model catalog

Upstream generates each provider's catalog at build time from models.dev and other sources (`src/providers/data/<provider>.json`, not checked in). The npm package ships the generated output. delta does not duplicate upstream's generation scripts; it uses the data from the npm package directly:

- `crates/ai/data/providers/*.json`: copied from the `@earendil-works/pi-ai` npm package, currently 1.3.0, with 42 providers and 1714 model entries (chat 1614, image 60, classifier 40).
- `scripts/sync-models.sh [version]`: re-syncs from the npm package of the given version.
- The file format is the same as upstream: grouped by API id, with keys `"<type>:<model id>"`.

The `catalog` module compiles the data into the binary with `include_str!`. Each provider's file is parsed on first access and cached (`OnceLock`). A single entry that fails to parse is skipped after logging a `tracing` warning.

| Function | Upstream |
| --- | --- |
| `builtin_provider_ids()` | `getBuiltinProviders()` |
| `all_builtin_models(provider)` | `getAllBuiltinModels(provider)` |
| `builtin_models(provider)` | `getBuiltinModels(provider)` |
| `builtin_model(provider, id)` | `getBuiltinModel(provider, id)` |
| `builtin_model_of_type(type, provider, id)` | `getBuiltinImageModel` / `getBuiltinClassifierModel` |

Upstream's static lookups have literal types (provider and model ids autocomplete). Rust uses strings and does not generate enums.

Tests check that every entry in every file parses as an `AnyModel`.

## faux provider

Upstream: `src/providers/faux.ts`. An in-memory provider for tests and demos that returns scripted responses.

```rust
let faux = FauxProvider::new(FauxOptions::default());
models.set_provider(faux.clone());
faux.set_responses(vec![faux_assistant_message(vec![faux_text("Hello!")]).into()]);
```

Behavior shared with upstream:

- Responses are dequeued in the order requests start. When the queue is empty, it returns the error message "No more faux responses queued".
- `set_responses` replaces the queue; `append_responses` appends to it. `FauxResponse::Factory` builds a response from the request context.
- Text, thinking and tool arguments are streamed in chunks of 3–5 tokens (4 characters per token). When `tokens_per_second` is set, chunks are emitted at that real-time rate.
- Usage is estimated at 1 token per 4 characters. When `session_id` is set and cache retention is not `none`, prompt caching is simulated (the common prefix with the previous request counts as a cache read).
- The cancellation token is checked between chunks. On abort, partial content is kept.
- Helper functions: `faux_text`, `faux_thinking`, `faux_tool_call`, `faux_assistant_message` (the stop reason is `toolUse` when there are tool calls).

- Deferred responses (`FauxOptions::deferred`, `stream_simple` with `deferred`, `fetch_deferred`, `cancel_deferred`, `FauxState::deferred_fetch_count` / `cancelled_deferred`): see [images-classifiers-deferred.md](images-classifiers-deferred.md#faux-provider). As upstream, the faux provider has no image or classifier support.

Differences from upstream: `FauxProvider` itself implements `Provider` and is cloneable (clones share the queue); there is no separate handle and provider. Chunk sizes use a simple xorshift pseudo-random generator instead of adding a `rand` dependency.
