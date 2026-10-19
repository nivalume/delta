# Transcript processing (`transcript`, `transform`, `estimate`, `simple_options`)

Upstream: `src/utils/transcript.ts`, `src/utils/text.ts`, `src/api/transform-messages.ts`, `src/utils/estimate.ts`, `src/api/simple-options.ts`. All four modules are pure functions and map one-to-one to upstream.

## System messages (`transcript`)

`Context::system_prompt` and `tools` are shorthand for the first system message; `normalize_context` folds them into it. Later system messages can change the prompt and the tool set in the middle of the conversation: `content` appends instructions, `sections` replaces by name (`None` deletes), and `tools_added` / `tools_removed` add and remove tools. Replaying all system messages in order gives the current prompt and tools.

| Function | Purpose |
| --- | --- |
| `normalize_context` | Builds the `TranscriptContext` |
| `get_current_tools` | Replays tool additions and removals; a tool redefined under the same name keeps its original position (as JS `Map.set` does) |
| `get_current_system_message` / `get_current_system_prompt` | Replays the prompt and sections |
| `collapse_system_messages` | For models that do not support mid-conversation system messages: the replayed result becomes the first message, and the other system messages are removed |
| `resolve_transcript(ctx, supports)` | Keeps the transcript as is when supported, otherwise collapses it |
| `render_system_message_update` | Renders the update text for APIs that support mid-conversation system messages |
| `resolve_transcript_tools` | For transports that can anchor added tools: only the initial tools go at the top level; when there are removals or redefinitions, the full current tool list is sent |
| `get_tool_state_changes`, `get_declared_tools`, `has_non_additive_tool_changes`, `declarations_equal` | Tool state comparison |

Upstream's deprecated `hasToolRedefinitions` is not ported. `declarations_equal` compares fields directly (upstream compares via JSON serialization, with the same goal).

## Cross-provider switching (`transform`)

`transform_messages(messages, model, normalize_tool_call_id)`:

- The model does not support images: images in user messages and tool results are replaced with placeholder text (several consecutive images get a single placeholder).
- Assistant messages from another model (any of provider, api, id differs): thinking blocks are removed; text loses its signature; tool calls lose `thought_signature` and optionally get normalized ids (for example, Anthropic requires `^[a-zA-Z0-9_-]{1,64}$`), and later tool results follow the mapping.
- Thinking blocks from the same model: kept if signed or redacted (even with empty text, because encrypted reasoning must be replayed); removed if unsigned with empty text.
- Errored or aborted assistant messages are skipped and not replayed.
- Tool calls without a result: an error result "No result provided" is inserted before the next assistant or user message. System messages between a tool call and its result are moved after the result.

`split_error_tool_result_images`: Claude on Anthropic and Bedrock rejects images in error tool results, so the images are split out and placed after this turn's tool results.

## Token estimation (`estimate`)

3.5 characters per token; an image counts as 4800 characters; characters are counted as UTF-16 code units (as in upstream JS). `estimate_context_tokens` prefers the usage reported by the last usable assistant message, plus an estimate for the messages after it. If an older prefix message was inserted after it (for example a compaction summary), that usage no longer applies.

## Simple options (`simple_options`)

| Function | Purpose |
| --- | --- |
| `build_base_options` | Common options: merges sampling parameters by thinking level (model default → level override → request parameters); `max_tokens` is capped to the context window (4096 reserved) |
| `clamp_reasoning` | Lowers `xhigh` and `max` to `high` for APIs with budget-based thinking |
| `thinking_budget_for_level` | Default budgets: minimal 1024, low 2048, medium 8192, high 16384; can be overridden per level |
| `adjust_max_tokens_for_thinking` | When the thinking budget and the answer share one limit, leaves at least 1024 tokens for the answer |
