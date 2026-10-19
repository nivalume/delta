# API: openai-decisions

## Purpose

`openai-decisions` calls OpenAI's [Decisions API](https://developers.openai.com/api/docs/guides/decisions): `POST <baseUrl>/decisions` with `{ model, input, questions }`. GPT-6 Luna (`gpt-6-luna` on the `openai` provider) is its only catalog model. The request goes through the shared classifier plumbing described in [api-system-one.md](api-system-one.md#shared-classifier-plumbing-classifier_shared).

Code: `crates/ai/src/api/openai_decisions.rs` with `OpenAIDecisionsApi` (implements `ProviderClassifier`), `classify(model, context, options)`, `API` and `MAX_IMAGES`.

## Upstream source files

- `packages/ai/src/api/openai-decisions.ts` (+ `.lazy.ts`).
- The `classifiers` map and `filterAllModels` of `packages/ai/src/providers/openai.ts` → `providers::builtin::with_operations` and `openai_filter_all_models`.
- Tests: `test/openai-decisions.test.ts` → `tests/openai_decisions.rs`; the GPT-6 Luna cases of `test/classifier-models.test.ts` → `tests/classifier_models.rs`.

## Request mapping

1. `model.api` must be `openai-decisions`, else `Unsupported classifier API: <api>`.
2. `input`: the state as compact JSON text. With images, one user message instead: `[{ role: "user", content: [{ type: "input_text", text: <state JSON> }, { type: "input_image", image_url: "data:<mime>;base64,<data>" }, ...] }]`. More than 128 images fail before any request: `OpenAI Decisions accepts at most 128 images, got <n>`.
3. `questions`: an array in question order, each with `name` = the question id:

| Question | Wire question |
| --- | --- |
| `choice` | `{ type: "choice", name, instructions, choices: [{ value, description? }] }`; an empty description is omitted |
| `score` | `{ type: "score", name, instructions, levels: [{ label }] }` |
| `bool` | `{ type: "predicate", name, instructions }`; the instructions get `\n\nTrue means: <true>\nFalse means: <false>` (empty meanings left out) |

4. `post_classifier_request` with label `OpenAI Decisions` and no-retry status 504.

`options.temperature` is ignored. The endpoint needs an API key: `No API key for provider: openai` without one.

## Response mapping

The body must be an object, else `OpenAI Decisions returned an unexpected response`. Usage (`input_tokens`, `output_tokens`) is parsed and priced before the answers, so refused or malformed answers keep the billed usage. The catalog price of `gpt-6-luna` has a tier above 272K input tokens; `calculate_cost` applies it to the whole request.

`answers` must be an array; answers are matched to questions by `name` (the last answer with a name wins), so their order does not matter.

| Question | Answer | Result |
| --- | --- | --- |
| any | `{ type: "refusal" }` | error `OpenAI Decisions refused to answer <id>` |
| `choice` | `{ type: "choice", choice, probabilities: [{ value, probability }], confidence }` | `Choice` with `probabilities` keyed by `value` |
| `score` | `{ type: "score", score, confidence }` (its `probabilities` are ignored) | `Score` |
| `bool` | `{ type: "predicate", probability }` | `Bool` |

Missing answers fail with `did not return an answer for <id>`, wrong types with `did not return a <choice|score|predicate> answer for <id>`, bad probability arrays with `returned invalid probabilities for <id>`, and missing numbers with `returned an invalid <field>`. Every failure clears the answers.

## Errors and retries

Cloudflare in front of api.openai.com answers 504 with an HTML page when a request runs longer than about five seconds (inputs above roughly 600K tokens). Retrying the same input hits the limit again, so 504 is not retried, and its message replaces the HTML: `OpenAI Decisions error (504): the request timed out at the gateway. Very large inputs (above roughly 600K tokens) currently exceed its time limit.` Other HTTP errors are retried with the default policy and formatted as `OpenAI Decisions error (<status>): <body>`.

## Availability with ChatGPT OAuth

Sign in with ChatGPT tokens only reach the Responses API; the Decisions route rejects them. The `openai` provider therefore sets `filter_all_models`: with an OAuth credential it drops every classifier model, with an API key (or no stored credential) it keeps all models. `get_available_of_type(Classifier, "openai")` is then empty for OAuth users, while the chat entry `gpt-6-luna` stays available.

The rule keys on the stored credential type only. It is in place before the ChatGPT OAuth method is attached to the `openai` provider; until then an OAuth credential does not authenticate the provider at all. `tests/classifier_models.rs` checks the full `Models` path with an OAuth method attached to a wrapper of the built-in provider.

`filter_all_models` replaces `filter_models` in `Models::get_all_available`. If the `openai` provider later gets a chat filter (`filter_models`), `openai_filter_all_models` has to apply it too. Upstream's `openai.ts` has only `filterAllModels`.

## Rust-specific differences

The differences of the shared plumbing apply (see [api-system-one.md](api-system-one.md#rust-specific-differences)). In addition:

- The state JSON is written by `serde_json`. It matches `JSON.stringify` except for floats with an integral value, which serde_json keeps as `1.0` where JavaScript writes `1`.

## Improvements pending confirmation

None.
