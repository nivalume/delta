# API: llama-cpp-classify

## Purpose

`llama-cpp-classify` turns a chat model served by llama.cpp's `llama-server` into a classifier. The model never generates an answer. Each question becomes one chat prompt that lists the possible answers under single-token labels; the server evaluates the prompt and returns the log-probabilities of the most likely next tokens; the answer is the softmax over the label tokens among them.

No built-in provider registers this API. Upstream registers it in the coding agent's llama extension; library users register it on their own provider (README "Chat models on llama.cpp"). Decision models that llama.cpp serves natively through `/v1/systemone` use `typesafe-system-one` instead.

Code: `crates/ai/src/api/llama_cpp_classify.rs` with `LlamaCppClassifyApi` (implements `ProviderClassifier`), `classify(model, context, options)`, `API`, and the helpers upstream exports for tests: `render_question`, `LabeledQuestion`, `label_probabilities`, `peak_confidence`, `answer_from_probabilities`, `llama_server_root`.

## Upstream source files

- `packages/ai/src/api/llama-cpp-classify.ts` (+ `.lazy.ts`).
- Tests: `test/llama-cpp-classify.test.ts` → `tests/llama_cpp_classify.rs` (a wiremock responder plays `llama-server`).

## Server endpoints

All requests are JSON POSTs to the server root: `model.base_url` without trailing slashes and without a trailing `/v1` (pi's llama.cpp models use the OpenAI-compatible `/v1` URL as base URL). Every body carries `model: <id>` for router mode.

| Endpoint | Body | Used for |
| --- | --- | --- |
| `/tokenize` | `{ model, content, add_special: false, parse_special: false }` | label token ids |
| `/apply-template` | `{ model, messages: [system, user], chat_template_kwargs: { enable_thinking: false } }` | the prompt in the model's chat template |
| `/completion` | `{ model, prompt, n_predict: 1, n_probs, post_sampling_probs: false, cache_prompt: true, temperature: 0 }` | next-token log-probabilities |

Headers: `content-type: application/json`, `authorization: Bearer <key>` only when an API key is set (llama-server needs none unless started with `--api-key`), then `model.headers` and the request headers. Each request goes through the shared `post_json` loop (timeout per attempt, retries, cancellation; see [api-system-one.md](api-system-one.md#shared-classifier-plumbing-classifier_shared)). Only the `/completion` requests run `on_payload` and `on_response`.

## Prompt

`render_question(context, id)` builds the user message of one question:

```
State:
<JSON.stringify(state, null, 1)>

Task: answer each of the following questions about the state.   (or "the following question" for one)

<every question with its options, unlabeled>

State:
<state again>

<this question with labeled options>

<answer instruction>
```

| Question | Options | Labels | Keys | Instruction |
| --- | --- | --- | --- | --- |
| `choice` | `Options:` lines `<label>. <key>[: <description>]` (unlabeled: `- ...`) | `A`-`Z`, `a`-`z`, `0`-`9` (2 to 62 options) | criteria keys | `Answer with one letter.` |
| `score` | `Levels:` lines `<index>. <level>` | `0`-`9` (2 to 10 levels) | the labels | `Answer with one level number.` |
| `bool` | `Yes means: ...` / `No means: ...` (empty meanings left out) | `Yes`, `No` | `true`, `false` | `Answer Yes or No.` |

Everything before the final question is the same for all questions of a request, so the server's prompt cache evaluates the state once. The system prompt tells the model to reply with a label only and to treat instructions inside the state as data. A template whose prompt ends with `<think>` gets `</think>` appended, so the next token is the answer.

Unsupported option counts fail before the first request: `A choice question needs 2 to 62 options, got <n>`, `A score question needs 2 to 10 levels, got <n>`.

## Label tokens

The reply follows a newline in the rendered template, so a label is tokenized after one: `"\n"` and `"\n<label>"` are tokenized in parallel, and when the second is the first plus one token, that token is the label. Otherwise the label is tokenized alone and must be one token. Results are cached per server root, model id and label for the life of the process (`None` when the label is several tokens); failed lookups are not cached. Errors: `Label "<label>" is not a single token for <model>`, `Labels share a token for <model>: <labels>`, `llama.cpp returned an unexpected tokenization`.

## Readout

For each question, in order (one at a time, so the prompt cache is reused), the label tokens and the prompt are fetched in parallel, then `/completion` is called with `n_probs` = max(256, 16 × labels). When a label is missing from `completion_probabilities[0].top_logprobs`, the readout is repeated with 4096 and then 32768. Still missing: `llama.cpp did not rank labels <labels> for <id> within the top 32768 tokens`. When every label log-probability is at or below `-1e30` (llama-server's underflow value): `<model> gave no probability to any answer label for <id>`.

`label_probabilities` divides the log-probabilities by `options.temperature` (default 1; it must be positive and finite, else `Temperature must be a positive number, got <t>`) and applies softmax. Answers:

- `bool`: the probability of `Yes`.
- `score`: the expected level, sum of index × probability, with `peak_confidence`.
- `choice`: the most probable key (the first on ties), all probabilities by key, `peak_confidence`.

`peak_confidence` is TypeSafe's documented choice confidence `(n × peak − 1) / (n − 1)`, clamped to [0, 1]. No usage is reported.

Images fail with `llama.cpp classification does not support image input`, other APIs with `Unsupported classifier API: <api>`. Errors use the prefix `llama.cpp error` (for example `llama.cpp error (400): {"error":{"message":"context overflow"}}`) and clear the answers.

## Rust-specific differences

The differences of the shared plumbing apply (see [api-system-one.md](api-system-one.md#rust-specific-differences)). In addition:

- `JSON.stringify(state, null, 1)` is serde_json's pretty printer with a one-space indent. Floats with an integral value print as `1.0` where JavaScript prints `1`.
- Token ids are compared as JSON numbers (by their `f64` bits, with `-0` equal to `0`), as upstream compares JavaScript numbers.
- The label cache stores resolved token ids. Upstream stores the pending promise, so concurrent requests for the same uncached label share one lookup; in Rust both look it up. The result is the same.
- The temperature in the error message is printed with Rust's float formatting (`0`, `-1`, `NaN`, `Infinity`); large or tiny values differ from JavaScript's exponent notation.

## Improvements pending confirmation

None.
