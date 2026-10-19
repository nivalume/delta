# HTTP、SSE 与 API 任务框架（`http`、`json`、`api::common`）

上游没有单独的 HTTP 层：各 API 用厂商 SDK，SDK 负责重试、超时和 SSE。手写客户端需要自己实现这些，所以集中在 `http` 模块里，行为对齐上游所用 SDK 的策略（`src/utils/provider-retry.ts`、`src/utils/error-body.ts`、`src/utils/headers.ts`），并参考了 Zed 和 litellm-rs 的实现（见 `docs/research/provider-references.md`）。

## 客户端与请求头

- `http::client(&options)`：`options.http_client` 或当前 Tokio 运行时的默认客户端（rustls，连接超时 30 秒，User-Agent `tau/<版本> (<os>; <arch>)`）。代理遵循 `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` 和系统代理设置。
  默认客户端按运行时缓存（只保留最近一个运行时的）：连接池里的连接属于打开它的运行时，运行时关闭后这些连接失效，跨运行时复用会让请求随机失败（例如每个 `#[tokio::test]` 各有一个运行时）。不在运行时里调用时每次新建。
- `http::build_headers([..])`：按顺序合并多个来源，名字不区分大小写，后者覆盖前者，`None` 删除前面的同名头。API 实现的顺序：API 默认头 → `options.headers`。

## 重试（`send_with_retry`）

与上游 `retryProviderRequest` 相同（上游复刻了 OpenAI/Anthropic SDK 的策略并让退避可中断）：

- 可重试：`x-should-retry: true`；或 408、409、429、5xx；或连接错误。`x-should-retry: false` 不重试。
- 次数：`max_retries`，默认 2。
- 延迟：`retry-after-ms`，其次 `retry-after`（秒数或 HTTP 日期），否则指数退避 `min(0.5·2^n, 8)` 秒并减去至多 25% 的抖动。
- 服务器要求的延迟超过 `max_retry_delay`（默认 60 秒，0 表示不限）时立即失败，错误信息包含要求的秒数。
- 退避期间和请求过程中都响应 `options.cancel`。
- 请求超时：`timeout`，默认 10 分钟（与 SDK 默认一致）。

错误响应变成 `Error::Status { status, message: "<status> <正文消息>", body, retry_after_ms }`。正文消息取 JSON 的 `error.message` / `message` / `error` / `detail`，否则取整段文本，截断到 4000 字符（上游 `MAX_PROVIDER_ERROR_BODY_CHARS`）。`body` 是原始正文，按上游 `normalizeProviderError` 的做法去掉首尾空白（JavaScript `trim` 的空白集合）、空正文为 `None`、截断到 4000 字符并注明截掉的字符数（`http::error_body_text`）。

上游的 HTTP 错误文本来自厂商 SDK 的错误对象，各 API 用 `body` 按上游格式重写 `message`：

- `openai-completions`、`openai-responses`、`azure-openai-responses`：OpenAI SDK 的 `APIError` 消息加 `formatProviderError`（`api::common::openai_sdk_error`、`api::common::format_provider_error`）。
- `anthropic-messages`：Anthropic SDK 的 `APIError` 消息，`"<status> <整个 JSON body>"`（`api::common::sdk_api_error_message`）。
- `google-generative-ai`：`@google/genai` 的 `ApiError` 消息，即整段 JSON 正文（`google_shared::format_google_http_error`）。
- `mistral-conversations` 不经过 `send_with_retry`，自己按上游格式 `Mistral API error (<status>): <正文>` 生成消息，也填写 `body`。

截断按字符计数，上游按 UTF-16 码元计数；超过 4000 字符的 JSON 正文截断后无法解析，按文本处理。

## SSE 解码（`SseDecoder`、`sse_events`）

按 WHATWG event-stream 规范实现：

- 行结束符 `\n`、`\r\n`、单独的 `\r`（包括跨 chunk 的 `\r` + `\n`）。
- 字节按行缓冲后再解码 UTF-8，跨 chunk 截断的多字节字符正确解码。
- 多行 `data:` 用 `\n` 拼接；字段值只去掉一个前导空格；`:` 开头的注释忽略；暴露 `event` 和 `id`。
- 空行分派事件；流结束时冲刷最后一个未以空行结尾的事件。
- `[DONE]` 不在解码器里特殊处理，由各 API 决定（OpenAI 兼容流用它表示结束；Anthropic、Gemini 跳过代理注入的 `[DONE]`）。

`sse_events(response, cancel)` 把响应体变成 `Stream<Item = Result<SseEvent>>`，读取错误和取消以 `Err` 返回。

## 流式 JSON（`json`）

上游用 `partial-json` 包解析未完成的工具参数。Rust 里手写（没有合适的高质量 crate）：

- `repair_json`：转义字符串里的原始控制字符，非法转义前的反斜杠加倍。
- `parse_json_with_repair`：先严格解析，失败后修复再解析一次。
- `parse_partial_json`：递归下降解析 JSON 前缀，在输入结束处闭合未完成的字符串、数组、对象；不完整的数字回退到最长合法前缀；`tr` 之类的字面量前缀补全；对象里没有值的键丢弃；非法（而非截断）输入返回 `None`。
- `parse_streaming_json`：依次尝试完整解析、前缀解析、修复后的前缀解析，结果不是对象时返回空对象。
- `parse_tool_call_arguments`：完成的工具调用参数，非对象时返回空参数和包含原文的错误信息。

## API 任务框架（`api::common`）

`spawn_stream(model, cancel, run)` 在 Tokio 任务里运行一次 API 请求，并统一结束事件流：

- `run` 返回 `Err`，或取消令牌已触发：发 `Error`（取消时为 `aborted`，消息 "Request was aborted"），保留已收到的部分内容。
- `run` 返回 `Ok`，但 `stop_reason` 是 error/aborted：发 `Error`，消息取 `error_message`。
- 否则补齐未结束的工具调用后发 `Done`。
- 消费者已丢弃流：什么都不发。

另有回调包装（`apply_on_payload`、`notify_response`、`observe_provider_event`）、`finalize_usage`（重算 total 和费用）、`require_auth`、`join_url`、`short_hash`（上游 `utils/hash.ts` 的 `shortHash`，按 UTF-16 码元计算，输出与 TypeScript 逐位一致）。

各 API 共用的约束采样逻辑（上游 `api/constrained-sampling.ts`：严格 JSON schema 转换、grammar 工具）在 `api::constrained_sampling`。`make_strict_json_schema` 和 `resolve_json_schema_strict_sampling` 与上游一样接受可选的关键字检查（`Option<fn(&str, &Value) -> bool>`），provider 用它拒绝自己的严格模式不支持的关键字，错误文本为 `<key>: <值的 JSON> is unsupported`；目前只有 Anthropic 传入检查。
