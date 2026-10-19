# 代理流（`proxy`）

对应上游 `src/proxy.ts`。应用把模型请求转发到自己的服务器，由服务器持有 provider 凭据；`stream_proxy` 是这种部署方式的 `StreamFn` 实现。

## 协议

请求：`POST {proxy_url}/api/stream`，`Authorization: Bearer {auth_token}`，JSON 体：

```json
{ "model": { ... }, "context": { "messages": [ ... ] }, "options": { ... } }
```

`options` 是 `SimpleStreamOptions` 中可序列化的部分，字段名与上游相同：`temperature`、`samplingParams`、`maxTokens`、`reasoning`、`cacheRetention`、`sessionId`、`headers`、`metadata`、`transport`、`thinkingBudgets`、`maxRetryDelayMs`。值为空的字段省略。

响应：SSE，每个 `data:` 是一个 `ProxyAssistantMessageEvent`。它和 `AssistantMessageEvent` 相同，只是去掉了 `partial`（服务器不重复发送整条消息），`done`/`error` 只带停止原因、用量和错误文本：

| 事件 | 字段 |
| --- | --- |
| `start` | |
| `text_start` / `thinking_start` | `contentIndex` |
| `text_delta` / `thinking_delta` | `contentIndex`、`delta` |
| `text_end` / `thinking_end` | `contentIndex`、`contentSignature?` |
| `toolcall_start` | `contentIndex`、`id`、`toolName` |
| `toolcall_delta` | `contentIndex`、`delta` |
| `toolcall_end` | `contentIndex`、`toolCall` |
| `done` | `reason`、`usage`、`providerThinkingLevel?` |
| `error` | `reason`、`errorMessage?`、`usage`、`providerThinkingLevel?` |

## 客户端重建

`stream_proxy` 用 `delta_ai::events::MessageBuilder` 重建消息，和手写的 provider 实现走同一套生产端工具，所以发给循环的事件、`partial()` 归约结果与直连 provider 一致：

- `*_start` 的 `contentIndex` 必须等于当前内容块数量，`*_delta`/`*_end` 必须指向对应类型的块，否则以错误结束。
- `toolcall_delta` 的参数用 `delta_ai::json` 的流式 JSON 解析；`toolcall_end` 用服务器给的最终 `toolCall` 替换（`MessageBuilder::end_tool_call_with`）。
- `done`/`error` 写入停止原因、用量和错误文本后结束。

## 错误

| 情况 | 结果 |
| --- | --- |
| HTTP 非 2xx | `Proxy error: {status} {reason}`；响应体是带 `error` 字段的 JSON 时改为 `Proxy error: {error}` |
| 连接或解析失败 | 错误文本为底层错误 |
| 流在 `done`/`error` 之前正常关闭 | `Connection closed by proxy server before the response completed` |
| 取消 | 停止原因 `aborted`，文本 `Request aborted by user` |

## 与上游的差异

- 上游从 `fetch` 的 `ReadableStream` 手动按行切分 SSE；delta 用 `delta_ai::http::sse_events`，与其他手写 provider 共用。
- 上游在客户端维护 partial 对象并原地修改；delta 用 `MessageBuilder`，事件流的消费端归约出 partial（`delta_ai` 的设计）。
- 事件流被 drop 时停止读取并断开连接（`delta_ai` 的约定）。
