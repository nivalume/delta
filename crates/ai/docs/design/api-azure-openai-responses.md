# API：azure-openai-responses

## 目的

实现 Azure OpenAI 的 v1 Responses 接口：`POST {baseUrl}/responses?api-version=<版本>`，SSE 流式，用 `api-key` 请求头认证。输入项转换、工具转换和流事件处理全部复用 `api::openai_responses_shared`（见 `api-openai-responses.md`），本模块只负责 Azure 的端点、部署名、请求体差异和错误前缀。单元结构体为 `AzureOpenAIResponsesApi`。

## 上游源文件

- `packages/ai/src/api/azure-openai-responses.ts`：`stream`、`streamSimple`、`buildParams`。
- `packages/ai/src/api/azure-openai-config.ts`：部署名映射、端点解析与规范化、API 版本。
- 测试参照：`test/azure-openai-base-url.test.ts`、`test/azure-openai-responses-reasoning-replay.test.ts`、`test/azure-openai-tool-choice.test.ts`，以及 pi-py 的 `tests/test_azure_openai_responses.py`。

## 端点与配置

Azure 模型没有可直接使用的 baseUrl：每个用户一个资源，按请求解析。选项来自 `ApiOptions::AzureOpenAIResponses(AzureOpenAIResponsesOptions)`，环境变量先查 `options.env` 再查进程环境（空值视为未设置）。

- 基础 URL，按顺序取第一个非空值：
  1. `azure_base_url`（去首尾空白）；
  2. `AZURE_OPENAI_BASE_URL`（去首尾空白）；
  3. `azure_resource_name` 或 `AZURE_OPENAI_RESOURCE_NAME` → `https://<name>.openai.azure.com/openai/v1`；
  4. `model.base_url`；
  5. 都没有时报错 `Azure OpenAI base URL is required. ...`。
- 规范化：去掉结尾 `/` 后解析 URL，失败报错 `Invalid Azure OpenAI base URL: <原值>`。主机以 `.openai.azure.com`、`.cognitiveservices.azure.com` 或 `.ai.azure.com` 结尾，且路径为空、`/`、`/openai` 或 `/openai/v1/responses` 时，路径改为 `/openai/v1` 并去掉查询串。其他 URL（包括代理）保持不变。
- API 版本：`azure_api_version`，否则 `AZURE_OPENAI_API_VERSION`，否则 `v1`。
- 请求 URL：基础 URL 的路径加 `/responses`，查询串替换为 `api-version=<版本>`。
- 部署名（作为请求体的 `model`）：`azure_deployment_name`，否则 `AZURE_OPENAI_DEPLOYMENT_NAME_MAP` 中该模型的条目，否则 `model.id`。映射格式为逗号分隔的 `模型=部署`，空项和缺少任一边的项跳过；同 JavaScript `split("=", 2)`，第二个 `=` 之后的内容丢弃；后出现的同名项覆盖先出现的。

## 请求映射

- 请求头（后者覆盖前者）：`api-key: <api_key>`、`Accept`、`Content-Type: application/json`、`User-Agent: tau/...`，然后 `model.headers`，最后 `options.headers`。不发送 `Authorization`，也不发送会话亲和头。
- 必须有 `options.api_key`，请求头不能代替（与上游一致），否则报错 `No API key for provider: <provider>`。
- 工具调用 id 允许列表：`openai`、`openai-codex`、`opencode`、`azure`。

请求体字段（按插入顺序）：

| 字段 | 规则 |
| --- | --- |
| `model` | 部署名 |
| `input` | 共享转换；`supportsMidConvoSystemMessages`、`supportsAdditionalTools`、`supportsToolSearch` 默认 false |
| `stream` | `true` |
| `prompt_cache_key` | `session_id` 截断到 64 个码点（不看缓存保留设置） |
| `store` | `false` |
| `max_output_tokens` | `max_tokens > 0` 时发送，最小 16（不看 `supportsMaxOutputTokens`） |
| `temperature` | 设置时发送 |
| `tools` | 请求级工具；`supportsStrictMode` 默认 **true**（OpenAI 为 false），`supportsOpenAIGrammarTools` 默认 false |
| `tool_choice` | 原样发送 |
| `reasoning` / `include` | 与 `openai-responses` 相同，但没有 GitHub Copilot 与 xAI 分支，也没有中途 effort |
| 采样参数 | 最后合并 |

上游 Azure 不发送 `service_tier`、`prompt_cache_retention`、`prompt_cache_options`，不做服务等级计价，不使用 `configuration_update`。`OpenAIResponsesOptions::service_tier` 在 Azure 下被忽略。

## 流事件映射

与 `openai-responses` 完全相同（共享 `process_responses_stream`），没有服务等级计价。Azure 可能只在 `response.completed` 的 `output` 里给出推理项的 `encrypted_content`，共享层会把它回填到已保存的推理签名里，保证 `store: false` 下的多轮回放。

输出消息的 `api` 固定为 `azure-openai-responses`（与上游一致）。

## 错误与重试

- 重试与取消同 `openai-responses`（`http::send_with_retry`）。
- HTTP 错误格式为 `Azure OpenAI API error (<status>): <消息>`；端点解析、缺少 key、流内错误保留原消息。
- 流结束时停止原因仍为 pending 报错 `Azure OpenAI Responses stream ended without a stop reason`（实际上终止事件总会设置停止原因）。

## stream_simple

与上游一致，只传递基础选项、`tool_choice`（`auto` / `none`）和按模型支持范围钳制后的推理等级（`off` 不发送 effort）。`build_base_options` 会清空 `options.api`，所以简单接口下的 Azure 端点只能来自环境变量或 `model.base_url`。

## Rust 特有差异

1. 请求 URL 用 `url` crate 拼接：路径加 `/responses`，查询串替换为 `api-version`。上游 SDK 把 `/responses` 直接接在整个基础 URL 字符串后面，基础 URL 带查询串时 `/responses` 会落进查询串、随后被 `api-version` 替换掉；这里沿用 pi-py 的行为（`/v1/responses?api-version=...`）。只影响带查询串的代理 URL。
2. 缺少 API key 时上游 `streamSimple` 同步抛错，这里返回以同样消息结束的错误流。
3. 其余差异（事件不带 partial、工具调用结束参数、空增量等）见 `api-openai-responses.md`。

## 未移植

- 上游 `azure` provider 对 `openai-completions` 的包装（`resolveAzureModel` / `withDeploymentName`）属于 provider 层，不在本 API 模块内。
- 遥测、`fetch` 选项（由 `StreamOptions::http_client` 替代）。

## 待确认的改进

- 简单接口丢弃显式 Azure 配置（上游行为）。可以在 `stream_simple` 中保留调用方传入的 `azure_*` 字段。
- Azure 忽略 `service_tier`、缓存保留和会话亲和头（上游行为）。Azure v1 接口是否接受这些字段待确认。
- 带查询串的基础 URL 目前丢弃原查询串，可以改为保留原查询参数再追加 `api-version`。
