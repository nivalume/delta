# 内置模型目录与 faux provider（`providers::catalog`、`providers::faux`）

## 模型目录

上游在构建时从 models.dev 等来源生成每个 provider 的目录（`src/providers/data/<provider>.json`，不入库），npm 包里带有生成结果。tau 不重复上游的生成脚本，而是直接使用 npm 包里的数据：

- `crates/ai/data/providers/*.json`：从 `@earendil-works/pi-ai` npm 包复制，当前为 1.3.0，42 个 provider、1714 个模型条目（chat 1614、image 60、classifier 40）。
- `scripts/sync-models.sh [版本]`：从指定版本的 npm 包重新同步。
- 文件格式与上游相同：按 API id 分组，键为 `"<类型>:<模型 id>"`。

`catalog` 模块用 `include_str!` 把数据编进二进制，每个 provider 的文件在第一次访问时解析并缓存（`OnceLock`）。解析失败的单个条目记录 `tracing` 警告后跳过。

| 函数 | 上游 |
| --- | --- |
| `builtin_provider_ids()` | `getBuiltinProviders()` |
| `all_builtin_models(provider)` | `getAllBuiltinModels(provider)` |
| `builtin_models(provider)` | `getBuiltinModels(provider)` |
| `builtin_model(provider, id)` | `getBuiltinModel(provider, id)` |
| `builtin_model_of_type(type, provider, id)` | `getBuiltinImageModel` / `getBuiltinClassifierModel` |

上游的静态读取有字面量类型（provider 和模型 id 可以自动补全）；Rust 用字符串，不生成枚举。

测试会检查每个文件里的每个条目都能解析成 `AnyModel`。

## faux provider

上游：`src/providers/faux.ts`。用于测试和演示的内存 provider，按脚本返回响应。

```rust
let faux = FauxProvider::new(FauxOptions::default());
models.set_provider(faux.clone());
faux.set_responses(vec![faux_assistant_message(vec![faux_text("Hello!")]).into()]);
```

与上游相同的行为：

- 响应按请求开始的顺序出队；队列为空时返回错误消息 "No more faux responses queued"。
- `set_responses` 替换队列，`append_responses` 追加；`FauxResponse::Factory` 根据请求上下文生成响应。
- 文本、思考、工具参数按 3–5 个 token（每 token 4 个字符）分块流式发出；设置 `tokens_per_second` 时按实时速度发出。
- usage 按每 4 个字符 1 个 token 估算；设置 `session_id` 且缓存保留不为 `none` 时模拟提示词缓存（与上一次请求的公共前缀计为缓存读取）。
- 取消令牌在分块之间检查，中止时保留部分内容。
- 辅助函数：`faux_text`、`faux_thinking`、`faux_tool_call`、`faux_assistant_message`（有工具调用时停止原因为 `toolUse`）。

与上游的差异：`FauxProvider` 本身实现 `Provider` 并可克隆（克隆共享队列），不再区分 handle 和 provider；deferred 响应相关功能不在第一阶段。分块大小用简单的 xorshift 伪随机数，不引入 `rand` 依赖。
