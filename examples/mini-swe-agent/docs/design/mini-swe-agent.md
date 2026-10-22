# mini-swe-agent on delta-harness

Corresponds to upstream [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
2.4.6 (`src/minisweagent`): `agents/{default,interactive}.py`, `models/` (litellm,
text-based, OpenRouter and deterministic classes), `environments/{local,docker}.py`,
`config/`, `run/mini.py`, `run/utilities/{config,inspector,mini_extra}.py` and
`run/benchmarks/{swebench,swebench_single}.py`.

## Problem

Upstream is a hundred-line agent loop: query the model, run the actions it states, render
each output with `observation_template`, repeat until an output's first line is
`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. The port keeps its configs, its trajectory files
and its command lines, and runs the loop on a delta-harness conversation, so model turns,
tool rounds and inputs are committed to the harness's storage while the trajectory is
written in upstream's format beside it.

## Mapping

One agent run opens one `Harness` (SQLite beside the trajectory: `x.traj.json` keeps
`x.harness.db`; memory without an output path) with one root conversation:

- `instructions` is the rendered `system_template`; the task's `instance_template` is the
  first input. No context provider is installed, so the request holds exactly upstream's
  messages.
- Tools run sequentially, compaction is off (upstream has none), and `max_continuations`
  is unbounded: `step_limit` is the limit.

| Upstream | Here |
| --- | --- |
| `model.query` | One generation. The `after_response` hook counts the call, adds its cost and parses its actions into the round. |
| Tool-call actions | The `bash` tool with upstream's `BASH_TOOL` schema. The round's first call runs every action (confirmation, execution until one submits, observations), so each call knows whether the round ends the run and asks to `terminate` together. |
| Text-based actions, an answer without tool calls | `on_yield`: one `action_regex` match runs and its observation is the continuation; anything else is a format error whose rendered template is the continuation. |
| `Submitted`, `LimitsExceeded`, `TimeExceeded`, `RepeatedFormatError` | The trajectory's exit message; the run ends through `terminate`, `on_yield` returning nothing, or (when no tool of ours can end it) an abort by the driver. |
| Limits | Checked where the next model call would start: at the end of a round, before a continuation, before a run. |
| Rejection, new task, `/u`, Ctrl+C | The run ends; the message is queued and becomes the next run's input. |
| Human mode | The driver runs the user's commands between runs; their messages queue, and go in as passive `delta.user` writes and the next input when the mode leaves human. |

The trajectory is built message by message in upstream's OpenAI-style shape and saved after
every step (a temporary file, then a rename).

## Pieces

- `config`: specs resolve as upstream's `get_config_path` (path, `MSWEA_CONFIG_DIR`, the
  built-in configs, also under `extra/` and `benchmarks/`); configs merge recursively;
  `key.path=value` sets one value (JSON when it parses).
- `template`: minijinja with strict undefined and jinja2's `tojson` (sorted keys, ASCII,
  `<>&'` escaped). Rendered observations and format errors of every built-in config are
  byte-identical to jinja2's.
- `model`: the class from `model_class`; the name from `-m`, the config or
  `MSWEA_MODEL_NAME`, resolved to a delta-ai model (litellm prefixes such as `gemini/` map to
  delta-ai's provider ids; OpenRouter classes use the whole name as the OpenRouter id).
  `MiniModelClient` streams it with `temperature`, `max_tokens` and `reasoning_effort` from
  `model_kwargs`; OpenAI-style sampling keys pass through.
- `env`: `LocalEnvironment` on the harness's `LocalEnv` (stdout and stderr merged in arrival
  order) and `DockerEnvironment` with upstream's `docker run`/`exec`/`stop` command lines.
- `agent`: the driver, the `bash` tool, the hooks and the `Frontend` the interactive agent
  asks through (`Terminal`; `Scripted` for tests).
- `trajectory`, `inspector`, `swebench`, `global` and `cli` as their upstream modules.

## Differences from upstream

- Models are delta-ai's, not litellm's. Prices come from delta-ai's catalog; a model it does
  not price fails cost tracking as upstream does, unless `cost_tracking: ignore_errors`. A
  model id a known provider's catalog lacks is sent with that provider's API.
- `extra.response` of an assistant message is delta-ai's `AssistantMessage`, not litellm's
  `ModelResponse`.
- The harness commits every answer, so after a format error the model sees its own
  malformed reply before the error message; upstream drops the reply. An unknown tool or a
  call without `command` is refused by the harness itself ("Tool X is not available",
  "Invalid arguments"), which is what the model sees for that call; the trajectory records
  upstream's format error.
- Commands run with bash (`LocalEnv`), not `sh`.
- `multimodal_regex` and `set_cache_control` are accepted and not acted on: delta-ai places
  cache markers itself.
- Only the local and Docker environments; Singularity, bubblewrap, SWE-ReX, Modal and
  Contree are not ported. ProgramBench and the Response-API classes are not ported.
- The global `.env` is not loaded into the process environment (that needs `set_var`, which
  the workspace's `unsafe_code = "forbid"` rules out): `global::env_var` and the credential
  lookups of delta-ai read the process environment first and the file second, which is what
  `load_dotenv` without `override` gives.
- The global config directory is `dirs::config_dir()/mini-swe-agent` (platformdirs puts it
  under `%LOCALAPPDATA%\mini-swe-agent\mini-swe-agent` on Windows).
- `/m` reads lines until an empty one instead of prompt_toolkit's Esc-then-Enter.
- Datasets come from the Hugging Face datasets server's `/rows` endpoint (or a local JSON or
  JSON Lines file) instead of the `datasets` library. `--shuffle` reimplements Python's
  `random.seed(42)` and `shuffle`, so the order is upstream's.
- The inspector and the batch progress are drawn with ratatui and crossterm instead of
  textual and rich; the command palette is not ported.
- YAML is read with serde-saphyr (safe Rust). It keeps a line break that a block scalar
  ending the file without one does not have; the loader drops it, so every built-in config
  reads as PyYAML reads it.
- `info.mini_version` is `2.4.6+delta.<version>`; `agent_type`, `model_type` and
  `environment_type` keep upstream's class paths so tools that read them still work.
