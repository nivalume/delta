# delta-mini: mini-swe-agent on delta-harness

Follows the functionality of upstream
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 2.4.6:

- the default and interactive agents;
- tool-call and text actions;
- the local and Docker environments;
- the trajectory inspector;
- the SWE-bench runners;
- global setup.

The config, the credentials and the trajectory format are delta's own and do not read
upstream's files.

## Problem

Upstream is a hundred-line agent loop:

1. Query the model.
2. Run the actions it states.
3. Render each output with `observation_template`.
4. Repeat until an output's first line is `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.

Here the same loop runs on a delta-harness conversation, so model turns, tool rounds and
inputs are committed to the harness's storage. The models are reached through delta-ai,
named by delta-ai's provider and model ids, with delta-ai's credentials. The trajectory
holds delta-ai messages.

## Mapping

One agent run opens one `Harness` with one root conversation. The storage is SQLite beside
the trajectory (`x.traj.json` keeps `x.harness.db`), or memory without an output path.

- `instructions` is the rendered `system_template`; the task's `instance_template` is the
  first input. No context provider is installed, so the request holds exactly the
  trajectory's messages.
- `model.thinking` sets the conversation's thinking level.
- Tools run sequentially, compaction is off, and `max_continuations` is unbounded:
  `step_limit` is the limit.

| Upstream | Here |
| --- | --- |
| `model.query` | One generation. The `after_response` hook counts the call, adds its cost and parses its actions into the round. |
| Tool-call actions | The `bash` tool. The round's first call runs every action (confirmation, execution until one submits, observations), so each call knows whether the round ends the run and asks to `terminate` together. |
| Text actions, an answer without tool calls | `on_yield`: one `action_regex` match runs and its observation is the continuation; anything else is a format error whose rendered template is the continuation. |
| Exit statuses | `trajectory.exit`; the run ends through `terminate`, `on_yield` returning nothing, or (when no tool of ours can end it) an abort by the driver. |
| Limits | Checked where the next model call would start: at the end of a round, before a continuation, before a run. A batch's limits are checked at the same places. |
| Rejection, new task, `/u`, Ctrl+C | The run ends; the message is queued and becomes the next run's input. |
| Human mode | The driver runs the user's commands between runs; their messages queue, and go in as passive `delta.user` writes and the next input when the mode leaves human. |

## Config

YAML, merged in order:

1. the global config (`config.yaml` in the home directory);
2. the `-c` files;
3. `key.path=value` pairs;
4. the command line.

Each section deserializes with unknown fields denied.

- `agent`: templates, `step_limit`, `cost_limit`, `time_limit`, `format_error_limit`, `mode`,
  `whitelist`, `confirm_exit`.
- `model`:
  - `provider` and `id`;
  - `thinking`, `temperature`, `max_tokens`, `sampling`;
  - `base_url` and `headers`, which override the resolved model;
  - `actions` (`tool_calls` | `text`) and `action_regex`;
  - the templates;
  - `script` for provider `faux`.
- `environment`: `type` (`local` | `docker`) and that environment's fields;
  `startup_command` for SWE-bench.
- `providers`: id → `{name, api, base_url, api_key_env, headers, models}`.
  - Each becomes a delta-ai `create_provider` with `ProviderApi::ByApi` over the APIs its
    models use (`openai-completions`, `openai-responses`, `anthropic-messages`,
    `google-generative-ai`, `mistral-conversations`).
  - Its models are delta-ai `Model` JSON whose omitted fields default: the provider's API and
    base URL, text input, no price, 128k context, 16k output.
  - `api_key_env` gives an `env_api_key_auth`; without it the provider is keyless.
  - A provider replaces a built-in one of the same id.

`-m provider/id` splits at the first `/`. A bare id is looked up across providers; more
than one match is an error. An unknown provider is retried as a bare id `provider/id`. A
known provider's catalog without the id gets its first model's API and base URL, no price
and a warning.

## Credentials

`FileCredentialStore` implements delta-ai's `CredentialStore` on `auth.json` in the home
directory, in the shape of pi's file.

- `modify` and `delete` run under a process mutex and an exclusive lock on `auth.json.lock`
  (`File::lock`).
- Writes go to a temporary file renamed over the old one, mode 0600 on Unix.
- Entries this crate cannot parse are kept.

`mini-extra login` runs `Models::login` with `TerminalInteraction`, which:

- prompts on stdin, reading secrets without echo on a terminal;
- races manual codes against the provider's callback server;
- prints URLs and device codes.

Resolution is delta-ai's: a stored credential, then the provider's environment variables.

## Trajectory

`format: delta-mini/1`, saved after every step (a temporary file, then a rename). The
top-level fields:

- `version` (this crate's);
- `exit` (`null` while running);
- `stats` (`calls`, `cost`, `elapsed_seconds`);
- `model` (the resolved delta-ai model);
- `config` (`agent`, `model`, `environment`, `providers` as run);
- `extra` (the SWE-bench instance id);
- `entries`.

An entry is `{kind, message, ...}`, where `message` is a delta-ai `Message` (pi's wire
shape):

| Kind | Message | Also |
| --- | --- | --- |
| `system` | system | |
| `task` | user | |
| `answer` | the `AssistantMessage` | `actions`, `cost` |
| `observation` | tool result (tool calls) or user (text actions, human mode) | `output` |
| `format_error` | tool result (a bash call) or user (no tool call) | `error` |
| `interrupt` | user | `reason`: `user_rejection`, `user_interruption`, `user_new_task` |
| `human` | user | `command` |

Entries are recorded when their message is delivered, so they hold what the model
received. A call the harness refuses itself (an unknown tool, arguments that fail the
schema) has no entry; its answer has no actions.

Exit statuses: `submitted` (with `submission`), `limits_exceeded`, `time_exceeded`,
`format_errors`, `interrupted`, `eof`, and `error` (with `error`: a failed model call, a
template error, the environment).

The inspector groups entries into steps: an `answer` or a `human` entry starts one. It shows
each entry under its kind, with thinking, cost, tokens and return codes, and the exit after
the last step.

## Pieces

- `config`: spec resolution (a file, else a built-in config), recursive merge,
  `key.path=value`.
- `home`: the home directory and the global config.
- `auth`: the credential store and the terminal login.
- `template`: minijinja with strict undefined.
- `model`: the `model` and `providers` sections, model resolution, `MiniModelClient`
  (streams the resolved model with `temperature`, `max_tokens` and `sampling`), the faux
  script, batch stats.
- `env`:
  - `LocalEnvironment` on the harness's `LocalEnv`, with stdout and stderr merged in
    arrival order;
  - `DockerEnvironment` with `docker run`/`exec`/`stop` command lines.
- `agent`: the driver, the `bash` tool, the hooks and the `Frontend` the interactive agent
  asks through (`Terminal`; `Scripted` for tests).
- `trajectory`, `inspector`, `swebench`, `cli`.

## Differences from upstream

- No compatibility with upstream's config keys (`model_name`, `model_class`,
  `model_kwargs`, `environment_class`, class paths), its `.env` global config and `MSWEA_*`
  variables, or its `.traj.json` format.
- Models are delta-ai's, not litellm's.
  - Prices come from delta-ai's catalog or a provider's model definitions.
  - A model without a price is not an error; `mini` warns that the cost limit cannot apply.
- `mini-extra login`, `logout` and `models` are new; `config setup` stores a model and logs
  in instead of writing `.env` keys. `config set`/`unset` are gone: edit the YAML.
- `agent_class` is gone: `mini` and `swebench-single` run the interactive agent, the batch
  the default one.
- The harness commits every answer, so after a format error the model sees its own
  malformed reply before the error message; upstream drops the reply.
- Format error templates get delta-ai's `stop_reason` instead of litellm's
  `finish_reason`.
- The text configs (`default`, `mini_textbased`, `swebench_backticks`, `swebench_xml`) set
  `model.actions: text` themselves; upstream needs a text model class.
- Commands run with bash (`LocalEnv`), not `sh`.
- Only the local and Docker environments; Singularity, bubblewrap, SWE-ReX, Modal and
  Contree are not ported. ProgramBench is not ported.
- Batch-wide limits are `--total-cost-limit` and `--total-call-limit`, checked before each
  call.
- `/m` reads lines until an empty one instead of prompt_toolkit's Esc-then-Enter.
- Datasets come from the Hugging Face datasets server's `/rows` endpoint (or a local JSON or
  JSON Lines file) instead of the `datasets` library. `--shuffle` reimplements Python's
  `random.seed(42)` and `shuffle`, so the order matches upstream's.
- The inspector and the batch progress are drawn with ratatui and crossterm instead of
  textual and rich; the command palette is not ported.
- YAML is read with serde-saphyr. It keeps a line break that a block scalar ending the file
  without one does not have; the loader drops it.
