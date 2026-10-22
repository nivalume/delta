# delta-mini: mini-swe-agent on delta-harness

[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)'s functionality built on
`delta-harness` and `delta-ai`. It includes:

- the `mini` agent, with confirm, yolo and human modes;
- local and Docker environments;
- trajectory files and a trajectory inspector;
- the SWE-bench batch and single-instance runners.

The config, the credentials and the trajectory format are delta's own; upstream's config
files and `.traj.json` files are not read. The agent runs on a harness conversation, so
every model turn, tool round and input is also committed to a SQLite database beside the
trajectory. How the pieces map onto the harness is in
[docs/design/mini-swe-agent.md](docs/design/mini-swe-agent.md).

## Install

```bash
cargo install --path examples/mini-swe-agent
```

This installs `mini` and `mini-extra`.

## Models and credentials

A model is a delta-ai provider id and a model id:

```bash
mini -m anthropic/claude-sonnet-4-5
mini -m openrouter/minimax/minimax-m2     # the id is everything after the first /
mini -m gpt-5                             # a bare id works when one provider has it
```

`mini-extra models` lists the models whose provider has credentials (`--all` lists every
model). A provider's credentials come from one of two places:

- **A stored credential**: `mini-extra login [provider]` runs the provider's OAuth login
  where it has one, or stores an API key (`--api-key`). Credentials are kept in `auth.json`
  in the home directory, in the shape of pi's `auth.json`, so a file copied from pi works.
  `mini-extra logout <provider>` removes one.
- **The provider's environment variable**: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and so on.

A stored credential wins over the variable.

`mini-extra config setup` picks a provider and a model, logs in when the provider has no
credentials, and saves the model as the default in the global config.

### The home directory

The home directory is `$DELTA_MINI_HOME`, else `delta-mini` in the user's config directory
(`~/.config/delta-mini` on Linux). It holds:

- `config.yaml`: the global config;
- `auth.json`: the credentials;
- the prompt history;
- `last_mini_run.traj.json` and `last_swebench_single_run.traj.json`.

`mini-extra config show` prints the global config and which providers have stored
credentials; `mini-extra config edit` opens the global config in `$EDITOR`.

## Config

A run's config is YAML, merged in this order, later layers winning:

1. the global `config.yaml`;
2. the `-c` files (built-in names such as `mini.yaml` work from anywhere);
3. `-c key.path=value` pairs (the value is parsed as JSON when it is JSON);
4. command-line options (`-m`, `--thinking`, `-y`, `-l`, ...).

Unknown keys are errors.

```yaml
agent:
  system_template: |          # Jinja; see "Templates" below
    You are a helpful assistant that can interact with a computer.
  instance_template: |
    Please solve this issue: {{task}}
  step_limit: 0               # model calls; 0: no limit
  cost_limit: 3.0             # USD; 0: no limit (default 3)
  time_limit: 0               # seconds; 0: no limit
  format_error_limit: 3       # format errors in a row that end the run
  mode: confirm               # confirm | yolo | human
  whitelist: ["ls", "cat "]   # regexes matched at the start; never confirmed
  confirm_exit: true          # ask before finishing: a new task, or Enter

model:
  provider: anthropic         # a delta-ai provider id, one from `providers`, or faux
  id: claude-sonnet-4-5
  thinking: medium            # off | minimal | low | medium | high | xhigh | max
  temperature: 0.0
  max_tokens: 8192            # output tokens per call
  sampling: { top_p: 0.9 }    # added to the request body as they are
  base_url: https://proxy.example.com/v1   # replaces the model's base URL
  headers: { X-Team: infra }  # added to the model's headers
  actions: tool_calls         # tool_calls (the bash tool) | text (a fenced block)
  action_regex: ...           # text actions: the block; its first group is the command
  observation_template: ...
  format_error_template: ...

environment:
  type: local                 # local | docker
  cwd: /path/to/repo          # local default: the current directory
  env: { PAGER: cat }
  timeout: 30                 # seconds per command
  # docker only: image, executable, run_args, forward_env, container_timeout,
  # pull_timeout, interpreter, and startup_command (rendered with the SWE-bench instance)
```

### A provider of your own

`providers` defines providers delta-ai does not ship, such as a local OpenAI-compatible
server. They are built with delta-ai's `create_provider`.

```yaml
providers:
  ollama:
    name: Ollama
    api: openai-completions   # openai-completions | openai-responses | anthropic-messages
                              # | google-generative-ai | mistral-conversations
    base_url: http://localhost:11434/v1
    # api_key_env: [OLLAMA_API_KEY]   # without it the provider needs no key
    models:                   # delta-ai model definitions; every field but id has a default
      - id: qwen3-coder:30b
        contextWindow: 131072
        cost: { input: 0, output: 0 }   # USD per million tokens

model:
  provider: ollama
  id: qwen3-coder:30b
```

A model definition may set its own `api` and `baseUrl`, so one provider can serve several
APIs. A provider with the id of a built-in one replaces it. With `api_key_env`, `mini-extra
login <id> --api-key` stores a key for it. A model without a price cannot be stopped by
`cost_limit`; `mini` warns about it at the start.

### Templates

Templates are Jinja (minijinja, strict: an unknown variable is an error). Every template
gets:

- `task`;
- `system`, `release`, `version` and `machine` (as `uname` reports them);
- the `agent`, `model` and `environment` sections;
- `n_model_calls`, `model_cost` and `elapsed_seconds`.

`observation_template` also gets `output.output`, `output.returncode` and
`output.exception_info`. `format_error_template` also gets:

- `error`;
- `stop_reason` (delta-ai's: `stop`, `length`, `toolUse`);
- `has_tool_calls`;
- `actions` (the text actions found).

### Built-in configs

| Name | What |
| --- | --- |
| `mini.yaml` | `mini`'s default: tool calls, confirm mode, $3 cost limit |
| `mini_textbased.yaml` | the same with text actions (one fenced block per answer) |
| `default.yaml` | text actions, no limits |
| `swebench.yaml` | SWE-bench in Docker, tool calls |
| `swebench_backticks.yaml` | SWE-bench with text actions in backticks |
| `swebench_xml.yaml` | SWE-bench with text actions in XML tags |

The SWE-bench configs name a model (`anthropic/claude-sonnet-4-5-20250929`, and
`openrouter/minimax/minimax-m2` for the XML one); `-m` replaces it.

## mini

```bash
mini                                   # asks for the task
mini -t "Fix the failing test" -y      # yolo: no confirmation
mini -c mini.yaml -c agent.cost_limit=1 -o run.traj.json
mini --thinking high --environment docker -c environment.image=python:3.12
```

At a prompt:

- `/y` switches to yolo mode, `/c` to confirm mode;
- `/u` switches to human mode, where you type the commands;
- `/m` reads several lines;
- `/h` lists the commands.

Ctrl+C interrupts the running step and asks for a comment for the model. When the agent
wants to finish, type a new task or press Enter.

## Trajectories

Each step saves the trajectory (by default `last_mini_run.traj.json` in the home
directory). The file is JSON with `format: delta-mini/1`:

- `exit`: the status and the submission.
- `stats`: calls, cost and elapsed seconds.
- `model`: the delta-ai model the run used.
- `config`: the sections as run.
- `entries`: the delta-ai messages the model received, each with its `kind` and what
  produced it:
  - `system` and `task`;
  - `answer`, with the actions parsed from it and its cost;
  - `observation`, with the command's output;
  - `format_error`, `interrupt` and `human`.

The exit statuses are `submitted`, `limits_exceeded`, `time_exceeded`, `format_errors`,
`interrupted`, `eof` and `error` (with `exit.error` saying what failed).

Browse trajectories with the inspector:

```bash
mini-extra inspect run.traj.json       # or a directory: every *.traj.json below it
```

Inspector keys:

- `l`/`h` (or the arrows) move between steps; `0`/`$` jump to the first and last.
- `j`/`k` scroll.
- `L`/`H` move between trajectories.
- `e`/`E` open the step or the file in `jless`.
- `r` shows or hides thinking.
- `R` reloads.
- `q` quits.

## SWE-bench

```bash
mini-extra swebench --subset verified --split test --slice 0:10 -w 4 -m anthropic/claude-sonnet-4-5 -o results
mini-extra swebench-single --subset lite --split dev -i 0
```

Each instance runs in its SWE-bench Docker image. The batch runner writes:

- `results/<id>/<id>.traj.json`;
- `results/preds.json`, for the SWE-bench harness;
- `results/exit_statuses_<time>.yaml`.

Options:

- It skips instances `preds.json` already holds, unless `--redo-existing`.
- `--filter` takes a regex on the id; `--shuffle` shuffles.
- `--total-cost-limit` and `--total-call-limit` stop the whole batch from starting more model
  calls.
- `--subset` also takes a local JSON or JSON Lines file.

## License

The prompt templates in `config/` come from upstream's configs, under its MIT license
([config/LICENSE-mini-swe-agent.md](config/LICENSE-mini-swe-agent.md)).
