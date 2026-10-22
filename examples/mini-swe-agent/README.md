# mini-swe-agent on delta-harness

A port of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (2.4.6) built on
`delta-harness`: the `mini` agent with its confirm, yolo and human modes, local and Docker
environments, upstream-compatible trajectory files (`.traj.json`, `mini-swe-agent-1.1`),
the trajectory inspector, and the SWE-bench batch and single-instance runners.

The configs are upstream's (YAML and Jinja, in `config/`, built in). The agent runs on a
harness conversation, so every model turn, tool round and input is also committed to a
SQLite database beside the trajectory. How upstream's loop maps onto the harness, and what
differs, is in [docs/design/mini-swe-agent.md](docs/design/mini-swe-agent.md).

## Install

```bash
cargo install --path examples/mini-swe-agent
```

This installs `mini` and `mini-extra`.

## Models

Models are delta-ai's. Name them as litellm does, provider first:

```bash
export ANTHROPIC_API_KEY=...
mini -m anthropic/claude-sonnet-4-5
```

`openai/...`, `gemini/...` (delta-ai's `google`), `openrouter/...` and the other delta-ai
providers work the same way. `mini-extra config setup` stores a default model and an API key
in the global config file (`.env` in `~/.config/mini-swe-agent`, or `$MSWEA_GLOBAL_CONFIG_DIR`).

`model.model_kwargs` sets `temperature`, `max_tokens` and `reasoning_effort`; other
OpenAI-style sampling keys (`top_p`, `seed`, ...) pass through to OpenAI-compatible APIs.

## mini

```bash
mini                                   # asks for the task
mini -t "Fix the failing test" -y      # yolo: no confirmation
mini -c mini.yaml -c agent.cost_limit=1 -o run.traj.json
```

At a prompt, `/y` switches to yolo mode, `/c` to confirm mode, `/u` to human mode (you type
the commands), `/m` reads several lines and `/h` lists them. Ctrl+C interrupts the running
step and asks for a comment for the model. When the agent wants to finish, type a new task
or press Enter.

`-c` takes config files, built-in config names and `key.path=value` pairs; they merge in
order. Setting `-c` replaces the default `mini.yaml`, so name it too.

## Trajectories

Each step saves the trajectory (default `last_mini_run.traj.json` in the global config
directory). Browse trajectories with the inspector:

```bash
mini-extra inspect run.traj.json       # or a directory: every *.traj.json below it
```

`l`/`h` (or the arrows) move between steps, `0`/`$` jump to the first and last, `j`/`k`
scroll, `L`/`H` move between trajectories, `e`/`E` open the step or the file in `jless`, `r`
toggles reasoning, `R` reloads and `q` quits.

## SWE-bench

```bash
mini-extra swebench --subset verified --split test --slice 0:10 -w 4 -m anthropic/claude-sonnet-4-5 -o results
mini-extra swebench-single --subset lite --split dev -i 0
```

Each instance runs in its SWE-bench Docker image. The batch runner writes
`results/<id>/<id>.traj.json`, `results/preds.json` (for the SWE-bench harness) and
`results/exit_statuses_<time>.yaml`, skips instances `preds.json` already holds unless
`--redo-existing`, and takes `--filter` (a regex on the id) and `--shuffle`. `--subset`
also takes a local JSON or JSON Lines file.

## License

The configs in `config/` are upstream's, under its MIT license
([config/LICENSE-mini-swe-agent.md](config/LICENSE-mini-swe-agent.md)).
