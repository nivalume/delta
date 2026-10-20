# delta-harness

Durable agent harness; see [`docs/design/`](docs/design/README.md).

Implemented so far: the kernel (`records`, `documents`, `storage` with memory and SQLite
backends and the storage conformance suite, `session`, `scheduler`), the local execution
environment (`env`), and the agent core: the `Harness` surface with conversations and
submissions, the registry of extensions and hooks, agent resolution and settings, the inbox,
model context and the positional system prompt, the `delta.generation` task with streaming,
retries, deferred polling and tool rounds, and usage. The tool and compaction tasks,
approval and observation follow.
