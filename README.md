# Delta

Delta is a Rust port of the libraries behind [pi](https://github.com/earendil-works/pi), a minimal,
extensible agent harness. It provides a unified LLM API across providers and a stateful agent runtime
with tool calling.

Messages, models and credentials keep upstream's JSON shape, so transcripts and `auth.json` entries
move between pi, [pi-py](https://github.com/nivalume/pi-py) and delta.

Status: in development. Nothing is published to crates.io yet.

## Getting started

Add the crates you need. Until the first release, depend on the git repository:

```toml
[dependencies]
delta-ai = { git = "https://github.com/nivalume/pi-rs" }
delta-agent = { git = "https://github.com/nivalume/pi-rs" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

After the first release: `cargo add delta-ai delta-agent`.

Delta requires Rust 1.90 or newer (edition 2024) and a Tokio runtime.

Set the API key of a provider in its environment variable, for example `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY`, or pass credentials through a credential store. Then follow the quick start of
[delta-ai](crates/ai/README.md#quick-start) to call a model, or of
[delta-agent](crates/agent/README.md#quick-start) to run an agent with tools.

## Run with Nix

Not available. Delta has no command-line interface yet; the crates build with Cargo alone.

## Packages

This workspace contains one crate per ported upstream package.

| Crate | Import | Upstream | Description |
|-------|--------|----------|-------------|
| **[delta-ai](crates/ai)** | `delta_ai` | `@earendil-works/pi-ai` | Unified multi-provider LLM API (OpenAI, Anthropic, Google, etc.) |
| **[delta-agent](crates/agent)** | `delta_agent` | `@earendil-works/pi-agent-core` | Agent runtime with tool calling and state management |

Not ported yet: `@earendil-works/chord`, `@earendil-works/pi-telemetry`, `@earendil-works/pi-durable`,
`@earendil-works/pi-coding-agent` and `@earendil-works/pi-tui`.

Each crate has design notes in `crates/<crate>/docs/design/`. They describe the modules and list
every difference from upstream.

## Permissions & Containerization

Delta does not include a permission system for restricting filesystem, process, network, or
credential access. Agent tools run in your process with its permissions.

To restrict tool calls, block them in `AgentHooks::before_tool_call` (see
[delta-agent](crates/agent/README.md)). For stronger boundaries, run the process that hosts the agent
in a container or sandbox.

## Contributing

See [AGENTS.md](AGENTS.md) for project rules (for both humans and agents): wire compatibility with
upstream, which provider SDKs may be used, design notes, tests and commit conventions.

Upstream's design wins. A difference from upstream that Rust does not force is listed in the crate's
design notes under "Pending confirmation" and is implemented only after the maintainer confirms it.

## Development

```bash
cargo fmt --all                                                      # Format
cargo clippy --workspace --all-targets --all-features -- -D warnings # Lint
cargo test --workspace --all-features                                # Run tests, including README examples
cargo test --workspace --no-default-features                         # Run tests without optional features
cargo doc --workspace --all-features --no-deps                       # Build API docs
scripts/sync-models.sh [version]                                     # Refresh the model catalog (needs npm)
```

Tests do not hit the network. Provider tests replay recorded SSE bodies or use `wiremock`.

`scripts/sync-models.sh` copies the provider catalogs in `crates/ai/data/providers` from the
`@earendil-works/pi-ai` npm package (default: latest version).

### Using local packages outside the workspace

Point a dependency at your checkout:

```toml
[dependencies]
delta-agent = { path = "../pi-rs/crates/agent" }
```

If your project already depends on delta from git or crates.io, override both crates with a patch, so
that your code and `delta-agent` use the same `delta-ai`:

```toml
[patch."https://github.com/nivalume/pi-rs"]   # or [patch.crates-io]
delta-ai = { path = "../pi-rs/crates/ai" }
delta-agent = { path = "../pi-rs/crates/agent" }
```

## Building standalone binaries from release source

Not applicable. Delta ships libraries only. A release publishes the crates to crates.io from a `v*`
tag (`.github/workflows/release.yml`).

## Supply-chain hardening

We treat dependency changes as reviewed code changes.

- `Cargo.lock` is committed.
- CI runs [cargo-deny](https://github.com/EmbarkStudios/cargo-deny) with `deny.toml`: it rejects
  crates with security advisories and yanked versions, licenses outside an allowlist, wildcard version
  requirements, and crates from unknown registries or git sources.
- Providers use the vendor's official Rust SDK when one exists (`aws-sdk-bedrockruntime`,
  `google-cloud-aiplatform-v1`), behind the optional `bedrock` and `vertex` features. Other providers
  use a hand-written client on `reqwest` with rustls. No third-party provider SDK crates.
- Dependabot proposes weekly updates for Cargo dependencies and GitHub Actions.
- Releases publish to crates.io with Trusted Publishing, so no long-lived registry token is stored.
  The first release of a new crate needs a `CARGO_REGISTRY_TOKEN` secret.
- CI checks formatting, clippy, tests on Linux, macOS and Windows (with all features and with none),
  the minimum supported Rust version, docs, and `cargo package`.

## License

MIT. See [LICENSE](LICENSE).
