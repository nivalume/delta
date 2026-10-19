# delta (pi-rs): notes for contributors and coding agents

Rust port of [pi](https://github.com/earendil-works/pi). The Python port lives in
[nivalume/pi-py](https://github.com/nivalume/pi-py) and is a useful second reference.

## Layout

- `crates/<upstream package>`: one crate per upstream package. crates.io names are
  `delta-*` (the `pi-*` names belong to upstream); library names are `delta_<package>`, e.g.
  `crates/ai` is published as `delta-ai` and imported as `delta_ai`.
- `crates/<crate>/docs/design/`: design documents (English), one per module. Upstream
  design wins; any deviation is listed in the document and confirmed with the maintainer
  before it is implemented.
- `docs/research/`: notes on reference implementations.
- `.claude/skills/diagram-design`: the diagram skill used for every design diagram.

## Rules

- Wire compatibility: serialized types (messages, models, credentials) keep upstream's
  JSON shape, so transcripts and `auth.json` files move between pi, pi-py and delta.
- Providers: use the vendor's official Rust SDK when one exists (AWS Bedrock:
  `aws-sdk-bedrockruntime`; Vertex AI: `google-cloud-aiplatform-v1`), otherwise hand-write
  the client on `reqwest` + `delta_ai::http`. No third-party SDK crates.
- Zed's provider crates are GPL-3.0: read them for behavior, never copy code.
  `docs/research/provider-references.md` lists the behaviors to cover.
- Commits are authored by `nivalume <25273209+nivalume@users.noreply.github.com>` and
  carry no AI attribution lines.

## Checks

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

Tests must not hit the network; provider tests replay recorded SSE bodies (or use
`wiremock`).
