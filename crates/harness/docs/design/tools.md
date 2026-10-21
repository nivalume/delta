# Built-in tools (`tools`)

Follows the tool paragraphs at the end of spec §7.3 (`read`, `write`, `edit`, `bash`, the
file mutation queue, output limits, images) and the `bash` / spill paragraphs of the same
section. Corresponds to upstream `src/tools/{read,write,edit,edit-diff,bash,
file-mutation-queue,path-utils,env}.ts` and `src/truncate.ts`.

## Problem

The coding agent needs four tools that work only through `api.env` ([env.md](env.md)) so
they run wherever the conversation does, bound what the model sees, report everything else
as diagnostics, and survive concurrent calls on one file. They are ported from the agent
harness's tools with durable's channels (output, structured output, details, diagnostics).

```rust
impl ReadTool { pub fn new() -> Self }
impl WriteTool { pub fn new(queue: FileMutationQueue) -> Self }
impl EditTool { pub fn new(queue: FileMutationQueue) -> Self }
impl BashTool { pub fn new(options: BashOptions) -> Self }
pub fn coding_tools(bash: BashOptions) -> Extension;  // "coding-tools": all four, one queue; nothing installs it automatically
```

All four are `ToolReplay::Unsafe`, `callers` both, and raise `No execution environment is
configured` (an ordinary `tool_error` result) when `api.env()` is `None`. The module is a
side module: it names only `env` and the tool seams (`tool`, `registry`), which
`tests/layering.rs` checks.

`agent::LocalEnvBuilder::new(default_cwd)` is the usual `HarnessOptions.env`: one cached
`LocalEnv` per directory, the conversation's `cwd` or else the default.

## Output limits

```rust
pub const DEFAULT_MAX_LINES: usize = 2000;
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;
// tool::DEFAULT_OUTPUT_LIMITS: { max_bytes: 50 KiB, max_lines: 2000, retain: Head }

pub struct Truncation { pub truncated: bool, pub truncated_by: Option<Limit /* Lines | Bytes */>, pub total_lines: usize, pub total_bytes: usize,
                        pub output_lines: usize, pub output_bytes: usize, pub last_line_partial: bool, pub first_line_exceeds_limit: bool,
                        pub max_lines: usize, pub max_bytes: usize }
pub fn truncate_head(text: &str, limits: &OutputLimits) -> (&str, Truncation);      // `retain` ignored
pub fn truncate_head_of(prefix: &str, totals: Totals /* lines, bytes */, limits) -> (&str, Truncation);
pub fn format_size(bytes: u64) -> String;   // 512B, 1.5KB, 2.0MB
```

`truncate_head` keeps whole lines until either limit is hit (lines are counted ignoring a
trailing newline; empty text has none). A first line longer than `max_bytes` yields empty
content with `first_line_exceeds_limit`. `truncate_head_of` computes the same result from
a prefix plus known totals when the prefix is the whole text, longer than `max_bytes + 1`
bytes, or holds at least `max_lines` newlines. Streaming output is bounded by
`delta_chord::OutputBuffer` instead ([tool.md](tool.md)); these functions serve `read`.

## Paths

- `normalize_tool_path`: Unicode spaces (U+00A0, U+2000–U+200A, U+202F, U+205F, U+3000)
  become ASCII spaces; a leading `@` is dropped. `resolve_tool_path(env, path)` then takes
  `env.absolute_path`.
- `resolve_read_path` additionally tries variants that exist on disk: ` AM.`/` PM.` with a
  narrow no-break space (macOS screenshots), NFD normalization, `'` → `’`, and both; the
  first that exists wins, else the plain resolution.

## `read`

The description is upstream's without the image sentence: "Read the contents of a text
file. Output is truncated to 2000 lines or 50KB ...".

Schema: `{ path: string, offset?: number (1-based line), limit?: number }`. Details:
`{ truncation?: Truncation }` (without the content).

1. Open the file with `open_binary_reader`. Loop at most twice: take `info` before, sniff
   the header for an image signature, read as text (or image, below), take `info` after.
   Unchanged size and mtime, or a text file that only grew (a log being appended), returns
   the result; otherwise read again once, then fail `<path> changed while it was read`.
2. Text: `start_line = max(0, offset - 1)`; `scan_lines(start_line, start_line + limit)`
   in one pass (a fractional or non-finite index is handled as JS `slice` would: NaN is
   0, values truncate toward zero; a negative end counts from the end and needs a second
   scan once the line count is known). `offset` at or past the line count fails
   `Offset <n> is beyond end of file (<total> lines total)`. Read only the head: from
   `scan.start`, 64 KiB chunks, decoding as a stream, until the end of the selection or
   until the text holds more than `DEFAULT_MAX_BYTES + 1` bytes or `DEFAULT_MAX_LINES`
   newlines (skipping a leading BOM). Apply `truncate_head_of` with the scan's totals.
   The result equals decoding the whole file, splitting on `\n`, slicing, and
   `truncate_head`.
3. Diagnostics:
   - first line over the limit: show its first `DEFAULT_MAX_BYTES` bytes cut on a character
     boundary (`char_end`); `warn` `truncated`: `Line <n> is <size>, exceeds the 50.0KB
     limit; showing its first <k>. Use bash: sed -n '<n>p' <path> | tail -c +<k+1>`;
     details `output_bytes = k, output_lines = 1`.
   - truncated: `info` `truncated`: `Showing lines <a>-<b> of <total>[ (50.0KB limit)].
     Use offset=<b+1> to continue.`; details carry the truncation.
   - a user `limit` that stopped before the end: `info`: `<n> more lines in file. Use
     offset=<k> to continue.` (no code).
4. Output: one text item, or none for empty text.

Images: detection by content (PNG, JPEG, GIF, WebP signatures; a BMP header must also give
the file's size, so text starting with `BM` stays text) stays, because binary
in the context is never useful; in phase 1 a detected image returns the error result with
code `unsupported_image` (`<path> is an image (<mime>); image input is not supported in this
build`). The image path (pass-through within the model's `input_limits.images.resize`
limits, an `ImageProcessor` seam for resizing and conversion, the `image` info diagnostic,
the placeholder for models without vision) is deferred with the images decision in
[README.md](README.md).

## `write`

Schema: `{ path: string, content: string }`. Resolve the path, take the file's mutation
slot, check cancellation, `write_file` (parents created), check cancellation, output
`Successfully wrote to <path>`.

## `edit`

Schema: `{ path: string, edits: [{ oldText: string, newText: string }] }`. Details:
`{ diff: string, patch: string, firstChangedLine?: number }`.

`prepare_arguments` repairs common shapes on a copy: `edits` as a JSON string (an array,
or a single edit object), `edits` as a single edit object, and a top-level
`oldText`/`newText` pair appended to `edits`. Empty `edits` fails `Edit tool input is
invalid. edits must contain at least one replacement.`

Inside the file's mutation slot:

1. `file_info`: a `FileError` maps to `Could not edit file: <path>. Error code: <code>.`;
   a kind other than file or symlink fails `Path is not a file`.
2. `read_text_file`; strip a BOM; detect the line ending (`\r\n` when the first `\r\n`
   precedes the first `\n`); normalize to LF (also lone `\r`).
3. Apply the edits against the same original (never incrementally), each `oldText` /
   `newText` LF-normalized:
   - an empty `oldText` fails;
   - exact match first; else fuzzy: NFKC, trailing whitespace per line stripped, smart
     quotes to `'`/`"`, Unicode dashes (U+2010–U+2015, U+2212) to `-`, special spaces to
     ` `; when any edit needs fuzzy matching, all edits are located in the normalized text;
   - not found: `Could not find the exact text in <path>. The old text must match exactly
     including all whitespace and newlines.` (`edits[i]` named when several);
   - more than one occurrence (counted in normalized space): `Found <n> occurrences ...
     The text must be unique. Please provide more context to make it unique.`;
   - overlapping matches: `edits[i] and edits[j] overlap in <path>. Merge them into one
     edit or target disjoint regions.`;
   - replacements apply from the last match to the first so offsets stay stable; with
     fuzzy matching, only the changed line blocks are taken from the normalized text and
     unchanged lines keep their original bytes;
   - identical content afterwards: `No changes made to <path>. The replacement produced
     identical content. ...`.
4. Restore the BOM and line endings, `write_file`, check cancellation between steps.
5. Details: a diff string with `firstChangedLine` and a unified patch (4 context lines)
   from `similar`; output `Successfully replaced <n> block(s) in <path>.`

## `bash`

Schema: `{ command: string, timeout?: number (seconds) }`; `output_limits` retain `Tail`;
`structured_output_schema`: `{ output: string, truncated: boolean, fullOutputPath?: string,
exitCode: number }`.

```rust
pub struct BashOptions { pub command_prefix: Option<String>, pub prepare: Option<Arc<dyn BashPrepare>> }
pub struct BashExecution { pub command: String, pub cwd: String, pub env: BTreeMap<String, String>, pub inherit_env: bool }
pub trait BashPrepare: Send + Sync {
    fn prepare<'a>(&'a self, execution: &'a mut BashExecution, api: &'a ToolApi<'a>, ctx: &'a Context) -> BoxFuture<'a, Result<(), ToolError>>;
}
```

1. `timeout` must be finite, positive and at most 2 147 483.647 s, else
   `Invalid timeout: ...`.
2. Build the execution: `command_prefix + "\n" + command` when a prefix is set, the
   environment's `cwd`, empty env, `inherit_env: true`; then `prepare` may change it.
3. `env.exec(Command::Shell(command), ExecOptions { cwd, env, inherit_env, timeout,
   on_output: chunk => api.output_str(text) after api.output_skipped(skipped) when the
   chunk reports one, spill: { after_bytes: DEFAULT_MAX_BYTES, after_lines:
   DEFAULT_MAX_LINES }, window: api.output_window() })`. The stream name is ignored.
4. A spill path (from the result or the error) is an `info` diagnostic `full_output`:
   `Full output: <path>`.
5. Errors: `Aborted` with the context cancelled propagates (the task is being aborted);
   `Timeout` fails `Command timed out after <n> seconds`; `Aborted` otherwise fails
   `Command aborted`; other codes fail with their message. Each becomes a `tool_error`
   result that still carries the retained output and diagnostics.
6. Otherwise `structured_output = { output: retained.text, truncated: retained.truncated,
   fullOutputPath?, exitCode }` from `api.retained_output()`; exit 0 is a plain result
   (the model gets the retained tail as `output`); nonzero is `is_error` with an `error`
   diagnostic `exit_code`: `Command exited with code <n>`.

## File mutation queue

```rust
#[derive(Clone, Default)]
pub struct FileMutationQueue { /* Arc<Mutex<HashMap<Key, Weak<tokio::sync::Mutex<()>>>>> */ }
impl FileMutationQueue {
    pub async fn run<T>(&self, env: &dyn ExecutionEnv, path: &str, ctx: &Context, mutation: impl Future<Output = T>) -> FileResult<T>;
}
```

`edit` and `write` serialize their read-modify-write of one file. The key is
`(env.id(), canonical path)`: for a file that does not exist yet, the canonical parent
joined with the name (recursively, so a `write` that creates a file and a later `edit` of
it share one key even under a symlinked directory); `NotSupported` from `canonical_path`
falls back to the absolute path. The queue hands out one fair tokio mutex per key (dead
entries are pruned as new ones are made), so holders run in the order their keys
resolved; other files and other file systems never wait; waiting is cancellable
(`aborted`). It is not a lock against `bash` or other processes.

The queue is a value, not a process-wide map (the architecture rules out module-level
state): `coding_tools` gives its `write` and `edit` one queue, and a host that builds the
tools itself passes them a shared one. Two queues do not see each other's mutations.

## Deferrals

- Images in `read` (above) and the `ImageProcessor` seam.
- `tool_search` and deferred tools: the seam is `Agent.deferred` plus
  `ToolControl.add_tools` written to `model_tools` by the generation's `tools` phase
  ([generation.md](generation.md)); the BM25 ranking extension is later.
- `powershell`: not ported in phase 1.
- Code mode and a remote environment: later, as [README.md](README.md) lists.

## Differences from upstream

- `truncate_head` returns a slice of its input (the kept lines are always a prefix) and
  counts bytes natively.
- The mutation queue is a value shared by the tools that hold it, not a process-wide map,
  and a map of weak async mutexes instead of a promise chain; same ordering guarantee
  within one queue.
- `Truncation` serializes snake_case in `read`'s details; the model- and program-facing
  shapes keep upstream's names (`oldText`/`newText` arguments, `firstChangedLine`,
  `fullOutputPath`, `exitCode`).
- `read` describes itself as a text reader; BMP detection checks the header's size field.
- `edit`'s diff and patch come from `similar` (upstream `diff`), so the diff string's
  exact lines may differ; the tests compare semantics (changed lines, first changed line).
- Image handling is a deferral rather than a pass-through; the error code is the same
  `unsupported_image` so the model's experience matches an upstream build without a
  processor.
- `BashPrepare` is a trait object instead of a function type.

## Decided

- Detected images return `unsupported_image` in phase 1, as an upstream build without an
  image processor does for formats it cannot pass through.
- Upstream's error and diagnostic texts are kept word for word (models were tuned against
  them).
- `read` without `limit` on a huge file reports the `offset=` continuation only, as
  upstream.
