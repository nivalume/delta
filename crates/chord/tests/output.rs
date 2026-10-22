//! Tests of `delta_chord::output`, ported from upstream's `output.test.ts` and
//! `output-skip.test.ts`.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use pretty_assertions::assert_eq;
use tokio::sync::Notify;
use tokio::time::Instant;

use delta_chord::output::{
    BoundedOutput, OutputBuffer, OutputLimits, OutputSkip, OutputSlice, PROGRESS_BYTES_PER_SECOND, Progress, Retain,
    bound_output, char_end, sanitize_output,
};

fn head(max_lines: usize, max_bytes: usize) -> OutputLimits {
    OutputLimits {
        max_bytes,
        max_lines,
        retain: Retain::Head,
    }
}

fn tail(max_lines: usize, max_bytes: usize) -> OutputLimits {
    OutputLimits {
        max_bytes,
        max_lines,
        retain: Retain::Tail,
    }
}

fn slice(text: &str, dropped_bytes: usize, dropped_lines: usize) -> OutputSlice<'_> {
    OutputSlice {
        text,
        dropped_bytes,
        dropped_lines,
    }
}

fn output(text: &str, dropped_bytes: usize, dropped_lines: usize) -> BoundedOutput {
    BoundedOutput {
        text: text.to_owned(),
        dropped_bytes,
        dropped_lines,
    }
}

fn expected_output(stream: &str, limits: &OutputLimits) -> BoundedOutput {
    let kept = bound_output(stream, limits);
    output(&sanitize_output(kept.text), kept.dropped_bytes, kept.dropped_lines)
}

/// Measure omitted text as a producer reports it.
fn measure(text: &str) -> OutputSkip {
    OutputSkip {
        bytes: text.len(),
        newlines: text.matches('\n').count(),
        ends_with_newline: text.ends_with('\n'),
    }
}

/// A byte-wise tail cut on a character boundary, like Node's `Buffer` slice.
fn byte_tail(content: &str, max_bytes: usize) -> &str {
    if content.len() <= max_bytes {
        return content;
    }
    let mut start = content.len() - max_bytes;
    while !content.is_char_boundary(start) {
        start += 1;
    }
    &content[start..]
}

/// A single line longer than the byte limit is cut like a byte-wise tail on a character
/// boundary.
fn assert_matches_byte_tail(input: &str, max_byte_values: &[usize]) {
    for &max_bytes in max_byte_values {
        let kept = bound_output(input, &tail(10, max_bytes)).text;
        assert_eq!(
            kept,
            byte_tail(input, max_bytes),
            "input={input:?} max_bytes={max_bytes}"
        );
        assert!(kept.len() <= max_bytes, "tail output exceeded {max_bytes} bytes");
    }
}

fn sampled_byte_limits(input: &str) -> Vec<usize> {
    let total = input.len();
    let mut values: Vec<usize> = [0, 1, 2, 3, 4, 5, 8, total / 2, total + 1, total]
        .into_iter()
        .chain(total.checked_sub(4))
        .chain(total.checked_sub(1))
        .collect();
    values.sort_unstable();
    values.dedup();
    values
}

/// Deterministic PRNG (mulberry32) so failures reproduce from the printed seed.
fn mulberry32(seed: u32) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state.wrapping_add(0x6d2b_79f5);
        let mut t = state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 0x3d));
        f64::from(t ^ (t >> 14)) / 4_294_967_296.0
    }
}

fn pick<T: Copy>(next: &mut impl FnMut() -> f64, items: &[T]) -> T {
    items[(next() * items.len() as f64) as usize]
}

// --- bound_output and sanitize_output ---

#[test]
fn removes_control_characters_but_keeps_tabs_newlines_and_other_text() {
    assert_eq!(
        sanitize_output("a\0b\tc\nd\re\u{7}f\u{fff9}g\u{fffb}h😀"),
        "ab\tc\ndefgh😀"
    );
    assert!(matches!(
        sanitize_output("plain\ttext\n"),
        std::borrow::Cow::Borrowed(_)
    ));
}

#[test]
fn keeps_output_within_the_limits_unchanged() {
    assert_eq!(bound_output("a\nb\n", &head(2, 1000)), slice("a\nb\n", 0, 0));
    assert_eq!(bound_output("a\nb", &tail(2, 1000)), slice("a\nb", 0, 0));
    assert_eq!(bound_output("", &tail(2, 1000)), slice("", 0, 0));
    assert_eq!(bound_output("", &head(2, 1000)), slice("", 0, 0));
}

#[test]
fn keeps_nothing_with_a_zero_limit() {
    assert_eq!(bound_output("ab\ncd\n", &head(10, 0)), slice("", 6, 2));
    assert_eq!(bound_output("ab\ncd\n", &tail(0, 1000)), slice("", 6, 2));
}

#[test]
fn keeps_exact_slices_of_whole_lines_trailing_newline_included() {
    assert_eq!(bound_output("a\nb\nc\n", &head(2, 1000)), slice("a\nb\n", 2, 1));
    assert_eq!(bound_output("a\nb\nc\n", &tail(2, 1000)), slice("b\nc\n", 2, 1));
    assert_eq!(bound_output("a\nb\nc", &tail(2, 1000)), slice("b\nc", 2, 1));
    // Blank lines are lines.
    assert_eq!(bound_output("a\nb\nc\n\n", &tail(3, 1000)), slice("b\nc\n\n", 2, 1));
    // A final line without a newline counts.
    assert_eq!(bound_output("a\nb\nc", &head(2, 1000)), slice("a\nb\n", 1, 1));
}

#[test]
fn cuts_at_the_byte_limit_on_whole_lines_when_possible() {
    assert_eq!(bound_output("aa\nbb\ncc\n", &head(10, 7)), slice("aa\nbb\n", 3, 1));
    assert_eq!(bound_output("aa\nbb\ncc\n", &tail(10, 7)), slice("bb\ncc\n", 3, 1));
}

#[test]
fn cuts_a_single_line_longer_than_the_byte_limit_on_a_character_boundary() {
    // "é" is two bytes; five bytes hold two whole characters.
    assert_eq!(bound_output("ééé\n", &head(10, 5)), slice("éé", 3, 0));
    assert_eq!(bound_output("x\néééé", &tail(10, 5)), slice("éé", 6, 1));
}

#[test]
fn keeps_a_feff_at_the_start_of_a_kept_slice() {
    assert_eq!(bound_output("x\u{feff}a", &tail(10, 4)), slice("\u{feff}a", 1, 0));
}

#[test]
fn cuts_tails_of_multi_byte_edge_cases_exactly_like_a_byte_tail() {
    // Upstream's lone surrogates encode as U+FFFD; Rust strings hold the result directly.
    for input in ["a\u{fffd}", "\u{fffd}b", "a\u{fffd}b", "\u{fffd}🙂", "🙂\u{fffd}", "👩‍💻"] {
        let limits: Vec<usize> = (0..input.len() + 5).collect();
        assert_matches_byte_tail(input, &limits);
    }
}

#[test]
fn cuts_tails_exactly_like_a_byte_tail_across_deterministic_fuzz_cases() {
    const ALPHABET: [&str; 12] = [
        "a", "\u{7f}", "\u{80}", "é", "\u{7ff}", "\u{800}", "中", "\u{d7ff}", "\u{fffd}", "🙂", "\u{e000}", "\u{ffff}",
    ];
    fn check_exhaustive(prefix: &str, depth: usize) {
        assert_matches_byte_tail(prefix, &sampled_byte_limits(prefix));
        if depth == 0 {
            return;
        }
        for character in ALPHABET {
            check_exhaustive(&format!("{prefix}{character}"), depth - 1);
        }
    }
    check_exhaustive("", 3);
    // Upstream's linear congruential generator.
    let mut seed: u32 = 0x1234_5678;
    let mut random = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        f64::from(seed) / 4_294_967_296.0
    };
    for _ in 0..1_000 {
        let length = (random() * 80.0) as usize;
        let input: String = (0..length).map(|_| pick(&mut random, &ALPHABET)).collect();
        assert_matches_byte_tail(&input, &sampled_byte_limits(&input));
    }
}

#[test]
fn char_end_finds_boundaries_in_invalid_utf8() {
    let euro = "€".as_bytes();
    let bytes = [b"a".as_slice(), euro, b"b"].concat();
    assert_eq!(char_end(&bytes, 0), 0);
    assert_eq!(char_end(&bytes, 1), 1);
    assert_eq!(char_end(&bytes, 2), 1);
    assert_eq!(char_end(&bytes, 3), 1);
    assert_eq!(char_end(&bytes, 4), 4);
    assert_eq!(char_end(&bytes, 5), 5);
    // Past the end clamps to the length.
    assert_eq!(char_end(&bytes, 99), 5);
    // Stray continuation bytes and a truncated character.
    assert_eq!(char_end(&[0x80, 0x80, b'x'], 2), 2);
    assert_eq!(char_end(&[0x80, 0x80], 1), 0);
    assert_eq!(char_end(&[b'a', 0xe2, 0x82], 3), 3);
    assert_eq!(char_end(&[b'a', 0xe2, 0x82], 2), 1);
}

// --- OutputBuffer ---

#[test]
fn keeps_exact_totals_across_chunks_and_decodes_utf8_split_across_byte_chunks() {
    let mut buffer = OutputBuffer::new(tail(2, 1000));
    let bytes = "😀\n".as_bytes();
    buffer.push_str("a\nb\n");
    buffer.push_bytes(&bytes[..2]);
    buffer.push_bytes(&bytes[2..]);
    assert_eq!(buffer.snapshot(), output("b\n😀\n", 2, 1));
}

#[test]
fn decodes_a_character_split_into_single_bytes() {
    let mut buffer = OutputBuffer::new(head(10, 1000));
    for byte in "a😀€é".as_bytes() {
        buffer.push_bytes(std::slice::from_ref(byte));
    }
    buffer.end();
    assert_eq!(buffer.snapshot(), output("a😀€é", 0, 0));
}

#[test]
fn sanitizes_the_retained_text_but_counts_the_raw_stream() {
    let mut buffer = OutputBuffer::new(tail(1, 1000));
    buffer.push_str("a\u{7}\n");
    assert_eq!(buffer.snapshot(), output("a\n", 0, 0));
    buffer.push_str("b\u{1b}\n");
    assert_eq!(buffer.snapshot(), output("b\n", 3, 1));
}

#[test]
fn drops_a_byte_order_mark_only_at_the_start_of_the_output() {
    let bom = [0xef, 0xbb, 0xbf];
    let mut buffer = OutputBuffer::new(tail(10, 1000));
    buffer.push_bytes(&bom[..1]);
    buffer.push_bytes(&[0xbb, 0xbf, 0x61]);
    buffer.push_str("b");
    // A U+FEFF after a string chunk is text.
    buffer.push_bytes(&[0xef, 0xbb, 0xbf, 0x63]);
    buffer.end();
    assert_eq!(buffer.snapshot().text, "ab\u{feff}c");

    // A U+FEFF after other bytes is text too, and a string chunk keeps its own.
    let mut buffer = OutputBuffer::new(tail(10, 1000));
    buffer.push_bytes(b"x");
    buffer.push_bytes(&bom);
    assert_eq!(buffer.snapshot().text, "x\u{feff}");
    let mut buffer = OutputBuffer::new(tail(10, 1000));
    buffer.push_str("\u{feff}y");
    assert_eq!(buffer.snapshot().text, "\u{feff}y");
}

#[test]
fn flushes_an_incomplete_character_before_a_string_chunk_and_at_the_end() {
    let mut buffer = OutputBuffer::new(tail(10, 1000));
    let euro = "€".as_bytes();
    buffer.push_bytes(&euro[..1]);
    buffer.push_str("x");
    buffer.push_bytes(&euro[..2]);
    buffer.end();
    assert_eq!(buffer.snapshot(), output("\u{fffd}x\u{fffd}", 0, 0));
}

#[test]
fn replaces_invalid_bytes() {
    let mut buffer = OutputBuffer::new(tail(10, 1000));
    // A lone continuation byte, an impossible byte, and a lead byte the next chunk breaks.
    buffer.push_bytes(b"a\x80b\xffc\xe2");
    buffer.push_bytes(b"\x82d");
    buffer.push_bytes(b"\xf0\x9f");
    buffer.push_bytes(b"e");
    buffer.end();
    assert_eq!(buffer.snapshot().text, "a\u{fffd}b\u{fffd}c\u{fffd}d\u{fffd}e");
}

#[test]
fn decodes_byte_streams_like_lossy_decoding_of_the_whole() {
    // Valid and broken sequences, the byte-order mark, and newlines.
    const BYTES: [u8; 16] = [
        b'a', b'\n', 0xc3, 0xa9, 0xe2, 0x82, 0xac, 0xf0, 0x9f, 0x98, 0x80, 0xff, 0x80, 0xed, 0xa0, 0xef,
    ];
    for seed in 1..=500 {
        let mut next = mulberry32(seed);
        let length = (next() * 60.0) as usize;
        let mut stream: Vec<u8> = (0..length).map(|_| pick(&mut next, &BYTES)).collect();
        if next() < 0.3 {
            stream.splice(0..0, [0xef, 0xbb, 0xbf]);
        }
        let mut buffer = OutputBuffer::new(tail(1000, 10_000));
        let mut rest = stream.as_slice();
        while !rest.is_empty() {
            let cut = ((next() * 5.0) as usize).min(rest.len());
            buffer.push_bytes(&rest[..cut]);
            rest = &rest[cut..];
        }
        buffer.end();
        let decoded = String::from_utf8_lossy(&stream);
        let expected = decoded.strip_prefix('\u{feff}').unwrap_or(&decoded);
        assert_eq!(
            buffer.snapshot(),
            expected_output(expected, &tail(1000, 10_000)),
            "seed {seed}"
        );
    }
}

#[test]
fn matches_bounding_the_whole_stream_when_several_chunks_arrive_between_snapshots() {
    for limits in [head(3, 40), tail(3, 40), head(50, 25), tail(50, 25)] {
        let mut buffer = OutputBuffer::new(limits);
        let mut stream = String::new();
        for index in 0..300 {
            let chunk = if index % 7 == 0 {
                format!("{}\n", "é".repeat(index % 30))
            } else {
                format!("line {index}\n")
            };
            stream.push_str(&chunk);
            buffer.push_str(&chunk);
            if index % 5 != 4 {
                continue;
            }
            assert_eq!(
                buffer.snapshot(),
                expected_output(&stream, &limits),
                "{limits:?} at {index}"
            );
        }
    }
}

#[test]
fn matches_bounding_the_whole_stream_for_byte_chunks_cut_anywhere() {
    for limits in [
        head(3, 17),
        tail(3, 17),
        head(40, 9),
        tail(40, 9),
        tail(1, 1),
        head(1, 1),
    ] {
        let mut next = mulberry32(7);
        let mut buffer = OutputBuffer::new(limits);
        let mut stream = String::new();
        for index in 0..200 {
            let chunk = pick(&mut next, &["é\n", "€€", "😀 x\n", "\n", "abc"]).repeat(index % 4);
            let mut rest = chunk.as_bytes();
            while !rest.is_empty() {
                let cut = ((next() * 4.0) as usize).min(rest.len());
                buffer.push_bytes(&rest[..cut]);
                rest = &rest[cut..];
            }
            stream.push_str(&chunk);
            assert_eq!(
                buffer.snapshot(),
                expected_output(&stream, &limits),
                "{limits:?} at {index}"
            );
            assert!(buffer.stored_bytes() <= limits.max_bytes + 1);
        }
    }
}

#[test]
fn stops_storing_head_output_once_the_window_is_full() {
    let mut buffer = OutputBuffer::new(head(2, 1000));
    for index in 0..1000 {
        buffer.push_str(&format!("line {index}\n"));
    }
    assert!(buffer.stored_bytes() < 20);
    assert_eq!(buffer.snapshot(), output("line 0\nline 1\n", 8876, 998));
}

#[test]
fn stores_at_most_one_byte_past_the_byte_limit() {
    let mut buffer = OutputBuffer::new(head(1000, 10));
    buffer.push_str(&"x".repeat(100));
    assert_eq!(buffer.stored_bytes(), 11);
    assert_eq!(buffer.snapshot(), output(&"x".repeat(10), 90, 0));

    let mut buffer = OutputBuffer::new(tail(3, 100));
    let mut stream = String::new();
    for index in 0..2000 {
        let chunk = format!("line {index}\n\n");
        stream.push_str(&chunk);
        buffer.push_str(&chunk);
        let snapshot = buffer.snapshot();
        assert!(buffer.stored_bytes() <= 101);
        assert_eq!(snapshot.text, bound_output(&stream, &tail(3, 100)).text);
    }
}

// --- skipped output ---

// Decoded shell output: ASCII, multi-byte and astral characters, replacement characters,
// CR/LF, tabs, and control characters that sanitizing removes after bounding.
const SHELL_ALPHABET: [&str; 15] = [
    "a", "b", "z", " ", "\n", "\n", "\n", "\r\n", "\t", "é", "€", "😀", "\u{fffd}", "\x01", "\x1b",
];

fn random_chunk(next: &mut impl FnMut() -> f64) -> String {
    let length = (next() * 12.0) as usize;
    (0..length).map(|_| pick(next, &SHELL_ALPHABET)).collect()
}

fn random_tail_limits(next: &mut impl FnMut() -> f64) -> OutputLimits {
    tail(1 + (next() * 5.0) as usize, 1 + (next() * 40.0) as usize)
}

/// Whether `text` proves that everything before it is outside the tail window.
fn exceeds_window(text: &str, limits: &OutputLimits) -> bool {
    let OutputSkip { bytes, newlines, .. } = measure(text);
    bytes > limits.max_bytes || newlines > limits.max_lines
}

// Progress commits snapshot at arbitrary moments; snapshots must never change the window.
#[test]
fn keeps_the_same_tail_whenever_progress_snapshots_happen() {
    for seed in 1..=3000 {
        let mut next = mulberry32(seed);
        let limits = random_tail_limits(&mut next);
        let mut plain = OutputBuffer::new(limits);
        let mut sampled = OutputBuffer::new(limits);
        let chunks = (next() * 30.0) as usize;
        for _ in 0..chunks {
            let chunk = random_chunk(&mut next);
            plain.push_str(&chunk);
            sampled.push_str(&chunk);
            if next() < 0.4 {
                sampled.snapshot();
            }
        }
        assert_eq!(sampled.snapshot(), plain.snapshot(), "seed {seed}");
    }
}

/// Feed random output to one buffer in full and to another through a reference skipper:
/// it holds undelivered text, and when it flushes, it may replace a prefix of it by counts
/// if the rest exceeds the window. Snapshots must agree whenever both buffers have seen the
/// same output.
fn check_skip_seed(seed: u32) {
    let mut next = mulberry32(seed);
    let limits = random_tail_limits(&mut next);
    let mut full = OutputBuffer::new(limits);
    let mut skipping = OutputBuffer::new(limits);
    let mut pending = String::new();
    let mut flush = |pending: &mut String, next: &mut dyn FnMut() -> f64, full: &mut OutputBuffer| {
        if pending.is_empty() {
            return;
        }
        let cuts: Vec<usize> = pending
            .char_indices()
            .map(|(index, _)| index)
            .chain([pending.len()])
            .filter(|&cut| cut > 0 && exceeds_window(&pending[cut..], &limits))
            .collect();
        if !cuts.is_empty() && next() < 0.7 {
            let cut = cuts[(next() * cuts.len() as f64) as usize];
            skipping.skip(measure(&pending[..cut]));
            skipping.push_str(&pending[cut..]);
        } else {
            skipping.push_str(pending);
        }
        pending.clear();
        if next() < 0.5 {
            skipping.snapshot();
        }
        assert_eq!(skipping.snapshot(), full.snapshot(), "seed {seed}");
    };
    let chunks = (next() * 40.0) as usize;
    for _ in 0..chunks {
        let chunk = random_chunk(&mut next);
        full.push_str(&chunk);
        if next() < 0.3 {
            full.snapshot();
        }
        pending.push_str(&chunk);
        if next() < 0.3 {
            flush(&mut pending, &mut next, &mut full);
        }
    }
    flush(&mut pending, &mut next, &mut full);
    full.end();
    skipping.end();
    assert_eq!(skipping.snapshot(), full.snapshot(), "seed {seed}");
}

#[test]
fn matches_the_full_stream_for_every_legal_skip_pattern() {
    for seed in 1..=3000 {
        check_skip_seed(seed);
    }
}

#[test]
fn counts_skipped_bytes_lines_and_the_final_newline_exactly() {
    let limits = tail(2, 1000);
    let mut full = OutputBuffer::new(limits);
    let mut skipping = OutputBuffer::new(limits);
    // The skipped text ends without a newline, so its last line continues in the delivered text.
    let omitted = "one\ntwo\nthr";
    let kept = "ee\nfour\nfive\nsix";
    full.push_str(&format!("{omitted}{kept}"));
    skipping.skip(measure(omitted));
    skipping.push_str(kept);
    assert_eq!(skipping.snapshot(), full.snapshot());
    assert_eq!(skipping.snapshot(), output("five\nsix", 19, 4));
}

#[test]
fn a_skip_flushes_an_incomplete_character_first() {
    let mut buffer = OutputBuffer::new(tail(1, 1000));
    buffer.push_bytes(&"€".as_bytes()[..1]);
    buffer.skip(measure("x\n"));
    buffer.push_str("a\nb\n");
    // U+FFFD (3 bytes) and the skipped "x\n" make the first line; "a\n" the second.
    assert_eq!(buffer.snapshot(), output("b\n", 7, 2));
}

#[test]
#[should_panic(expected = "tail retention")]
fn refuses_skips_for_head_retention() {
    let mut buffer = OutputBuffer::new(head(2, 10));
    buffer.skip(OutputSkip {
        bytes: 3,
        newlines: 1,
        ends_with_newline: true,
    });
}

// --- Progress ---

/// Commit times in milliseconds since `base`, and the size the next commit reports.
#[derive(Clone)]
struct Recorder {
    base: Instant,
    commits: Arc<Mutex<Vec<u64>>>,
    size: Arc<AtomicUsize>,
}

impl Recorder {
    fn new(size: usize) -> Self {
        Self {
            base: Instant::now(),
            commits: Arc::default(),
            size: Arc::new(AtomicUsize::new(size)),
        }
    }

    fn spawn(&self, min_interval_ms: u64) -> Progress<Infallible> {
        let recorder = self.clone();
        Progress::spawn(
            move || {
                let recorder = recorder.clone();
                async move {
                    let elapsed = recorder.base.elapsed().as_millis() as u64;
                    recorder.commits.lock().unwrap().push(elapsed);
                    Ok(recorder.size.load(Ordering::SeqCst))
                }
            },
            Duration::from_millis(min_interval_ms),
        )
    }

    fn commits(&self) -> Vec<u64> {
        self.commits.lock().unwrap().clone()
    }
}

/// Let the commit task run without moving the clock.
async fn settle() {
    for _ in 0..16 {
        tokio::task::yield_now().await;
    }
}

async fn advance(ms: u64) {
    tokio::time::advance(Duration::from_millis(ms)).await;
    settle().await;
}

#[tokio::test(start_paused = true)]
async fn commits_the_first_change_at_once_then_waits_the_minimum_and_the_written_size() {
    assert_eq!(PROGRESS_BYTES_PER_SECOND, 100 * 1024);
    let recorder = Recorder::new(50 * 1024);
    let progress = recorder.spawn(100);
    progress.mark();
    settle().await;
    assert_eq!(recorder.commits(), [0]);
    // 50 KiB buys 500 ms; changes meanwhile coalesce into one commit.
    recorder.size.store(10, Ordering::SeqCst);
    progress.mark();
    progress.mark();
    advance(499).await;
    assert_eq!(recorder.commits(), [0]);
    advance(1).await;
    assert_eq!(recorder.commits(), [0, 500]);
    // A small commit still waits the minimum 100 ms.
    progress.mark();
    advance(99).await;
    assert_eq!(recorder.commits(), [0, 500]);
    advance(1).await;
    assert_eq!(recorder.commits(), [0, 500, 600]);
    // After an idle period, the next change commits at once.
    advance(1000).await;
    progress.mark();
    settle().await;
    assert_eq!(recorder.commits(), [0, 500, 600, 1600]);
}

#[tokio::test(start_paused = true)]
async fn waits_the_configured_minimum_interval_between_small_commits() {
    let recorder = Recorder::new(10);
    let progress = recorder.spawn(500);
    progress.mark();
    settle().await;
    progress.mark();
    advance(499).await;
    assert_eq!(recorder.commits(), [0]);
    advance(1).await;
    assert_eq!(recorder.commits(), [0, 500]);
}

#[tokio::test(start_paused = true)]
async fn measures_the_pause_from_the_start_of_the_last_commit() {
    let recorder = Recorder::new(10);
    let progress = recorder.spawn(100);
    progress.mark_and_wait().await.unwrap();
    assert_eq!(recorder.commits(), [0]);
    advance(40).await;
    let second = progress.mark_and_wait();
    advance(59).await;
    assert_eq!(recorder.commits(), [0]);
    advance(1).await;
    second.await.unwrap();
    assert_eq!(recorder.commits(), [0, 100]);
}

#[tokio::test(start_paused = true)]
async fn resolves_waiters_with_the_commit_that_includes_their_mark() {
    let release = Arc::new(Notify::new());
    let commits = Arc::new(AtomicUsize::new(0));
    let progress = Progress::spawn(
        {
            let release = Arc::clone(&release);
            let commits = Arc::clone(&commits);
            move || {
                let release = Arc::clone(&release);
                let commits = Arc::clone(&commits);
                async move {
                    release.notified().await;
                    commits.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, Infallible>(10)
                }
            }
        },
        Duration::from_millis(100),
    );
    let first = tokio::spawn(progress.mark_and_wait());
    settle().await;
    // Made while the first commit is in flight: covered by the next one.
    let second = tokio::spawn(progress.mark_and_wait());
    release.notify_one();
    settle().await;
    assert!(first.is_finished());
    assert!(!second.is_finished());
    first.await.unwrap().unwrap();
    release.notify_one();
    advance(100).await;
    second.await.unwrap().unwrap();
    assert_eq!(commits.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn rejects_the_waiters_of_a_failed_commit_and_reports_its_error() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let progress = Progress::spawn(
        {
            let attempts = Arc::clone(&attempts);
            move || {
                let attempt = attempts.fetch_add(1, Ordering::SeqCst);
                async move { Err::<usize, _>(format!("commit {attempt} failed")) }
            }
        },
        Duration::from_millis(100),
    );
    let mut errors = progress.errors();
    let error = progress.mark_and_wait().await.unwrap_err();
    assert_eq!(*error, "commit 0 failed");
    let reported = errors.next().await.unwrap();
    assert!(Arc::ptr_eq(&error, &reported));
    // A failed commit still buys the minimum interval.
    let base = Instant::now();
    let error = progress.mark_and_wait().await.unwrap_err();
    assert_eq!(*error, "commit 1 failed");
    assert_eq!(base.elapsed(), Duration::from_millis(100));
    assert_eq!(*errors.next().await.unwrap(), "commit 1 failed");
    let pending = progress.stop().await;
    assert!(pending.is_empty());
    assert!(errors.next().await.is_none());
}

#[tokio::test(start_paused = true)]
async fn stop_waits_for_the_commit_in_flight_and_hands_back_waiters_no_commit_covered() {
    let release = Arc::new(Notify::new());
    let progress = Progress::spawn(
        {
            let release = Arc::clone(&release);
            move || {
                let release = Arc::clone(&release);
                async move {
                    release.notified().await;
                    Ok::<_, Infallible>(0)
                }
            }
        },
        Duration::from_millis(100),
    );
    let first = progress.mark_and_wait();
    settle().await;
    let second = progress.mark_and_wait();
    let stopping = tokio::spawn(progress.stop());
    settle().await;
    assert!(!stopping.is_finished());
    release.notify_one();
    let pending = stopping.await.unwrap();
    first.await.unwrap();
    assert_eq!(pending.len(), 1);
    pending.settle(Ok(()));
    second.await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn stop_disarms_the_scheduled_commit() {
    let recorder = Recorder::new(10);
    let progress = recorder.spawn(100);
    progress.mark();
    settle().await;
    let waiter = progress.mark_and_wait();
    advance(50).await;
    let pending = progress.stop().await;
    advance(1000).await;
    assert_eq!(recorder.commits(), [0]);
    assert_eq!(pending.len(), 1);
    pending.settle(Ok(()));
    waiter.await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn dropping_progress_ends_the_commit_task() {
    let recorder = Recorder::new(10);
    let progress = recorder.spawn(100);
    progress.mark();
    settle().await;
    progress.mark();
    let errors = progress.errors();
    drop(progress);
    advance(1000).await;
    assert_eq!(recorder.commits(), [0]);
    // The task released the shared state: the error stream ended.
    assert_eq!(errors.collect::<Vec<_>>().await.len(), 0);
}
