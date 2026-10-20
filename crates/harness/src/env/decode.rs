//! Streaming UTF-8 decoding that matches decoding the whole input at once.
//!
//! Invalid sequences become U+FFFD per maximal subpart (as [`String::from_utf8_lossy`] and the
//! WHATWG decoder), and a character split across chunks survives. Environments use it for
//! command output, text line readers and line scans.

const REPLACEMENT: &str = "\u{FFFD}";
const BYTE_ORDER_MARK: char = '\u{FEFF}';

/// A UTF-8 decoder for one stream of bytes fed in pieces.
#[derive(Debug, Default)]
pub(crate) struct Utf8Decoder {
    /// Bytes of an incomplete character at the end of the last piece.
    pending: Vec<u8>,
    /// Drop a byte-order mark at the very start of the stream.
    strip_bom: bool,
    /// Whether text has been emitted; a byte-order mark is dropped only before that.
    started: bool,
}

impl Utf8Decoder {
    /// A decoder that keeps every character, for a byte range inside a stream.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// A decoder for a whole stream: a leading byte-order mark is dropped, as when decoding
    /// a file or a pipe as text.
    pub(crate) fn stripping_bom() -> Self {
        Self {
            strip_bom: true,
            ..Self::default()
        }
    }

    /// Decode `bytes`, handing the text to `sink` in pieces; an incomplete character at the end
    /// waits for the next call.
    pub(crate) fn decode(&mut self, bytes: &[u8], sink: &mut impl FnMut(&str)) {
        let rest = self.complete_pending(bytes, sink);
        let mut pieces = rest.utf8_chunks().peekable();
        while let Some(piece) = pieces.next() {
            self.emit(piece.valid(), sink);
            let invalid = piece.invalid();
            if invalid.is_empty() {
                continue;
            }
            if pieces.peek().is_none() && is_incomplete(invalid) {
                self.pending.extend_from_slice(invalid);
            } else {
                self.emit(REPLACEMENT, sink);
            }
        }
    }

    /// End the stream: an incomplete trailing character becomes U+FFFD.
    pub(crate) fn finish(&mut self, sink: &mut impl FnMut(&str)) {
        if !self.pending.is_empty() {
            self.pending.clear();
            self.emit(REPLACEMENT, sink);
        }
    }

    /// [`decode`](Self::decode) into a new string.
    pub(crate) fn decode_to_string(&mut self, bytes: &[u8]) -> String {
        let mut text = String::new();
        self.decode(bytes, &mut |piece| text.push_str(piece));
        text
    }

    /// [`finish`](Self::finish) into a new string.
    pub(crate) fn finish_to_string(&mut self) -> String {
        let mut text = String::new();
        self.finish(&mut |piece| text.push_str(piece));
        text
    }

    /// Complete the pending character with the first bytes of `bytes`; returns the rest.
    fn complete_pending<'b>(&mut self, bytes: &'b [u8], sink: &mut impl FnMut(&str)) -> &'b [u8] {
        if self.pending.is_empty() {
            return bytes;
        }
        let mut pending = std::mem::take(&mut self.pending);
        let mut rest = bytes;
        loop {
            let Some((&byte, after)) = rest.split_first() else {
                // Still incomplete.
                self.pending = pending;
                return rest;
            };
            pending.push(byte);
            match std::str::from_utf8(&pending) {
                Ok(character) => {
                    self.emit(character, sink);
                    rest = after;
                    break;
                }
                Err(error) if error.error_len().is_none() => rest = after,
                Err(_) => {
                    // The byte does not continue the character; it starts the rest.
                    self.emit(REPLACEMENT, sink);
                    break;
                }
            }
        }
        pending.clear();
        self.pending = pending;
        rest
    }

    fn emit(&mut self, text: &str, sink: &mut impl FnMut(&str)) {
        if text.is_empty() {
            return;
        }
        let text = if self.started || !self.strip_bom {
            text
        } else {
            text.strip_prefix(BYTE_ORDER_MARK).unwrap_or(text)
        };
        self.started = true;
        if !text.is_empty() {
            sink(text);
        }
    }
}

/// Whether `bytes` are the start of a character that more bytes could complete.
fn is_incomplete(bytes: &[u8]) -> bool {
    std::str::from_utf8(bytes).is_err_and(|error| error.error_len().is_none())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::Utf8Decoder;

    /// A small deterministic generator (SplitMix64) for randomized comparisons.
    pub(crate) struct Random(u64);

    impl Random {
        pub(crate) fn new(seed: u64) -> Self {
            Self(seed)
        }

        /// A value in `0..bound`.
        pub(crate) fn below(&mut self, bound: usize) -> usize {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            (z % bound as u64) as usize
        }
    }

    /// Newlines, ASCII, a byte-order mark, valid multi-byte sequences, and bytes that form
    /// invalid or truncated sequences.
    const PIECES: &[&[u8]] = &[
        b"\n",
        b"\n",
        b"a",
        b"bc",
        &[0xef, 0xbb, 0xbf],
        &[0xc3, 0xa9],
        &[0xe2, 0x82, 0xac],
        &[0xf0, 0x9f, 0x98, 0x80],
        &[0xe2, 0x82],
        &[0xff],
        &[0x80],
        &[0xf0, 0x9f],
        b"\r\n",
    ];

    pub(crate) fn random_bytes(random: &mut Random) -> Vec<u8> {
        let pieces = random.below(60);
        (0..pieces)
            .flat_map(|_| PIECES[random.below(PIECES.len())].iter().copied())
            .collect()
    }

    /// Decoding the whole input at once, as the decoders must agree with.
    pub(crate) fn whole(bytes: &[u8], strip_bom: bool) -> String {
        let text = String::from_utf8_lossy(bytes);
        match text.strip_prefix('\u{FEFF}') {
            Some(rest) if strip_bom => rest.to_owned(),
            _ => text.into_owned(),
        }
    }

    #[test]
    fn keeps_a_byte_order_mark_that_does_not_start_the_stream() {
        let bytes = [0xe2, 0x82, 0xef, 0xbb, 0xbf, 0x61];
        let mut decoder = Utf8Decoder::stripping_bom();
        let mut text: String = bytes.iter().map(|byte| decoder.decode_to_string(&[*byte])).collect();
        text.push_str(&decoder.finish_to_string());
        assert_eq!(text, "\u{FFFD}\u{FEFF}a");
    }

    #[test]
    fn decodes_any_chunking_like_the_whole_stream() {
        for seed in 1..=5000 {
            let mut random = Random::new(seed);
            let bytes = random_bytes(&mut random);
            for strip_bom in [false, true] {
                let mut decoder = if strip_bom {
                    Utf8Decoder::stripping_bom()
                } else {
                    Utf8Decoder::new()
                };
                let mut text = String::new();
                let mut offset = 0;
                while offset < bytes.len() {
                    let end = (offset + 1 + random.below(7)).min(bytes.len());
                    text.push_str(&decoder.decode_to_string(&bytes[offset..end]));
                    offset = end;
                }
                text.push_str(&decoder.finish_to_string());
                assert_eq!(text, whole(&bytes, strip_bom), "seed {seed}");
            }
        }
    }
}
