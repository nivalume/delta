//! Where lines of a file are: [`LineScan`], and the [`LineScanner`] that computes it from the
//! file's bytes fed in order, so an environment scans any size in bounded memory.

use super::decode::Utf8Decoder;

const NEWLINE: u8 = b'\n';
const BYTE_ORDER_MARK: [u8; 3] = [0xef, 0xbb, 0xbf];

/// Where lines `[start_line, end_line)` of a file are, as
/// [`BinaryReader::scan_lines`](super::BinaryReader::scan_lines) found them. Line `k` starts
/// after the `k`-th newline byte. Offsets are bytes of the file.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LineScan {
    /// Newline bytes in the whole file; it has `newlines + 1` lines.
    pub newlines: u64,
    /// Start of the first selected line. A selection past the last line is empty at the end of
    /// the file.
    pub start: u64,
    /// End of the last selected line, without the newline that ends it.
    pub end: u64,
    /// Where the first selected line ends: its newline, or the end of the file.
    pub first_line_end: u64,
    /// Where the last selected line starts.
    pub last_line_start: u64,
    /// UTF-8 byte length of the selection as decoding the whole file yields it: a leading
    /// byte-order mark is not counted, an invalid sequence counts as U+FFFD (3 bytes).
    pub selected_bytes: u64,
    /// The same for the first selected line.
    pub first_line_bytes: u64,
}

/// Computes a [`LineScan`] from a file's bytes fed in order, in any pieces.
#[derive(Debug)]
pub(crate) struct LineScanner {
    start_line: u64,
    /// `None`: to the end of the file.
    end_line: Option<u64>,
    position: u64,
    newlines: u64,
    line_start: u64,
    start: Option<u64>,
    end: Option<u64>,
    first_line_end: Option<u64>,
    last_line_start: Option<u64>,
    selected_bytes: u64,
    first_line_bytes: u64,
    /// Counts decoded bytes of the selection while it is open.
    selection: Option<Utf8Decoder>,
    /// Counts decoded bytes of the first selected line while it is open.
    first_line: Option<Utf8Decoder>,
    /// The first bytes, held until it is known whether they are a byte-order mark.
    head: Option<Vec<u8>>,
    bom: bool,
}

impl LineScanner {
    /// A scanner for lines `[start_line, end_line)`; `None` when the range is empty.
    pub(crate) fn new(start_line: u64, end_line: Option<u64>) -> Option<Self> {
        if end_line.is_some_and(|end_line| end_line <= start_line) {
            return None;
        }
        let mut scanner = Self {
            start_line,
            end_line,
            position: 0,
            newlines: 0,
            line_start: 0,
            start: None,
            end: None,
            first_line_end: None,
            last_line_start: None,
            selected_bytes: 0,
            first_line_bytes: 0,
            selection: None,
            first_line: None,
            head: Some(Vec::with_capacity(BYTE_ORDER_MARK.len())),
            bom: false,
        };
        if start_line == 0 {
            scanner.begin(0);
        }
        Some(scanner)
    }

    /// Feed the next bytes of the file.
    pub(crate) fn push(&mut self, mut chunk: &[u8]) {
        if let Some(head) = &mut self.head {
            let take = (BYTE_ORDER_MARK.len() - head.len()).min(chunk.len());
            head.extend_from_slice(&chunk[..take]);
            if head.len() < BYTE_ORDER_MARK.len() {
                return;
            }
            chunk = &chunk[take..];
            self.release_head();
        }
        self.process(chunk);
    }

    /// The scan of the bytes fed so far, as the whole file.
    pub(crate) fn finish(mut self) -> LineScan {
        self.release_head();
        let size = self.position;
        let Some(start) = self.start else {
            return LineScan {
                newlines: self.newlines,
                start: size,
                end: size,
                first_line_end: size,
                last_line_start: size,
                selected_bytes: 0,
                first_line_bytes: 0,
            };
        };
        if self.first_line_end.is_none() {
            self.end_first_line(size);
        }
        if self.end.is_none() {
            self.end_selection(size);
        }
        LineScan {
            newlines: self.newlines,
            start,
            end: self.end.unwrap_or(size),
            first_line_end: self.first_line_end.unwrap_or(size),
            // A selection that reaches past the last line ends with the last line.
            last_line_start: self.last_line_start.unwrap_or(self.line_start),
            selected_bytes: self.selected_bytes,
            first_line_bytes: self.first_line_bytes,
        }
    }

    fn release_head(&mut self) {
        let Some(head) = self.head.take() else { return };
        self.bom = head.starts_with(&BYTE_ORDER_MARK);
        self.process(&head);
    }

    /// Whether `line` is the last selected line.
    fn is_last_line(&self, line: u64) -> bool {
        self.end_line.is_some_and(|end_line| line + 1 == end_line)
    }

    fn process(&mut self, chunk: &[u8]) {
        let base = self.position;
        let mut from = 0;
        for index in memchr_all(chunk) {
            // The newline ends line `self.newlines`. It belongs to the selection only between
            // selected lines.
            self.feed(chunk, base, from, index);
            let line = self.newlines;
            let position = base + index as u64;
            if line == self.start_line {
                self.end_first_line(position);
            }
            if self.is_last_line(line) {
                self.end_selection(position);
            }
            self.feed(chunk, base, index, index + 1);
            from = index + 1;
            self.newlines += 1;
            self.line_start = position + 1;
            if self.newlines == self.start_line {
                self.begin(self.line_start);
            }
            if self.is_last_line(self.newlines) {
                self.last_line_start = Some(self.line_start);
            }
        }
        self.feed(chunk, base, from, chunk.len());
        self.position += chunk.len() as u64;
    }

    fn begin(&mut self, start: u64) {
        self.start = Some(start);
        if self.is_last_line(self.start_line) {
            self.last_line_start = Some(start);
        }
        self.selection = Some(Utf8Decoder::new());
        self.first_line = Some(Utf8Decoder::new());
    }

    fn end_first_line(&mut self, position: u64) {
        self.first_line_end = Some(position);
        if let Some(mut decoder) = self.first_line.take() {
            decoder.finish(&mut |text| self.first_line_bytes += text.len() as u64);
        }
    }

    fn end_selection(&mut self, position: u64) {
        self.end = Some(position);
        if let Some(mut decoder) = self.selection.take() {
            decoder.finish(&mut |text| self.selected_bytes += text.len() as u64);
        }
    }

    /// Feed `chunk[from..to]`, which starts at file offset `base`, to the decoders still open.
    fn feed(&mut self, chunk: &[u8], base: u64, mut from: usize, to: usize) {
        // Decoding the whole file drops a leading byte-order mark.
        let mark = BYTE_ORDER_MARK.len() as u64;
        if self.bom && base + (from as u64) < mark {
            from = to.min((mark - base) as usize);
        }
        if to <= from {
            return;
        }
        let bytes = &chunk[from..to];
        if let Some(decoder) = &mut self.selection {
            decoder.decode(bytes, &mut |text| self.selected_bytes += text.len() as u64);
        }
        if let Some(decoder) = &mut self.first_line {
            decoder.decode(bytes, &mut |text| self.first_line_bytes += text.len() as u64);
        }
    }
}

/// Positions of the newlines in `bytes`.
fn memchr_all(bytes: &[u8]) -> impl Iterator<Item = usize> + '_ {
    bytes
        .iter()
        .enumerate()
        .filter(|&(_, &byte)| byte == NEWLINE)
        .map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::LineScanner;
    use crate::env::decode::tests::{Random, random_bytes, whole};

    /// Decode `bytes[from..to]` as the conformance suite does: a leading mark of the file is
    /// dropped, one inside it kept.
    fn range(bytes: &[u8], from: u64, to: u64) -> String {
        whole(&bytes[from as usize..to as usize], from == 0)
    }

    #[test]
    fn agrees_with_decoding_and_splitting_the_whole_file() {
        for seed in 1..=5000 {
            let mut random = Random::new(seed);
            let file = random_bytes(&mut random);
            let text = whole(&file, true);
            let lines: Vec<&str> = text.split('\n').collect();
            let start_line = random.below(lines.len() + 2);
            let end_line = if random.below(10) < 3 {
                None
            } else {
                Some(start_line + 1 + random.below(lines.len() + 1))
            };
            let Some(mut scanner) = LineScanner::new(start_line as u64, end_line.map(|line| line as u64)) else {
                panic!("valid range");
            };
            let mut offset = 0;
            while offset < file.len() {
                let end = (offset + 1 + random.below(7)).min(file.len());
                scanner.push(&file[offset..end]);
                offset = end;
            }
            let scan = scanner.finish();
            let message = format!("seed {seed}");
            assert_eq!(scan.newlines as usize, lines.len() - 1, "{message}");
            let last = end_line.unwrap_or(lines.len()).min(lines.len());
            let selected = lines
                .get(start_line..last)
                .map(|lines| lines.join("\n"))
                .unwrap_or_default();
            assert_eq!(range(&file, scan.start, scan.end), selected, "{message}");
            assert_eq!(scan.selected_bytes as usize, selected.len(), "{message}");
            if start_line < lines.len() {
                assert_eq!(
                    range(&file, scan.start, scan.first_line_end),
                    lines[start_line],
                    "{message}"
                );
                assert_eq!(scan.first_line_bytes as usize, lines[start_line].len(), "{message}");
                let last_line = last - 1;
                let last_end = if last_line + 1 < lines.len() {
                    let from = scan.last_line_start as usize;
                    file[from..]
                        .iter()
                        .position(|&byte| byte == b'\n')
                        .map_or(file.len(), |index| from + index)
                } else {
                    file.len()
                };
                assert_eq!(
                    range(&file, scan.last_line_start, last_end as u64),
                    lines[last_line],
                    "{message}"
                );
            } else {
                let size = file.len() as u64;
                assert_eq!(
                    (scan.start, scan.end, scan.selected_bytes),
                    (size, size, 0),
                    "{message}"
                );
            }
        }
    }

    #[test]
    fn rejects_empty_ranges() {
        assert!(LineScanner::new(2, Some(2)).is_none());
        assert!(LineScanner::new(3, Some(1)).is_none());
        assert!(LineScanner::new(0, None).is_some());
    }
}
