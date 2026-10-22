//! BM25 tool search, shared by the `tool_search` tool and code mode's `searchTools()`:
//! upstream's `search.ts`, the coding agent's tool search ranking.

/// One searchable document: a tool's name and the text it is found by.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchDocument {
    pub name: String,
    pub text: String,
}

/// Whether `term` is one of the stop words upstream drops.
fn is_stop_word(term: &str) -> bool {
    matches!(
        term,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "by"
            | "for"
            | "from"
            | "in"
            | "is"
            | "it"
            | "of"
            | "on"
            | "or"
            | "that"
            | "the"
            | "this"
            | "to"
            | "with"
    )
}

/// Naive singular form, so `issues` matches `issue` and `searches` matches `search`; in place.
fn stem(term: &mut String) {
    let long = term.len();
    if long > 4 && term.ends_with("ies") {
        term.truncate(long - 3);
        term.push('y');
    } else if long > 4
        && ["ches", "shes", "sses", "xes", "zes"]
            .iter()
            .any(|end| term.ends_with(end))
    {
        term.truncate(long - 2);
    } else if long > 3 && term.ends_with('s') && !term.ends_with("ss") {
        term.truncate(long - 1);
    }
}

/// Hands the term in `term` to `f` when it is not a stop word, stemmed, and empties it.
fn flush(term: &mut String, f: &mut impl FnMut(&str)) {
    if !term.is_empty() && !is_stop_word(term) {
        stem(term);
        f(term);
    }
    term.clear();
}

/// Calls `f` with each term of `text`, as [`tokenize`] returns them, in one pass that builds
/// each term in `term` instead of allocating it.
///
/// A word starts at an ASCII capital after an ASCII lowercase letter or digit (`aB`, `1B`), and
/// a run of capitals ends before a capital followed by a lowercase letter (`ABc`). Characters
/// are lowercased as `str::to_lowercase` does (and JavaScript's `toLowerCase`), so a non-ASCII
/// character that lowercases to ASCII, such as the Kelvin sign, joins its word; every other
/// character ends it.
fn for_each_term(text: &str, term: &mut String, mut f: impl FnMut(&str)) {
    /// What the previous character was, as far as word starts depend on it.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Previous {
        LowerOrDigit,
        Upper,
        Other,
    }
    term.clear();
    let bytes = text.as_bytes();
    let mut index = 0;
    let mut previous = Previous::Other;
    while let Some(&byte) = bytes.get(index) {
        if byte.is_ascii() {
            index += 1;
            if byte.is_ascii_uppercase() {
                let ends_capitals = previous == Previous::Upper && bytes.get(index).is_some_and(u8::is_ascii_lowercase);
                if previous == Previous::LowerOrDigit || ends_capitals {
                    flush(term, &mut f);
                }
                term.push(char::from(byte.to_ascii_lowercase()));
                previous = Previous::Upper;
            } else if byte.is_ascii_lowercase() || byte.is_ascii_digit() {
                term.push(char::from(byte));
                previous = Previous::LowerOrDigit;
            } else {
                flush(term, &mut f);
                previous = Previous::Other;
            }
            continue;
        }
        // `index` is always at a character boundary: ASCII advances one byte, the rest one
        // character.
        let Some(c) = text[index..].chars().next() else {
            break;
        };
        index += c.len_utf8();
        for lower in c.to_lowercase() {
            if lower.is_ascii_lowercase() || lower.is_ascii_digit() {
                term.push(lower);
            } else {
                flush(term, &mut f);
            }
        }
        previous = Previous::Other;
    }
    flush(term, &mut f);
}

/// Lowercase terms, split at camelCase boundaries and non-alphanumerics, without stop words.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for_each_term(text, &mut String::new(), |term| terms.push(term.to_owned()));
    terms
}

/// Okapi BM25 (k1 1.2, b 0.75): the names of the best matches, best first; ties keep document
/// order.
///
/// Scoring needs only each document's length and its counts of the query terms, so one pass
/// per document counts those, with no table of its other terms.
pub fn rank_documents(query: &str, documents: &[SearchDocument], limit: usize) -> Vec<String> {
    const K1: f64 = 1.2;
    const B: f64 = 0.75;
    let mut query_terms: Vec<String> = Vec::new();
    for term in tokenize(query) {
        if !query_terms.contains(&term) {
            query_terms.push(term);
        }
    }
    if query_terms.is_empty() || documents.is_empty() || limit == 0 {
        return Vec::new();
    }
    let width = query_terms.len();
    // `counts[document * width + i]`: how often query term `i` occurs in the document.
    let mut counts = vec![0_u32; documents.len() * width];
    let mut lengths: Vec<f64> = Vec::with_capacity(documents.len());
    let mut term = String::new();
    for (index, document) in documents.iter().enumerate() {
        let own = &mut counts[index * width..(index + 1) * width];
        let mut length = 0_u64;
        for_each_term(&document.text, &mut term, |found| {
            length += 1;
            if let Some(position) = query_terms.iter().position(|query| query == found) {
                own[position] += 1;
            }
        });
        #[allow(clippy::cast_precision_loss)] // Term counts are small.
        lengths.push(length as f64);
    }
    #[allow(clippy::cast_precision_loss)]
    let total = documents.len() as f64;
    let average = lengths.iter().sum::<f64>() / total;
    let average = if average == 0.0 { 1.0 } else { average };
    let idf: Vec<f64> = (0..width)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let frequency = counts.chunks_exact(width).filter(|own| own[i] > 0).count() as f64;
            (1.0 + (total - frequency + 0.5) / (frequency + 0.5)).ln()
        })
        .collect();
    let mut matches: Vec<(usize, f64)> = Vec::new();
    for (index, own) in counts.chunks_exact(width).enumerate() {
        let mut score = 0.0;
        for (i, &count) in own.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let count = f64::from(count);
            let norm = K1 * (1.0 - B + B * lengths[index] / average);
            score += idf[i] * (count * (K1 + 1.0)) / (count + norm);
        }
        if score > 0.0 {
            matches.push((index, score));
        }
    }
    // A stable sort: ties keep document order.
    matches.sort_by(|left, right| right.1.total_cmp(&left.1));
    matches
        .into_iter()
        .take(limit)
        .map(|(index, _)| documents[index].name.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;

    const STOP_WORDS: [&str; 21] = [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that",
        "the", "this", "to", "with",
    ];

    // The implementation before the streaming tokenizer, a line-by-line port of upstream's
    // `search.ts`, kept as the reference the faster one must match.
    /// Naive singular form, so `issues` matches `issue` and `searches` matches `search`.
    fn reference_stem(term: &str) -> String {
        let long = term.len();
        if long > 4
            && let Some(base) = term.strip_suffix("ies")
        {
            return format!("{base}y");
        }
        if long > 4
            && ["ches", "shes", "sses", "xes", "zes"]
                .iter()
                .any(|end| term.ends_with(end))
        {
            return term[..long - 2].to_owned();
        }
        if long > 3 && term.ends_with('s') && !term.ends_with("ss") {
            return term[..long - 1].to_owned();
        }
        term.to_owned()
    }

    /// Lowercase terms, split at camelCase boundaries and non-alphanumerics, without stop words.
    fn reference_tokenize(text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut spaced = String::with_capacity(text.len() + 8);
        for (index, &c) in chars.iter().enumerate() {
            let previous = index.checked_sub(1).map(|index| chars[index]);
            let next = chars.get(index + 1).copied();
            if c.is_ascii_uppercase()
                && let Some(previous) = previous
            {
                // `aB` and `1B`: a word starts at B. `ABc`: the run of capitals ends before B.
                let after_lower = previous.is_ascii_lowercase() || previous.is_ascii_digit();
                let ends_capitals = previous.is_ascii_uppercase() && next.is_some_and(|next| next.is_ascii_lowercase());
                if after_lower || ends_capitals {
                    spaced.push(' ');
                }
            }
            spaced.push(c);
        }
        spaced
            .to_lowercase()
            .split(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit()))
            .filter(|term| !term.is_empty() && !STOP_WORDS.contains(term))
            .map(reference_stem)
            .collect()
    }

    /// Okapi BM25 (k1 1.2, b 0.75): the names of the best matches, best first; ties keep document
    /// order.
    fn reference_rank(query: &str, documents: &[SearchDocument], limit: usize) -> Vec<String> {
        const K1: f64 = 1.2;
        const B: f64 = 0.75;
        let mut seen = HashSet::new();
        let query_terms: Vec<String> = reference_tokenize(query)
            .into_iter()
            .filter(|term| seen.insert(term.clone()))
            .collect();
        if query_terms.is_empty() || documents.is_empty() || limit == 0 {
            return Vec::new();
        }
        let term_counts: Vec<HashMap<String, usize>> = documents
            .iter()
            .map(|document| {
                let mut counts = HashMap::new();
                for term in reference_tokenize(&document.text) {
                    *counts.entry(term).or_insert(0) += 1;
                }
                counts
            })
            .collect();
        #[allow(clippy::cast_precision_loss)] // Term counts are small.
        let lengths: Vec<f64> = term_counts
            .iter()
            .map(|counts| counts.values().sum::<usize>() as f64)
            .collect();
        #[allow(clippy::cast_precision_loss)]
        let total = documents.len() as f64;
        let average = lengths.iter().sum::<f64>() / total;
        let average = if average == 0.0 { 1.0 } else { average };
        let idf: HashMap<&str, f64> = query_terms
            .iter()
            .map(|term| {
                #[allow(clippy::cast_precision_loss)]
                let frequency = term_counts.iter().filter(|counts| counts.contains_key(term)).count() as f64;
                (
                    term.as_str(),
                    (1.0 + (total - frequency + 0.5) / (frequency + 0.5)).ln(),
                )
            })
            .collect();
        let mut matches: Vec<(usize, f64)> = Vec::new();
        for (index, counts) in term_counts.iter().enumerate() {
            let mut score = 0.0;
            for term in &query_terms {
                let Some(&count) = counts.get(term) else { continue };
                #[allow(clippy::cast_precision_loss)]
                let count = count as f64;
                let norm = K1 * (1.0 - B + B * lengths[index] / average);
                score += idf[term.as_str()] * (count * (K1 + 1.0)) / (count + norm);
            }
            if score > 0.0 {
                matches.push((index, score));
            }
        }
        // A stable sort: ties keep document order.
        matches.sort_by(|left, right| right.1.total_cmp(&left.1));
        matches
            .into_iter()
            .take(limit)
            .map(|(index, _)| documents[index].name.clone())
            .collect()
    }

    fn document(name: &str, text: &str) -> SearchDocument {
        SearchDocument {
            name: name.to_owned(),
            text: text.to_owned(),
        }
    }

    #[test]
    fn terms_split_at_case_and_punctuation_and_lose_plurals() {
        assert_eq!(
            tokenize("listIssues HTTPServer for the gh_search-queries boxes"),
            ["list", "issue", "http", "server", "gh", "search", "query", "box"]
        );
        assert_eq!(tokenize("class glass is"), ["class", "glass"]);
    }

    #[test]
    fn the_best_matches_come_first() {
        let documents = [
            document("read", "read Reads a file"),
            document("issue_read", "issue read Reads one GitHub issue github"),
            document("issue_list", "issue list Lists GitHub issues github issues"),
            document("bash", "bash Runs a command"),
        ];
        assert_eq!(
            rank_documents("github issues", &documents, 8),
            ["issue_list", "issue_read"]
        );
        assert_eq!(rank_documents("read", &documents, 1), ["read"]);
        assert!(rank_documents("the", &documents, 8).is_empty());
        assert!(rank_documents("read", &documents, 0).is_empty());
    }

    /// A deterministic corpus of texts mixing the cases tokenizing distinguishes.
    fn corpus() -> Vec<String> {
        const PARTS: [&str; 34] = [
            "listIssues",
            "HTTPServer",
            "getURLForID",
            "gh_search-queries",
            "boxes",
            "classes",
            "glass",
            "the",
            "of",
            "a",
            "IS",
            "queries",
            "watches",
            "dishes",
            "buzzes",
            "taxes",
            "bus",
            "ss",
            "v2Api",
            "ABCdef",
            "x1Y2z3",
            "--",
            "  ",
            "\u{212A}elvin",
            "\u{130}stanbul",
            "Straße",
            "naïve",
            "\u{1F600}emoji",
            "ΣΑΣ",
            "repo/owner#42",
            "PR's",
            "Ünïcode",
            "aBcDeF",
            "ID",
        ];
        let mut seed = 0x2545_F491_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            usize::try_from(seed % 1_000_003).unwrap()
        };
        (0..300)
            .map(|_| {
                let words = next() % 12;
                (0..words)
                    .map(|_| PARTS[next() % PARTS.len()])
                    .collect::<Vec<_>>()
                    .join(if next() % 3 == 0 { "" } else { " " })
            })
            .collect()
    }

    #[test]
    fn the_streaming_tokenizer_matches_the_reference() {
        for text in corpus() {
            assert_eq!(tokenize(&text), reference_tokenize(&text), "{text:?}");
        }
    }

    #[test]
    fn ranking_matches_the_reference() {
        let documents: Vec<SearchDocument> = corpus()
            .into_iter()
            .enumerate()
            .map(|(index, text)| document(&format!("tool{index}"), &text))
            .collect();
        let queries = [
            "list issues",
            "http server url",
            "box class glass",
            "kelvin istanbul strasse",
            "the of a",
            "",
            "query query queries",
            "v2 api id",
        ];
        // The corpus gives the comparison real rankings, with ties.
        assert!(rank_documents("list issues", &documents, 1000).len() > 10);
        for query in queries {
            for limit in [0, 1, 8, 1000] {
                assert_eq!(
                    rank_documents(query, &documents, limit),
                    reference_rank(query, &documents, limit),
                    "{query:?} {limit}"
                );
            }
        }
        let empty = [document("blank", ""), document("stops", "the of a")];
        assert!(rank_documents("anything", &empty, 8).is_empty());
        assert_eq!(
            rank_documents("anything", &empty, 8),
            reference_rank("anything", &empty, 8)
        );
    }
}
