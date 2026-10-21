//! BM25 tool search for `searchTools()`: upstream's `search.ts`, the coding agent's tool search
//! ranking.

use std::collections::{HashMap, HashSet};

/// One searchable document: a tool's name and the text it is found by.
pub(super) struct SearchDocument {
    pub(super) name: String,
    pub(super) text: String,
}

const STOP_WORDS: [&str; 21] = [
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "with",
];

/// Naive singular form, so `issues` matches `issue` and `searches` matches `search`.
fn stem(term: &str) -> String {
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
pub(super) fn tokenize(text: &str) -> Vec<String> {
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
        .map(stem)
        .collect()
}

/// Okapi BM25 (k1 1.2, b 0.75): the names of the best matches, best first; ties keep document
/// order.
pub(super) fn rank_documents(query: &str, documents: &[SearchDocument], limit: usize) -> Vec<String> {
    const K1: f64 = 1.2;
    const B: f64 = 0.75;
    let mut seen = HashSet::new();
    let query_terms: Vec<String> = tokenize(query)
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
            for term in tokenize(&document.text) {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
