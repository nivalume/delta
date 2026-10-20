//! The kernel modules never name an agent, surface or side module (README.md, "Architecture
//! and seams"). Scans the kernel's sources for `crate::` paths into an upper layer.

use std::fs;
use std::path::{Path, PathBuf};

/// The kernel modules, as `src/<name>/` or `src/<name>.rs`.
const KERNEL: [&str; 5] = ["records", "documents", "storage", "session", "scheduler"];

/// Modules of the agent, surface and side layers.
const UPPER: [&str; 13] = [
    "inbox",
    "context",
    "prompt",
    "generation",
    "tool",
    "approval",
    "compaction",
    "usage",
    "registry",
    "harness",
    "observe",
    "env",
    "tools",
];

fn rust_files(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let mut entries: Vec<PathBuf> = fs::read_dir(path)
            .expect("kernel directories are readable")
            .map(|entry| entry.expect("directory entries are readable").path())
            .collect();
        entries.sort();
        for entry in entries {
            rust_files(&entry, files);
        }
    } else if path.is_file() && path.extension().is_some_and(|extension| extension == "rs") {
        files.push(path.to_owned());
    }
}

/// The first segment of every path a `crate::` occurrence on `source` names, with its line:
/// `crate::a::b` gives `a`; `crate::{a, b::c}` gives `a` and `b`, also across lines.
fn crate_roots(source: &str) -> Vec<(usize, String)> {
    let mut roots = Vec::new();
    let mut rest = source;
    let mut offset = 0;
    while let Some(found) = rest.find("crate::") {
        let start = offset + found + "crate::".len();
        let line = source[..start].matches('\n').count() + 1;
        let after = source[start..].trim_start();
        if let Some(group) = after.strip_prefix('{') {
            let mut depth = 1;
            let mut item = String::new();
            let mut at_item_start = true;
            for c in group.chars() {
                match c {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    _ => {}
                }
                if depth == 0 {
                    break;
                }
                if depth == 1 && c == ',' {
                    if !item.is_empty() {
                        roots.push((line, std::mem::take(&mut item)));
                    }
                    at_item_start = true;
                    continue;
                }
                if at_item_start {
                    if c.is_alphanumeric() || c == '_' {
                        item.push(c);
                    } else if !item.is_empty() {
                        roots.push((line, std::mem::take(&mut item)));
                        at_item_start = false;
                    }
                }
            }
            if !item.is_empty() {
                roots.push((line, item));
            }
        } else {
            let name: String = after.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            roots.push((line, name));
        }
        offset = start;
        rest = &source[start..];
    }
    roots
}

#[test]
fn the_scanner_finds_plain_and_grouped_paths() {
    let source = "use crate::records::X;\nuse crate::{\n    storage,\n    inbox::Item,\n};\nlet x = crate :: tools;\n";
    let roots: Vec<String> = crate_roots(source).into_iter().map(|(_, root)| root).collect();
    assert_eq!(roots, ["records", "storage", "inbox"]);
    assert_eq!(crate_roots("crate::env::Local")[0], (1, "env".to_owned()));
}

#[test]
fn kernel_modules_do_not_name_upper_layers() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    for module in KERNEL {
        rust_files(&src.join(module), &mut files);
        rust_files(&src.join(format!("{module}.rs")), &mut files);
    }
    assert!(!files.is_empty(), "no kernel sources found under {}", src.display());
    let mut offending = Vec::new();
    for file in &files {
        let source = fs::read_to_string(file).expect("kernel sources are readable");
        let lines: Vec<&str> = source.lines().collect();
        for (line, root) in crate_roots(&source) {
            if UPPER.contains(&root.as_str()) {
                let text = lines.get(line - 1).map_or("", |text| text.trim());
                offending.push(format!("{}:{line}: crate::{root} in `{text}`", file.display()));
            }
        }
    }
    assert!(
        offending.is_empty(),
        "kernel modules name upper layers:\n{}",
        offending.join("\n")
    );
}
