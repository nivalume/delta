//! Each layer names only the layers below it (README.md, "Architecture and seams"): the
//! kernel names no agent, surface or side module; the agent layer names no surface module;
//! the side layer (`env`) names nothing else in the crate. Scans the sources for `crate::`
//! paths into a layer a module may not reach.

use std::fs;
use std::path::{Path, PathBuf};

/// The kernel modules, as `src/<name>/` or `src/<name>.rs`.
const KERNEL: [&str; 5] = ["records", "documents", "storage", "session", "scheduler"];

/// The agent layer's modules.
const AGENT: [&str; 11] = [
    "agent",
    "approval",
    "compaction",
    "context",
    "generation",
    "inbox",
    "model",
    "prompt",
    "registry",
    "tool",
    "usage",
];

/// The surface modules.
const SURFACE: [&str; 2] = ["harness", "observe"];

/// The side layer's modules.
const SIDE: [&str; 2] = ["env", "tools"];

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

/// Every `crate::` path in the sources of `modules` whose first segment is in `forbidden`.
fn offending(modules: &[&str], forbidden: &[&str]) -> Vec<String> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    for module in modules {
        rust_files(&src.join(module), &mut files);
        rust_files(&src.join(format!("{module}.rs")), &mut files);
    }
    assert!(!files.is_empty(), "no sources of {modules:?} under {}", src.display());
    let mut offending = Vec::new();
    for file in &files {
        let source = fs::read_to_string(file).expect("sources are readable");
        let lines: Vec<&str> = source.lines().collect();
        for (line, root) in crate_roots(&source) {
            if forbidden.contains(&root.as_str()) {
                let text = lines.get(line - 1).map_or("", |text| text.trim());
                offending.push(format!("{}:{line}: crate::{root} in `{text}`", file.display()));
            }
        }
    }
    offending
}

#[test]
fn kernel_modules_do_not_name_upper_layers() {
    let upper: Vec<&str> = AGENT.iter().chain(&SURFACE).chain(&SIDE).copied().collect();
    let offending = offending(&KERNEL, &upper);
    assert!(
        offending.is_empty(),
        "kernel modules name upper layers:\n{}",
        offending.join("\n")
    );
}

#[test]
fn agent_modules_do_not_name_the_surface() {
    let offending = offending(&AGENT, &SURFACE);
    assert!(
        offending.is_empty(),
        "agent modules name surface modules:\n{}",
        offending.join("\n")
    );
}

#[test]
fn the_environment_names_nothing_else_in_the_crate() {
    let others: Vec<&str> = KERNEL.iter().chain(&AGENT).chain(&SURFACE).copied().collect();
    let offending = offending(&["env"], &others);
    assert!(
        offending.is_empty(),
        "env names other modules:\n{}",
        offending.join("\n")
    );
}
