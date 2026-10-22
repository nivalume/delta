//! Path arithmetic of the local environment: `~` and `file://` expansion, resolution against
//! the `cwd`, and lexical normalization as Node's `path.resolve` / `path.join`. Nothing here
//! touches the file system.

use std::path::{Component, MAIN_SEPARATOR_STR, Path, PathBuf};

/// `path` resolved against `cwd`: `~` and `~/…` name the home directory, a `file://` URL its
/// file path (a malformed URL stays an ordinary path), and the result is normalized.
pub(super) fn resolve(cwd: &Path, path: &str) -> PathBuf {
    let expanded = expand(path);
    if expanded.is_absolute() {
        normalize(&expanded)
    } else {
        normalize(&cwd.join(expanded))
    }
}

/// `parts` joined and normalized; a relative result stays relative, an empty one is `.`.
pub(super) fn join(parts: &[&str]) -> PathBuf {
    let joined = parts
        .iter()
        .filter(|part| !part.is_empty())
        .copied()
        .collect::<Vec<_>>()
        .join(MAIN_SEPARATOR_STR);
    let normalized = normalize(Path::new(&joined));
    if normalized.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        normalized
    }
}

fn expand(path: &str) -> PathBuf {
    if path == "~" {
        if let Some(home) = std::env::home_dir() {
            return home;
        }
    } else if let Some(rest) = path.strip_prefix("~/").or_else(|| home_relative_windows(path)) {
        if let Some(home) = std::env::home_dir() {
            return home.join(rest);
        }
    } else if path.starts_with("file://")
        && let Some(file) = url::Url::parse(path).ok().and_then(|url| url.to_file_path().ok())
    {
        return file;
    }
    PathBuf::from(path)
}

/// `~\…` on Windows.
fn home_relative_windows(path: &str) -> Option<&str> {
    if cfg!(windows) { path.strip_prefix("~\\") } else { None }
}

/// Resolve `.` and `..` lexically. `..` above the root stays at the root; leading `..` of a
/// relative path are kept.
pub(super) fn normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => normalized.push(component),
            Component::CurDir => {}
            Component::ParentDir => match normalized.components().next_back() {
                Some(Component::Normal(_)) => {
                    normalized.pop();
                }
                Some(Component::Prefix(_) | Component::RootDir) => {}
                _ => normalized.push(component),
            },
        }
    }
    normalized
}

#[cfg(all(test, unix))]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{join, normalize, resolve};

    #[test]
    fn resolves_against_the_cwd_and_normalizes() {
        let cwd = Path::new("/work/project");
        assert_eq!(
            resolve(cwd, "src/../lib/./a.rs"),
            PathBuf::from("/work/project/lib/a.rs")
        );
        assert_eq!(resolve(cwd, "/etc//hosts/"), PathBuf::from("/etc/hosts"));
        assert_eq!(resolve(cwd, "../../../x"), PathBuf::from("/x"));
        assert_eq!(resolve(cwd, "."), PathBuf::from("/work/project"));
    }

    #[test]
    fn expands_home_and_file_urls() {
        let cwd = Path::new("/work");
        if let Some(home) = std::env::home_dir() {
            assert_eq!(resolve(cwd, "~"), home);
            assert_eq!(resolve(cwd, "~/notes"), home.join("notes"));
        }
        assert_eq!(resolve(cwd, "file:///tmp/a%20b.txt"), PathBuf::from("/tmp/a b.txt"));
        // A URL with a remote host has no local path: an ordinary relative path.
        assert_eq!(resolve(cwd, "file://host/x"), PathBuf::from("/work/file:/host/x"));
    }

    #[test]
    fn joins_and_keeps_relative_paths_relative() {
        assert_eq!(join(&["/a", "b", "../c"]), PathBuf::from("/a/c"));
        assert_eq!(join(&["a", "/b"]), PathBuf::from("a/b"));
        assert_eq!(join(&["..", "a"]), PathBuf::from("../a"));
        assert_eq!(join(&["a", ".."]), PathBuf::from("."));
        assert_eq!(join(&[]), PathBuf::from("."));
        assert_eq!(normalize(Path::new("/../a")), PathBuf::from("/a"));
    }
}
