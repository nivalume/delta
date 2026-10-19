//! Proxy resolution for one target URL: port of upstream `utils/node-http-proxy.ts`.
//!
//! The proxy comes from `<protocol>_proxy` or `all_proxy` and `no_proxy` (each name lowercase,
//! then uppercase), looked up in the scoped `options.env` first and then in the process
//! environment.

use url::Url;

use super::EnvLookup;
use crate::error::{Error, Result};

pub(super) const UNSUPPORTED_PROXY_PROTOCOL_MESSAGE: &str =
    "Unsupported proxy protocol. SOCKS and PAC proxy URLs are not supported; use an HTTP or HTTPS proxy URL.";

fn default_proxy_port(protocol: &str) -> i64 {
    match protocol {
        "ftp" => 21,
        "gopher" => 70,
        "http" | "ws" => 80,
        "https" | "wss" => 443,
        _ => 0,
    }
}

/// `env[lower] || env[UPPER] || process[lower] || process[UPPER]`.
fn proxy_env(key: &str, env: &EnvLookup<'_>) -> Option<String> {
    let lower = key.to_lowercase();
    let upper = key.to_uppercase();
    env.scoped(&lower)
        .or_else(|| env.scoped(&upper))
        .or_else(|| env.process(&lower))
        .or_else(|| env.process(&upper))
}

/// JavaScript's `WhiteSpace` and `LineTerminator` characters (`\s`, `String.prototype.trim`).
fn is_js_whitespace(ch: char) -> bool {
    matches!(
        ch,
        '\t' | '\n' | '\u{0b}' | '\u{0c}' | '\r' | ' ' | '\u{a0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{feff}'
    )
}

/// `Number.parseInt(value, 10)`: leading whitespace, an optional sign, then decimal digits.
/// `None` stands for `NaN`.
fn js_parse_int(value: &str) -> Option<i64> {
    let value = value.trim_start_matches(is_js_whitespace);
    let (negative, digits) = match value.as_bytes().first() {
        Some(b'-') => (true, &value[1..]),
        Some(b'+') => (false, &value[1..]),
        _ => (false, value),
    };
    let digits = &digits[..digits.bytes().take_while(u8::is_ascii_digit).count()];
    if digits.is_empty() {
        return None;
    }
    let magnitude = digits.bytes().fold(0_i64, |acc, digit| {
        acc.saturating_mul(10).saturating_add(i64::from(digit - b'0'))
    });
    Some(if negative { -magnitude } else { magnitude })
}

fn strip_brackets(host: &str) -> &str {
    host.strip_prefix('[')
        .and_then(|inner| inner.strip_suffix(']'))
        .unwrap_or(host)
}

/// One `NO_PROXY` entry as `(host, port)`; port 0 matches every port.
fn parse_no_proxy_entry(entry: &str) -> Option<(String, i64)> {
    let trimmed = entry.trim_matches(is_js_whitespace).to_lowercase();
    if trimmed.is_empty() {
        return None;
    }

    if trimmed.starts_with('[')
        && let Some(closing) = trimmed.find(']')
    {
        let host = trimmed[1..closing].to_owned();
        let rest = &trimmed[closing + 1..];
        if let Some(port) = rest.strip_prefix(':') {
            return Some((host, js_parse_int(port).unwrap_or(0)));
        }
        return Some((host, 0));
    }

    // A bare IPv6 address.
    if trimmed.split(':').count() > 2 {
        return Some((trimmed, 0));
    }

    if let Some(colon) = trimmed.rfind(':')
        && trimmed.find(':') == Some(colon)
        && let Some(port) = js_parse_int(&trimmed[colon + 1..])
    {
        return Some((trimmed[..colon].to_owned(), port));
    }

    Some((trimmed, 0))
}

fn should_proxy_hostname(hostname: &str, port: i64, env: &EnvLookup<'_>) -> bool {
    let no_proxy = proxy_env("no_proxy", env).unwrap_or_default().to_lowercase();
    if no_proxy.is_empty() {
        return true;
    }
    if no_proxy == "*" {
        return false;
    }

    let target = hostname.to_lowercase();
    let target = strip_brackets(&target);

    no_proxy
        .split(|ch: char| ch == ',' || is_js_whitespace(ch))
        .all(|entry| {
            let Some((host, entry_port)) = parse_no_proxy_entry(entry) else {
                return true;
            };
            if entry_port != 0 && entry_port != port {
                return true;
            }

            let mut domain = strip_brackets(&host);
            if let Some(rest) = domain.strip_prefix("*.") {
                domain = rest;
            } else if let Some(rest) = domain.strip_prefix('.').or_else(|| domain.strip_prefix('*')) {
                domain = rest;
            }
            if domain.is_empty() {
                return true;
            }
            target != domain && !target.ends_with(&format!(".{domain}"))
        })
}

fn proxy_for_url(target_url: &str, env: &EnvLookup<'_>) -> Option<String> {
    let url = Url::parse(target_url).ok()?;
    let host = url.host_str().filter(|host| !host.is_empty())?;
    let protocol = url.scheme();
    let hostname = strip_brackets(host);
    let port = url
        .port()
        .map(i64::from)
        .filter(|port| *port != 0)
        .unwrap_or_else(|| default_proxy_port(protocol));
    if !should_proxy_hostname(hostname, port, env) {
        return None;
    }

    let proxy = proxy_env(&format!("{protocol}_proxy"), env).or_else(|| proxy_env("all_proxy", env))?;
    Some(if proxy.contains("://") {
        proxy
    } else {
        format!("{protocol}://{proxy}")
    })
}

/// The HTTP(S) proxy for `target_url`, or `None` when no proxy applies.
pub(super) fn resolve_http_proxy_url_for_target(target_url: &str, env: &EnvLookup<'_>) -> Result<Option<Url>> {
    let Some(proxy) = proxy_for_url(target_url, env) else {
        return Ok(None);
    };
    let Ok(proxy_url) = Url::parse(&proxy) else {
        let quoted = serde_json::to_string(&proxy).unwrap_or_else(|_| format!("\"{proxy}\""));
        // Node's `new URL()` fails with "Invalid URL".
        return Err(Error::msg(format!("Invalid proxy URL {quoted}: Invalid URL")));
    };
    if proxy_url.scheme() != "http" && proxy_url.scheme() != "https" {
        return Err(Error::msg(format!(
            "{UNSUPPORTED_PROXY_PROTOCOL_MESSAGE} Got {}:",
            proxy_url.scheme()
        )));
    }
    Ok(Some(proxy_url))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::types::ProviderEnv;

    fn env_of(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect()
    }

    /// Resolve `target` with the given process environment and optional scoped `options.env`.
    fn resolve(target: &str, process: &[(&str, &str)], scoped: Option<&[(&str, &str)]>) -> Result<Option<String>> {
        let process = env_of(process);
        let process_fn = move |name: &str| process.get(name).cloned();
        let scoped: Option<ProviderEnv> = scoped.map(|pairs| env_of(pairs).into_iter().collect());
        let env = EnvLookup {
            scoped: scoped.as_ref(),
            process: &process_fn,
        };
        resolve_http_proxy_url_for_target(target, &env).map(|url| url.map(String::from))
    }

    const BEDROCK: &str = "https://bedrock-runtime.us-east-1.amazonaws.com";

    #[test]
    fn respects_no_proxy_exclusions() {
        let process = [
            ("HTTPS_PROXY", "http://proxy.example:8080"),
            ("NO_PROXY", "bedrock-runtime.us-east-1.amazonaws.com"),
        ];
        assert_eq!(resolve(BEDROCK, &process, None).unwrap(), None);
    }

    #[test]
    fn resolves_http_and_https_proxy_urls() {
        let process = [("HTTPS_PROXY", "http://proxy.example:8080")];
        assert_eq!(
            resolve(BEDROCK, &process, None).unwrap().as_deref(),
            Some("http://proxy.example:8080/")
        );
    }

    #[test]
    fn prefers_scoped_proxy_env_aliases_before_process_env_aliases() {
        let process = [("https_proxy", "http://process-proxy.example:8080")];
        let scoped = [("HTTPS_PROXY", "http://scoped-proxy.example:8080")];
        assert_eq!(
            resolve(BEDROCK, &process, Some(&scoped)).unwrap().as_deref(),
            Some("http://scoped-proxy.example:8080/")
        );
    }

    #[test]
    fn rejects_socks_and_pac_proxy_urls_explicitly() {
        let process = [("HTTPS_PROXY", "socks5://proxy.example:1080")];
        let error = resolve(BEDROCK, &process, None).unwrap_err().to_string();
        assert!(error.contains(UNSUPPORTED_PROXY_PROTOCOL_MESSAGE), "{error}");
        assert!(error.ends_with("Got socks5:"), "{error}");
    }

    #[test]
    fn handles_subdomain_wildcards_ipv6_and_ports_in_no_proxy() {
        let process = [
            ("HTTPS_PROXY", "http://proxy.example:8080"),
            (
                "NO_PROXY",
                "example.com, .wildcard.org, *.star.net, ::1, [2001:db8::1], 127.0.0.1:8080",
            ),
        ];
        let proxied = Some("http://proxy.example:8080/".to_owned());

        assert_eq!(resolve("https://example.com", &process, None).unwrap(), None);
        assert_eq!(resolve("https://api.example.com", &process, None).unwrap(), None);
        assert_eq!(resolve("https://wildcard.org", &process, None).unwrap(), None);
        assert_eq!(resolve("https://api.wildcard.org", &process, None).unwrap(), None);
        assert_eq!(resolve("https://star.net", &process, None).unwrap(), None);
        assert_eq!(resolve("https://api.star.net", &process, None).unwrap(), None);
        assert_eq!(resolve("https://notexample.com", &process, None).unwrap(), proxied);

        assert_eq!(resolve("https://[::1]:80", &process, None).unwrap(), None);
        assert_eq!(resolve("https://[2001:db8::1]", &process, None).unwrap(), None);
        assert_eq!(resolve("https://127.0.0.1:8080", &process, None).unwrap(), None);
        assert_eq!(resolve("https://127.0.0.1:3000", &process, None).unwrap(), proxied);
    }

    // Cases below cover branches of `node-http-proxy.ts` the upstream tests do not reach.

    #[test]
    fn no_proxy_star_disables_every_proxy() {
        let process = [("HTTPS_PROXY", "http://proxy.example:8080"), ("no_proxy", "*")];
        assert_eq!(resolve(BEDROCK, &process, None).unwrap(), None);
    }

    #[test]
    fn scoped_no_proxy_wins_over_process_no_proxy() {
        let process = [
            ("HTTPS_PROXY", "http://proxy.example:8080"),
            ("NO_PROXY", "amazonaws.com"),
        ];
        let scoped = [("no_proxy", "internal.example")];
        assert_eq!(
            resolve(BEDROCK, &process, Some(&scoped)).unwrap().as_deref(),
            Some("http://proxy.example:8080/")
        );
    }

    #[test]
    fn lowercase_names_win_over_uppercase_names_in_the_same_source() {
        let scoped = [
            ("https_proxy", "http://lower.example:1"),
            ("HTTPS_PROXY", "http://upper.example:2"),
        ];
        assert_eq!(
            resolve(BEDROCK, &[], Some(&scoped)).unwrap().as_deref(),
            Some("http://lower.example:1/")
        );
    }

    #[test]
    fn uses_the_target_protocol_and_falls_back_to_all_proxy() {
        let process = [
            ("HTTPS_PROXY", "http://https-only.example:1"),
            ("ALL_PROXY", "http://all.example:2"),
        ];
        assert_eq!(
            resolve("http://bedrock.internal", &process, None).unwrap().as_deref(),
            Some("http://all.example:2/")
        );
        assert_eq!(
            resolve(BEDROCK, &process, None).unwrap().as_deref(),
            Some("http://https-only.example:1/")
        );
    }

    #[test]
    fn a_proxy_without_scheme_takes_the_target_protocol() {
        let process = [("HTTPS_PROXY", "proxy.example:8443")];
        assert_eq!(
            resolve(BEDROCK, &process, None).unwrap().as_deref(),
            Some("https://proxy.example:8443/")
        );
    }

    #[test]
    fn no_proxy_ports_compare_with_the_default_port_of_the_target() {
        let process = [
            ("HTTPS_PROXY", "http://proxy.example:8080"),
            ("NO_PROXY", "example.com:443"),
        ];
        assert_eq!(resolve("https://example.com", &process, None).unwrap(), None);
        assert!(resolve("https://example.com:8443", &process, None).unwrap().is_some());
        assert!(
            resolve(
                "http://example.com",
                &[("HTTP_PROXY", "http://p:1"), ("NO_PROXY", "example.com:443")],
                None
            )
            .unwrap()
            .is_some()
        );
    }

    #[test]
    fn invalid_proxy_urls_fail_with_the_quoted_value() {
        let process = [("HTTPS_PROXY", "http://proxy example")];
        assert_eq!(
            resolve(BEDROCK, &process, None).unwrap_err().to_string(),
            "Invalid proxy URL \"http://proxy example\": Invalid URL"
        );
    }

    #[test]
    fn unparsable_or_hostless_targets_get_no_proxy() {
        let process = [
            ("HTTPS_PROXY", "http://proxy.example:8080"),
            ("ALL_PROXY", "http://proxy.example:8080"),
        ];
        assert_eq!(resolve("", &process, None).unwrap(), None);
        assert_eq!(resolve("not a url", &process, None).unwrap(), None);
        assert_eq!(resolve("file:///tmp/socket", &process, None).unwrap(), None);
    }

    #[test]
    fn blank_and_whitespace_entries_are_ignored() {
        let process = [
            ("HTTPS_PROXY", "http://proxy.example:8080"),
            ("NO_PROXY", " ,\t, other.example "),
        ];
        assert!(resolve(BEDROCK, &process, None).unwrap().is_some());
    }

    #[test]
    fn parse_int_follows_javascript() {
        assert_eq!(js_parse_int("8080"), Some(8080));
        assert_eq!(js_parse_int("80abc"), Some(80));
        assert_eq!(js_parse_int("-1"), Some(-1));
        assert_eq!(js_parse_int("+5"), Some(5));
        assert_eq!(js_parse_int(""), None);
        assert_eq!(js_parse_int("abc"), None);
    }
}
