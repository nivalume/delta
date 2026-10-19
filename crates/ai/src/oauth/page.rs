//! Browser pages shown by the OAuth loopback callback server. Port of upstream
//! `utils/oauth-page.ts`; the markup is identical.

const LOGO_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 800" aria-hidden="true"><path fill="#F09082" d="M165.29 165.29H517.36V400H400V282.65H165.29Z"/><path fill="#4D9ABF" d="M165.29 282.65H282.65V400H400V517.36H282.65V634.72H165.29Z"/><path fill="#F1BE58" d="M517.36 400H634.72V634.72H517.36Z"/></svg>"##;

const PAGE_HEAD: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>"#;
const PAGE_BEFORE_LOGO: &str = r#"</title>
  <style>
    :root {
      --text: #fafafa;
      --text-dim: #a1a1aa;
      --page-bg: #09090b;
      --font-sans: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }
    * { box-sizing: border-box; }
    html { color-scheme: dark; }
    body {
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: var(--page-bg);
      color: var(--text);
      font-family: var(--font-sans);
      text-align: center;
    }
    main {
      width: 100%;
      max-width: 560px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }
    .logo {
      width: 72px;
      height: 72px;
      display: block;
      margin-bottom: 24px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: 28px;
      line-height: 1.15;
      font-weight: 650;
      color: var(--text);
    }
    p {
      margin: 0;
      line-height: 1.7;
      color: var(--text-dim);
      font-size: 15px;
    }
    .details {
      margin-top: 16px;
      font-family: var(--font-mono);
      font-size: 13px;
      color: var(--text-dim);
      white-space: pre-wrap;
      word-break: break-word;
    }
  </style>
</head>
<body>
  <main>
    <div class="logo">"#;
const PAGE_BEFORE_HEADING: &str = r"</div>
    <h1>";
const PAGE_BEFORE_MESSAGE: &str = r"</h1>
    <p>";
const PAGE_BEFORE_DETAILS: &str = r"</p>
    ";
const PAGE_TAIL: &str = r"
  </main>
</body>
</html>";

/// Escape `&`, `<`, `>`, `"` and `'` for HTML text and attribute values.
pub fn escape_html(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            c => escaped.push(c),
        }
    }
    escaped
}

fn render_page(title: &str, heading: &str, message: &str, details: Option<&str>) -> String {
    let details = details
        .filter(|details| !details.is_empty())
        .map(|details| format!("<div class=\"details\">{}</div>", escape_html(details)))
        .unwrap_or_default();
    let mut page = String::with_capacity(4096);
    page.push_str(PAGE_HEAD);
    page.push_str(&escape_html(title));
    page.push_str(PAGE_BEFORE_LOGO);
    page.push_str(LOGO_SVG);
    page.push_str(PAGE_BEFORE_HEADING);
    page.push_str(&escape_html(heading));
    page.push_str(PAGE_BEFORE_MESSAGE);
    page.push_str(&escape_html(message));
    page.push_str(PAGE_BEFORE_DETAILS);
    page.push_str(&details);
    page.push_str(PAGE_TAIL);
    page
}

/// The "Authentication successful" page with `message` below the heading.
pub fn oauth_success_html(message: &str) -> String {
    render_page("Authentication successful", "Authentication successful", message, None)
}

/// The "Authentication failed" page with `message` and optional monospace `details`.
pub fn oauth_error_html(message: &str, details: Option<&str>) -> String {
    render_page("Authentication failed", "Authentication failed", message, details)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_escape_text_and_carry_the_logo() {
        let page = oauth_error_html("Bad <state>", Some("a & 'b'"));
        assert!(page.starts_with("<!doctype html>"));
        assert!(page.contains("<title>Authentication failed</title>"));
        assert!(page.contains("<p>Bad &lt;state&gt;</p>"));
        assert!(page.contains("<div class=\"details\">a &amp; &#39;b&#39;</div>"));
        assert!(page.contains("fill=\"#F09082\""));
        let page = oauth_success_html("Signed in.");
        assert!(page.contains("<h1>Authentication successful</h1>"));
        assert!(!page.contains("class=\"details\">"));
    }
}
