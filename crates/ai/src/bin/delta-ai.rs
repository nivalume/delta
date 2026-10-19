//! `delta-ai`: log in to the built-in OAuth providers and save the credential to
//! `auth.json` in the current directory. Port of upstream `src/cli.ts`; built with the
//! `cli` feature.

use futures::future::BoxFuture;
use serde_json::{Map, Value};
use std::io::Write as _;
use std::sync::Arc;
use tokio::sync::mpsc;

use delta_ai::auth::{AuthEvent, AuthInteraction, AuthPrompt, LoginOptions};
use delta_ai::providers::builtin::builtin_providers;
use delta_ai::{Credential, ModelsError, ModelsErrorCode, Provider};

const AUTH_FILE: &str = "auth.json";

/// Built-in providers that have an OAuth login.
fn oauth_providers() -> Vec<Arc<dyn Provider>> {
    builtin_providers()
        .into_iter()
        .filter(|provider| provider.auth().oauth.is_some())
        .collect()
}

/// Lines of stdin, read by a background thread so a pending read never keeps the process
/// alive after `main` returns.
struct Stdin {
    lines: tokio::sync::Mutex<mpsc::UnboundedReceiver<String>>,
}

impl Stdin {
    fn spawn() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            loop {
                let mut line = String::new();
                match stdin.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let line = line.trim_end_matches(['\r', '\n']).to_owned();
                        if sender.send(line).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        Self {
            lines: tokio::sync::Mutex::new(receiver),
        }
    }

    /// Print `question` and read one line (upstream `rl.question`).
    async fn prompt(&self, question: &str) -> Result<String, ModelsError> {
        print!("{question}");
        let _ = std::io::stdout().flush();
        self.lines
            .lock()
            .await
            .recv()
            .await
            .ok_or_else(|| ModelsError::new(ModelsErrorCode::Aborted, "Input closed"))
    }
}

/// `Number.parseInt(text, 10)`: optional sign and leading digits after whitespace.
fn parse_int(text: &str) -> Option<i64> {
    let text = text.trim_start();
    let (sign, digits) = match text.strip_prefix('-') {
        Some(rest) => (-1, rest),
        None => (1, text.strip_prefix('+').unwrap_or(text)),
    };
    let end = digits.find(|c: char| !c.is_ascii_digit()).unwrap_or(digits.len());
    digits[..end].parse::<i64>().ok().map(|value| sign * value)
}

/// The option at the 1-based `answer`, if any.
fn pick<T>(items: &[T], answer: &str) -> Option<usize> {
    let index = parse_int(answer)?.checked_sub(1)?;
    usize::try_from(index).ok().filter(|index| *index < items.len())
}

fn load_auth() -> Map<String, Value> {
    std::fs::read_to_string(AUTH_FILE)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|value| match value {
            Value::Object(map) => Some(map),
            _ => None,
        })
        .unwrap_or_default()
}

fn save_auth(auth: &Map<String, Value>) -> Result<(), ModelsError> {
    let text = serde_json::to_string_pretty(auth)
        .map_err(|error| ModelsError::with_source(ModelsErrorCode::Auth, "Cannot serialize auth.json", error))?;
    std::fs::write(AUTH_FILE, text)
        .map_err(|error| ModelsError::with_source(ModelsErrorCode::Auth, format!("Cannot write {AUTH_FILE}"), error))
}

struct Terminal {
    stdin: Arc<Stdin>,
    cancel: tokio_util::sync::CancellationToken,
}

impl Terminal {
    async fn answer(&self, prompt: AuthPrompt) -> Result<String, ModelsError> {
        match prompt {
            AuthPrompt::Select { message, options } => {
                println!("\n{message}");
                for (index, option) in options.iter().enumerate() {
                    println!("  {}. {}", index + 1, option.label);
                }
                let answer = self
                    .stdin
                    .prompt(&format!("Enter number (1-{}): ", options.len()))
                    .await?;
                let index = pick(&options, &answer)
                    .ok_or_else(|| ModelsError::new(ModelsErrorCode::Auth, "Invalid selection"))?;
                Ok(options[index].id.clone())
            }
            AuthPrompt::Text { message, placeholder } | AuthPrompt::Secret { message, placeholder } => {
                self.stdin.prompt(&question(&message, placeholder.as_deref())).await
            }
            AuthPrompt::ManualCode {
                message,
                placeholder,
                cancel,
            } => {
                let question = question(&message, placeholder.as_deref());
                tokio::select! {
                    () = cancel.cancelled() => Err(ModelsError::new(ModelsErrorCode::Aborted, "Login cancelled")),
                    answer = self.stdin.prompt(&question) => answer,
                }
            }
        }
    }
}

fn question(message: &str, placeholder: Option<&str>) -> String {
    match placeholder.filter(|placeholder| !placeholder.is_empty()) {
        Some(placeholder) => format!("{message} ({placeholder}): "),
        None => format!("{message}: "),
    }
}

impl AuthInteraction for Terminal {
    fn prompt(&self, prompt: AuthPrompt) -> BoxFuture<'_, Result<String, ModelsError>> {
        Box::pin(self.answer(prompt))
    }

    fn notify(&self, event: AuthEvent) {
        match event {
            AuthEvent::AuthUrl { url, instructions } => {
                println!("\nOpen this URL in your browser:\n{url}");
                if let Some(instructions) = instructions.filter(|instructions| !instructions.is_empty()) {
                    println!("{instructions}");
                }
            }
            AuthEvent::DeviceCode {
                user_code,
                verification_uri,
                ..
            } => {
                println!("\nOpen this URL in your browser:\n{verification_uri}");
                println!("Enter code: {user_code}");
            }
            AuthEvent::Info { message, .. } | AuthEvent::Progress { message } => println!("{message}"),
        }
    }

    fn cancel(&self) -> tokio_util::sync::CancellationToken {
        self.cancel.clone()
    }
}

async fn login(providers: &[Arc<dyn Provider>], provider_id: &str, stdin: Arc<Stdin>) -> Result<(), ModelsError> {
    let provider = providers
        .iter()
        .find(|provider| provider.id() == provider_id)
        .ok_or_else(|| ModelsError::new(ModelsErrorCode::Provider, format!("Unknown provider: {provider_id}")))?;
    let Some(oauth) = provider.auth().oauth.clone() else {
        return Err(ModelsError::new(
            ModelsErrorCode::Provider,
            format!("Unknown provider: {provider_id}"),
        ));
    };
    let terminal = Terminal {
        stdin,
        cancel: tokio_util::sync::CancellationToken::new(),
    };
    // This CLI does not persist an installation id; apps should reuse one across logins.
    let options = LoginOptions {
        device_id: Some(Arc::new(|| uuid::Uuid::new_v4().to_string())),
        agent_name: None,
    };
    let credential = oauth.login(&terminal, &options).await?;
    let mut auth = load_auth();
    let value = serde_json::to_value(Credential::OAuth(credential))
        .map_err(|error| ModelsError::with_source(ModelsErrorCode::Auth, "Cannot serialize credential", error))?;
    auth.insert(provider_id.to_owned(), value);
    save_auth(&auth)?;
    println!("\nCredentials saved to {AUTH_FILE}");
    Ok(())
}

async fn run(args: Vec<String>) -> Result<(), ModelsError> {
    let providers = oauth_providers();
    let command = args.first().map(String::as_str);
    match command {
        None | Some("help" | "--help" | "-h") => {
            let provider_list = providers
                .iter()
                .map(|provider| format!("  {:<20} {}", provider.id(), provider.name()))
                .collect::<Vec<_>>()
                .join("\n");
            println!(
                "Usage: delta-ai <command> [provider]\n\nCommands:\n  login [provider]  Login to an OAuth provider\n  list              List available providers\n\nProviders:\n{provider_list}"
            );
            Ok(())
        }
        Some("list") => {
            for provider in &providers {
                println!("{:<20} {}", provider.id(), provider.name());
            }
            Ok(())
        }
        Some("login") => {
            let stdin = Arc::new(Stdin::spawn());
            let provider_id = match args.get(1) {
                Some(id) => Some(id.clone()),
                None => {
                    for (index, provider) in providers.iter().enumerate() {
                        println!("  {}. {}", index + 1, provider.name());
                    }
                    let answer = stdin.prompt(&format!("Enter number (1-{}): ", providers.len())).await?;
                    pick(&providers, &answer).map(|index| providers[index].id().to_owned())
                }
            };
            let provider_id = provider_id
                .filter(|id| providers.iter().any(|provider| provider.id() == id))
                .ok_or_else(|| {
                    ModelsError::new(
                        ModelsErrorCode::Provider,
                        format!("Unknown provider: {}", args.get(1).map_or("", String::as_str)),
                    )
                })?;
            login(&providers, &provider_id, stdin).await
        }
        Some(command) => Err(ModelsError::new(
            ModelsErrorCode::Provider,
            format!("Unknown command: {command}"),
        )),
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(error) = run(args).await {
        eprintln!("Error: {}", error.message());
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_numbers_like_parse_int() {
        assert_eq!(parse_int(" 2abc"), Some(2));
        assert_eq!(parse_int("-1"), Some(-1));
        assert_eq!(parse_int("x"), None);
        assert_eq!(pick(&[1, 2], "2"), Some(1));
        assert_eq!(pick(&[1, 2], "3"), None);
        assert_eq!(pick(&[1, 2], "0"), None);
    }

    #[test]
    fn lists_only_oauth_providers() {
        let ids: Vec<String> = oauth_providers()
            .iter()
            .map(|provider| provider.id().to_owned())
            .collect();
        assert!(ids.contains(&"anthropic".to_owned()));
        assert!(ids.contains(&"openrouter".to_owned()));
        assert!(ids.contains(&"xai".to_owned()));
        assert!(!ids.contains(&"groq".to_owned()));
    }
}
