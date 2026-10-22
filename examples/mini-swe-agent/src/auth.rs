//! Credentials: delta-ai's [`CredentialStore`] on a JSON file in the shape of pi's `auth.json`
//! (one type-tagged credential per provider id), and the terminal side of delta-ai's login
//! flows.

use std::io::{IsTerminal as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use delta_ai::auth::{
    AuthEvent, AuthInteraction, AuthPrompt, AuthType, CredentialInfo, CredentialStore, LoginOptions, ModifyFn,
};
use delta_ai::{Credential, Models, ModelsError, ModelsErrorCode, Provider};
use futures::future::BoxFuture;
use serde_json::{Map, Value as Json};

/// Credentials in a JSON file. Every write is a read-modify-write under a lock: a mutex in
/// this process and an exclusive lock on `<file>.lock` across processes. Writes go to a
/// temporary file renamed over the old one; on Unix the file is readable by its owner only.
#[derive(Debug)]
pub struct FileCredentialStore {
    path: PathBuf,
    lock: tokio::sync::Mutex<()>,
}

fn auth_error(message: String, error: impl std::error::Error + Send + Sync + 'static) -> ModelsError {
    ModelsError::with_source(ModelsErrorCode::Auth, message, error)
}

impl FileCredentialStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            lock: tokio::sync::Mutex::new(()),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Every entry; empty when the file does not exist.
    fn entries(&self) -> Result<Map<String, Json>, ModelsError> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Map::new()),
            Err(error) => return Err(auth_error(format!("Cannot read {}", self.path.display()), error)),
        };
        if text.trim().is_empty() {
            return Ok(Map::new());
        }
        match serde_json::from_str(&text) {
            Ok(Json::Object(entries)) => Ok(entries),
            Ok(_) => Err(ModelsError::new(
                ModelsErrorCode::Auth,
                format!("{} is not a JSON object", self.path.display()),
            )),
            Err(error) => Err(auth_error(format!("Cannot parse {}", self.path.display()), error)),
        }
    }

    fn credential(&self, entries: &Map<String, Json>, provider_id: &str) -> Result<Option<Credential>, ModelsError> {
        entries
            .get(provider_id)
            .map(|entry| {
                serde_json::from_value(entry.clone()).map_err(|error| {
                    auth_error(
                        format!("Invalid credential for {provider_id} in {}", self.path.display()),
                        error,
                    )
                })
            })
            .transpose()
    }

    fn write(&self, entries: &Map<String, Json>) -> Result<(), ModelsError> {
        let error = |error: std::io::Error| auth_error(format!("Cannot write {}", self.path.display()), error);
        if let Some(parent) = self.path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(error)?;
        }
        let text = serde_json::to_string_pretty(entries)
            .map_err(|error| auth_error("Cannot serialize credentials".to_owned(), error))?;
        let temporary = sibling(&self.path, &format!("tmp-{}", std::process::id()));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut file = options.open(&temporary).map_err(error)?;
        file.write_all(text.as_bytes()).map_err(error)?;
        file.sync_all().map_err(error)?;
        drop(file);
        std::fs::rename(&temporary, &self.path).map_err(error)
    }

    /// Takes both locks; they are held until the guard drops.
    async fn lock(&self) -> Result<Exclusive<'_>, ModelsError> {
        let guard = self.lock.lock().await;
        let lock_path = sibling(&self.path, "lock");
        let file = tokio::task::spawn_blocking(move || -> std::io::Result<std::fs::File> {
            if let Some(parent) = lock_path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&lock_path)?;
            file.lock()?;
            Ok(file)
        })
        .await
        .map_err(|error| auth_error("The credential lock task failed".to_owned(), error))?
        .map_err(|error| auth_error(format!("Cannot lock {}", self.path.display()), error))?;
        Ok(Exclusive { _guard: guard, file })
    }
}

/// Both locks of a [`FileCredentialStore`].
struct Exclusive<'a> {
    _guard: tokio::sync::MutexGuard<'a, ()>,
    file: std::fs::File,
}

impl Drop for Exclusive<'_> {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

/// `path` with `.suffix` appended to its file name.
fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().map(std::ffi::OsStr::to_os_string).unwrap_or_default();
    name.push(format!(".{suffix}"));
    path.with_file_name(name)
}

impl CredentialStore for FileCredentialStore {
    fn read<'a>(&'a self, provider_id: &'a str) -> BoxFuture<'a, Result<Option<Credential>, ModelsError>> {
        Box::pin(async move { self.credential(&self.entries()?, provider_id) })
    }

    fn list(&self) -> BoxFuture<'_, Result<Vec<CredentialInfo>, ModelsError>> {
        Box::pin(async move {
            let entries = self.entries()?;
            Ok(entries
                .iter()
                .filter_map(|(id, entry)| {
                    let credential: Credential = serde_json::from_value(entry.clone()).ok()?;
                    Some(CredentialInfo {
                        provider_id: id.clone(),
                        auth_type: credential.auth_type(),
                    })
                })
                .collect())
        })
    }

    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: ModifyFn,
    ) -> BoxFuture<'a, Result<Option<Credential>, ModelsError>> {
        Box::pin(async move {
            let _lock = self.lock().await?;
            let mut entries = self.entries()?;
            let current = self.credential(&entries, provider_id)?;
            let Some(next) = f(current.clone()).await? else {
                return Ok(current);
            };
            let value = serde_json::to_value(&next)
                .map_err(|error| auth_error("Cannot serialize the credential".to_owned(), error))?;
            entries.insert(provider_id.to_owned(), value);
            self.write(&entries)?;
            Ok(Some(next))
        })
    }

    fn delete<'a>(&'a self, provider_id: &'a str) -> BoxFuture<'a, Result<(), ModelsError>> {
        Box::pin(async move {
            let _lock = self.lock().await?;
            let mut entries = self.entries()?;
            if entries.remove(provider_id).is_some() {
                self.write(&entries)?;
            }
            Ok(())
        })
    }
}

/// Reads one line of stdin on a thread of its own, so a read a cancelled prompt leaves
/// behind never keeps the process alive. `None` at the end of input.
async fn read_line() -> Option<String> {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let read = std::io::stdin().read_line(&mut line);
        let _ = sender.send(match read {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(line.trim_end_matches(['\r', '\n']).to_owned()),
        });
    });
    receiver.await.ok().flatten()
}

/// Reads a line without echoing it, on a terminal; a plain line otherwise. `None` on Ctrl+C
/// or Ctrl+D.
async fn read_secret() -> Option<String> {
    if !std::io::stdin().is_terminal() {
        return read_line().await;
    }
    let (sender, receiver) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let _ = sender.send(read_hidden());
    });
    receiver.await.ok().flatten()
}

fn read_hidden() -> Option<String> {
    use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers, read};
    crossterm::terminal::enable_raw_mode().ok()?;
    let mut text = String::new();
    let entered = loop {
        let Ok(event) = read() else { break false };
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => match key.code {
                KeyCode::Enter => break true,
                KeyCode::Backspace => {
                    text.pop();
                }
                KeyCode::Char('c' | 'd') if key.modifiers.contains(KeyModifiers::CONTROL) => break false,
                KeyCode::Char(character) => text.push(character),
                _ => {}
            },
            Event::Paste(pasted) => text.push_str(&pasted),
            _ => {}
        }
    };
    let _ = crossterm::terminal::disable_raw_mode();
    println!();
    entered.then_some(text)
}

fn ask(message: &str, placeholder: Option<&str>) {
    match placeholder.filter(|placeholder| !placeholder.is_empty()) {
        Some(placeholder) => print!("{message} ({placeholder}): "),
        None => print!("{message}: "),
    }
    let _ = std::io::stdout().flush();
}

fn cancelled() -> ModelsError {
    ModelsError::new(ModelsErrorCode::Aborted, "Login cancelled")
}

/// The terminal side of a login: prompts on stdin and stdout, URLs and device codes printed.
#[derive(Clone, Copy, Debug, Default)]
pub struct TerminalInteraction;

impl AuthInteraction for TerminalInteraction {
    fn prompt(&self, prompt: AuthPrompt) -> BoxFuture<'_, Result<String, ModelsError>> {
        Box::pin(async move {
            match prompt {
                AuthPrompt::Select { message, options } => {
                    println!("\n{message}");
                    for (index, option) in options.iter().enumerate() {
                        match &option.description {
                            Some(description) => println!("  {}. {} - {description}", index + 1, option.label),
                            None => println!("  {}. {}", index + 1, option.label),
                        }
                    }
                    ask(&format!("Enter number (1-{})", options.len()), None);
                    let answer = read_line().await.ok_or_else(cancelled)?;
                    pick(options.len(), &answer)
                        .map(|index| options[index].id.clone())
                        .ok_or_else(|| ModelsError::new(ModelsErrorCode::Auth, "Invalid selection"))
                }
                AuthPrompt::Text { message, placeholder } => {
                    ask(&message, placeholder.as_deref());
                    read_line().await.ok_or_else(cancelled)
                }
                AuthPrompt::Secret { message, placeholder } => {
                    ask(&message, placeholder.as_deref());
                    read_secret().await.ok_or_else(cancelled)
                }
                AuthPrompt::ManualCode {
                    message,
                    placeholder,
                    cancel,
                } => {
                    ask(&message, placeholder.as_deref());
                    tokio::select! {
                        () = cancel.cancelled() => Err(cancelled()),
                        line = read_line() => line.ok_or_else(cancelled),
                    }
                }
            }
        })
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
            AuthEvent::Info { message, links } => {
                println!("{message}");
                for link in links {
                    match link.label {
                        Some(label) => println!("  {label}: {}", link.url),
                        None => println!("  {}", link.url),
                    }
                }
            }
            AuthEvent::Progress { message } => println!("{message}"),
        }
    }
}

/// The 0-based index a 1-based answer picks among `count` options.
pub fn pick(count: usize, answer: &str) -> Option<usize> {
    let index = answer.trim().parse::<usize>().ok()?.checked_sub(1)?;
    (index < count).then_some(index)
}

/// The ways one can log in to `provider`: OAuth first when it has it, then an API key it can
/// store.
pub fn login_methods(provider: &dyn Provider) -> Vec<AuthType> {
    let auth = provider.auth();
    let mut methods = Vec::new();
    if auth.oauth.is_some() {
        methods.push(AuthType::OAuth);
    }
    if auth.api_key.as_ref().is_some_and(|api_key| api_key.supports_login()) {
        methods.push(AuthType::ApiKey);
    }
    methods
}

/// This installation's id, kept in the home directory for the login flows that need one.
fn device_id() -> String {
    let path = crate::home::home_dir().join("device-id");
    if let Ok(id) = std::fs::read_to_string(&path)
        && !id.trim().is_empty()
    {
        return id.trim().to_owned();
    }
    let id = uuid::Uuid::new_v4().to_string();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, &id);
    id
}

/// Logs in to `provider_id` with `method` (its first method when `None`) and stores the
/// credential in `models`' store.
pub async fn login(
    models: &Models,
    provider_id: &str,
    method: Option<AuthType>,
    interaction: &dyn AuthInteraction,
) -> Result<AuthType, String> {
    let provider = models
        .provider(provider_id)
        .ok_or_else(|| format!("Unknown provider: {provider_id}"))?;
    let methods = login_methods(provider.as_ref());
    let method = match method {
        Some(method) if methods.contains(&method) => method,
        Some(method) => {
            return Err(format!(
                "{} has no {} login",
                provider.name(),
                match method {
                    AuthType::OAuth => "OAuth",
                    AuthType::ApiKey => "API key",
                }
            ));
        }
        None => *methods.first().ok_or_else(|| {
            format!(
                "{} has no login; it reads its credentials from the environment",
                provider.name()
            )
        })?,
    };
    let options = LoginOptions {
        device_id: Some(Arc::new(device_id)),
        agent_name: Some("delta-mini".to_owned()),
    };
    models
        .login(provider_id, method, interaction, &options)
        .await
        .map_err(|error| error.to_string())?;
    Ok(method)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use delta_ai::auth::{AuthResolutionOverrides, OAuthCredential};
    use delta_ai::providers::builtin::builtin_models;
    use delta_ai::{ApiKeyCredential, CreateModelsOptions};

    use super::*;

    fn api_key(key: &str) -> Credential {
        Credential::ApiKey(ApiKeyCredential {
            key: Some(key.to_owned()),
            env: None,
        })
    }

    #[tokio::test]
    async fn credentials_round_trip_in_pi_s_shape() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub/auth.json");
        let store = FileCredentialStore::new(&path);
        assert_eq!(store.read("anthropic").await.unwrap(), None);
        let stored = store
            .modify("anthropic", Box::new(|_| Box::pin(async { Ok(Some(api_key("sk-1"))) })))
            .await
            .unwrap();
        assert_eq!(stored, Some(api_key("sk-1")));
        let text = std::fs::read_to_string(&path).unwrap();
        let saved: Json = serde_json::from_str(&text).unwrap();
        assert_eq!(
            saved,
            serde_json::json!({ "anthropic": { "type": "api_key", "key": "sk-1" } })
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }

        // An entry this crate does not know survives a write; `None` leaves the entry as it is.
        let mut entries: Map<String, Json> = serde_json::from_str(&text).unwrap();
        entries.insert(
            "openai-codex".to_owned(),
            serde_json::json!({ "type": "oauth", "refresh": "r", "access": "a", "expires": 1, "accountId": "x" }),
        );
        entries.insert("custom".to_owned(), serde_json::json!({ "type": "something-else" }));
        std::fs::write(&path, serde_json::to_string(&entries).unwrap()).unwrap();
        let unchanged = store
            .modify("anthropic", Box::new(|_| Box::pin(async { Ok(None) })))
            .await
            .unwrap();
        assert_eq!(unchanged, Some(api_key("sk-1")));
        store
            .modify("openai", Box::new(|_| Box::pin(async { Ok(Some(api_key("sk-2"))) })))
            .await
            .unwrap();
        let saved: Json = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(saved["custom"], serde_json::json!({ "type": "something-else" }));
        let Some(Credential::OAuth(OAuthCredential { extra, .. })) = store.read("openai-codex").await.unwrap() else {
            panic!("an OAuth credential");
        };
        assert_eq!(extra["accountId"], "x");
        let listed: Vec<String> = store
            .list()
            .await
            .unwrap()
            .into_iter()
            .map(|info| info.provider_id)
            .collect();
        assert_eq!(listed, ["anthropic", "openai-codex", "openai"]);
        store.delete("anthropic").await.unwrap();
        assert_eq!(store.read("anthropic").await.unwrap(), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_writes_are_all_kept() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let writes = (0..8).map(|index| {
            // Separate stores share only the file, as separate processes do.
            let store = FileCredentialStore::new(&path);
            tokio::spawn(async move {
                store
                    .modify(
                        &format!("provider-{index}"),
                        Box::new(move |_| Box::pin(async move { Ok(Some(api_key(&format!("key-{index}")))) })),
                    )
                    .await
                    .unwrap();
            })
        });
        for write in writes.collect::<Vec<_>>() {
            write.await.unwrap();
        }
        assert_eq!(FileCredentialStore::new(&path).list().await.unwrap().len(), 8);
    }

    /// Answers prompts from a list.
    struct Answers(Mutex<VecDeque<String>>);

    impl AuthInteraction for Answers {
        fn prompt(&self, _prompt: AuthPrompt) -> BoxFuture<'_, Result<String, ModelsError>> {
            let answer = self.0.lock().unwrap().pop_front();
            Box::pin(async move { answer.ok_or_else(cancelled) })
        }

        fn notify(&self, _event: AuthEvent) {}
    }

    #[tokio::test]
    async fn an_api_key_login_is_stored_and_used() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FileCredentialStore::new(dir.path().join("auth.json")));
        let models = builtin_models(CreateModelsOptions {
            credentials: Some(Arc::clone(&store) as Arc<dyn CredentialStore>),
            ..CreateModelsOptions::default()
        });
        let provider = models.provider("anthropic").unwrap();
        assert_eq!(login_methods(provider.as_ref()), [AuthType::OAuth, AuthType::ApiKey]);
        let answers = Answers(Mutex::new(VecDeque::from(["sk-ant-test".to_owned()])));
        let method = login(&models, "anthropic", Some(AuthType::ApiKey), &answers)
            .await
            .unwrap();
        assert_eq!(method, AuthType::ApiKey);
        let Some(Credential::ApiKey(stored)) = store.read("anthropic").await.unwrap() else {
            panic!("an API key");
        };
        assert_eq!(stored.key.as_deref(), Some("sk-ant-test"));
        let auth = models
            .get_auth("anthropic", &AuthResolutionOverrides::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(auth.auth.api_key.as_deref(), Some("sk-ant-test"));
        assert!(login(&models, "no-such-provider", None, &answers).await.is_err());
    }

    #[test]
    fn answers_pick_options() {
        assert_eq!(pick(3, "2"), Some(1));
        assert_eq!(pick(3, " 3 "), Some(2));
        assert_eq!(pick(3, "0"), None);
        assert_eq!(pick(3, "4"), None);
        assert_eq!(pick(3, "x"), None);
    }
}
