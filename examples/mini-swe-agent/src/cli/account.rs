//! `mini-extra login`, `logout`, `models` and `config`: credentials, the models they make
//! available, and the default model in the global config.

use delta_ai::Models;
use delta_ai::auth::{AuthResolutionOverrides, AuthType};
use delta_ai::model::Model;

use crate::auth::{TerminalInteraction, login, login_methods, pick};
use crate::model::build_models;

/// The built-in providers and the global config's, with the credentials in `auth.json`.
pub(super) fn models() -> Result<Models, String> {
    let global = crate::home::global_config().map_err(|error| error.to_string())?;
    build_models(&global["providers"], Some(super::credential_store())).map_err(|error| error.to_string())
}

/// Asks for a line on the terminal, with `default` filled in; `None` at the end of input.
fn ask(question: &str, default: Option<&str>) -> Option<String> {
    let mut editor = rustyline::DefaultEditor::new().ok()?;
    let line = match default.filter(|default| !default.is_empty()) {
        Some(default) => editor.readline_with_initial(question, (default, "")),
        None => editor.readline(question),
    };
    line.ok().map(|line| line.trim().to_owned())
}

async fn has_auth(models: &Models, provider: &str) -> bool {
    matches!(
        models.get_auth(provider, &AuthResolutionOverrides::default()).await,
        Ok(Some(_))
    )
}

/// `mini-extra login [provider]`.
pub(super) async fn login_command(provider: Option<String>, method: Option<AuthType>) -> Result<(), String> {
    let models = models()?;
    let provider = match provider {
        Some(provider) => provider,
        None => {
            let choices: Vec<_> = models
                .providers()
                .into_iter()
                .filter(|provider| !login_methods(provider.as_ref()).is_empty())
                .collect();
            println!("Providers you can log in to:");
            for (index, provider) in choices.iter().enumerate() {
                println!("  {}. {} ({})", index + 1, provider.id(), provider.name());
            }
            let answer = ask("Provider (number or id): ", None).ok_or("No provider chosen")?;
            match pick(choices.len(), &answer) {
                Some(index) => choices[index].id().to_owned(),
                None => answer,
            }
        }
    };
    let method = login(&models, &provider, method, &TerminalInteraction).await?;
    let kind = match method {
        AuthType::OAuth => "OAuth credentials",
        AuthType::ApiKey => "API key",
    };
    println!(
        "Saved the {kind} for {provider} to {}",
        crate::home::auth_file().display()
    );
    Ok(())
}

/// `mini-extra logout <provider>`.
pub(super) async fn logout_command(provider: &str) -> Result<(), String> {
    models()?.logout(provider).await.map_err(|error| error.to_string())?;
    println!("Removed the credentials for {provider}");
    Ok(())
}

fn price(model: &Model) -> String {
    if crate::model::has_price(model) {
        format!("${}/${} per Mtok", model.cost.input, model.cost.output)
    } else {
        "no price".to_owned()
    }
}

/// `mini-extra models [provider] [--all]`: the models whose provider has credentials, or
/// every model.
pub(super) async fn models_command(provider: Option<&str>, all: bool) -> Result<(), String> {
    let models = models()?;
    if let Some(provider) = provider
        && models.provider(provider).is_none()
    {
        return Err(format!("Unknown provider: {provider}"));
    }
    let listed = if all {
        models.get_models(provider)
    } else {
        models
            .get_available(provider)
            .await
            .map_err(|error| error.to_string())?
    };
    if listed.is_empty() {
        println!(
            "No models available. Log in with `mini-extra login`, or set the provider's API key variable; `--all` lists every model."
        );
        return Ok(());
    }
    let width = listed
        .iter()
        .map(|model| model.provider.len() + model.id.len() + 1)
        .max()
        .unwrap_or_default();
    for model in &listed {
        let reference = format!("{}/{}", model.provider, model.id);
        println!(
            "{reference:<width$}  {}  ({}, {} tokens)",
            model.name,
            price(model),
            model.context_window
        );
    }
    Ok(())
}

/// `mini-extra config setup`: a provider, a login when it has no credentials, and a model,
/// saved as the global config's default.
pub(super) async fn setup_command() -> Result<(), String> {
    let models = models()?;
    let global = crate::home::global_config().map_err(|error| error.to_string())?;
    let mut providers = models.providers();
    providers.sort_by(|a, b| a.id().cmp(b.id()));
    println!("Providers (* has credentials):");
    for provider in &providers {
        let mark = if has_auth(&models, provider.id()).await {
            "*"
        } else {
            " "
        };
        println!(" {mark} {} ({})", provider.id(), provider.name());
    }
    let provider = ask("Provider: ", global["model"]["provider"].as_str()).ok_or("No provider chosen")?;
    let Some(chosen) = models.provider(&provider) else {
        return Err(format!("Unknown provider: {provider}"));
    };
    if !has_auth(&models, &provider).await {
        if login_methods(chosen.as_ref()).is_empty() {
            println!(
                "{} has no credentials; it reads them from its environment variables.",
                chosen.name()
            );
        } else if ask(&format!("Log in to {} now? [Y/n] ", chosen.name()), None)
            .is_some_and(|answer| !answer.eq_ignore_ascii_case("n"))
        {
            login(&models, &provider, None, &TerminalInteraction).await?;
            println!("Saved the credentials to {}", crate::home::auth_file().display());
        }
    }
    let catalog = models.get_models(Some(&provider));
    println!("Models of {provider}:");
    for (index, model) in catalog.iter().enumerate() {
        println!("  {}. {} ({}, {})", index + 1, model.id, model.name, price(model));
    }
    let current = global["model"]["id"]
        .as_str()
        .filter(|_| global["model"]["provider"] == provider.as_str());
    let answer = ask("Model (number or id): ", current).ok_or("No model chosen")?;
    let id = match pick(catalog.len(), &answer) {
        Some(index) => catalog[index].id.clone(),
        None => answer,
    };
    let path = crate::home::set_default_model(&provider, &id)?;
    println!("Default model {provider}/{id} saved to {}", path.display());
    Ok(())
}

/// `mini-extra config show`: the home directory, the global config and the providers with
/// stored credentials.
pub(super) async fn show_command() -> Result<(), String> {
    println!("Home: {}", crate::home::home_dir().display());
    let path = crate::home::config_file();
    match std::fs::read_to_string(&path) {
        Ok(text) => println!("\n{}:\n{}", path.display(), text.trim_end()),
        Err(_) => println!("\n{}: (none)", path.display()),
    }
    let store = crate::auth::FileCredentialStore::new(crate::home::auth_file());
    let stored = delta_ai::CredentialStore::list(&store)
        .await
        .map_err(|error| error.to_string())?;
    println!("\n{}:", store.path().display());
    if stored.is_empty() {
        println!("  (no credentials)");
    }
    for info in stored {
        let kind = match info.auth_type {
            AuthType::OAuth => "oauth",
            AuthType::ApiKey => "api_key",
        };
        println!("  {} ({kind})", info.provider_id);
    }
    Ok(())
}
