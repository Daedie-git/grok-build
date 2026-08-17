//! ChatGPT/Codex subscription credentials (`~/.codex/auth.json`).
//!
//! Mirrors the official Codex CLI ChatGPT OAuth file format and refresh
//! grant. Used by built-in Codex models so Grok does not need an external
//! proxy process.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use xai_grok_sampling_types::{CODEX_OAUTH_CLIENT_ID, CODEX_OAUTH_TOKEN_URL};

/// Env override for the Codex home directory (same as the Codex CLI).
pub const CODEX_HOME_ENV: &str = "CODEX_HOME";

const REFRESH_SKEW_SECS: i64 = 120;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexAuthFile {
    #[serde(default)]
    pub auth_mode: Option<String>,
    #[serde(default)]
    pub tokens: Option<CodexTokens>,
    #[serde(default)]
    pub last_refresh: Option<String>,
    /// Present when signed in with a Platform API key instead of ChatGPT.
    #[serde(default, rename = "OPENAI_API_KEY")]
    pub openai_api_key: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexTokens {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub id_token: Option<String>,
    #[serde(default)]
    pub account_id: Option<String>,
}

/// Live credentials ready for Requests to the Codex backend.
#[derive(Debug, Clone)]
pub struct CodexCredentials {
    pub access_token: String,
    pub account_id: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, thiserror::Error)]
pub enum CodexAuthError {
    #[error("Codex auth file not found at {0}. Run `grok login --codex` or `codex login` first.")]
    MissingFile(PathBuf),
    #[error(
        "Codex auth file is not ChatGPT mode (auth_mode={0:?}). Run `codex login` with ChatGPT."
    )]
    WrongMode(Option<String>),
    #[error("Codex auth file has no access_token. Run `grok login --codex` or `codex login`.")]
    MissingToken,
    #[error("Failed to read Codex auth file: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse Codex auth file: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Codex token refresh failed: {0}")]
    Refresh(String),
}

/// Resolve `$CODEX_HOME` or `~/.codex`.
pub fn codex_home() -> PathBuf {
    if let Ok(p) = std::env::var(CODEX_HOME_ENV) {
        let trimmed = p.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".codex")
}

pub fn default_auth_path() -> PathBuf {
    codex_home().join("auth.json")
}

fn read_auth_value(path: &Path) -> Result<serde_json::Value, CodexAuthError> {
    if !path.is_file() {
        return Err(CodexAuthError::MissingFile(path.to_path_buf()));
    }
    let raw = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&raw)?)
}

pub fn read_auth_file(path: &Path) -> Result<CodexAuthFile, CodexAuthError> {
    Ok(serde_json::from_value(read_auth_value(path)?)?)
}

/// Official Codex infers a missing `auth_mode`: a present API key is
/// `apikey`, otherwise a token-shaped file is ChatGPT.
fn resolved_auth_mode(data: &serde_json::Value) -> Option<String> {
    if let Some(mode) = data.get("auth_mode").and_then(serde_json::Value::as_str) {
        return Some(mode.to_string());
    }
    let api_key = data.get("OPENAI_API_KEY");
    if api_key.is_some_and(|value| match value {
        serde_json::Value::Null => false,
        serde_json::Value::String(s) => !s.trim().is_empty(),
        _ => true,
    }) {
        return Some("apikey".into());
    }
    let access = data
        .get("tokens")
        .and_then(|tokens| tokens.get("access_token"))
        .and_then(serde_json::Value::as_str);
    if access.is_some_and(|token| !token.trim().is_empty()) {
        return Some("chatgpt".into());
    }
    None
}

fn require_chatgpt_mode(data: &serde_json::Value) -> Result<(), CodexAuthError> {
    match resolved_auth_mode(data).as_deref() {
        Some("chatgpt") => Ok(()),
        other => Err(CodexAuthError::WrongMode(other.map(str::to_owned))),
    }
}

fn tokens_from_auth_value(data: &serde_json::Value) -> Result<CodexTokens, CodexAuthError> {
    let tokens: CodexTokens = serde_json::from_value(
        data.get("tokens")
            .cloned()
            .ok_or(CodexAuthError::MissingToken)?,
    )?;
    if tokens.access_token.trim().is_empty() {
        return Err(CodexAuthError::MissingToken);
    }
    Ok(tokens)
}

fn credentials_from_auth_value(
    data: &serde_json::Value,
) -> Result<CodexCredentials, CodexAuthError> {
    require_chatgpt_mode(data)?;
    let tokens = tokens_from_auth_value(data)?;
    Ok(CodexCredentials {
        expires_at: jwt_exp(&tokens.access_token),
        access_token: tokens.access_token,
        account_id: tokens.account_id,
    })
}

/// Read the current ChatGPT credentials without locking or refreshing them.
///
/// Model construction uses this fast, non-networking path. The per-turn async
/// preflight owns refresh, so selecting a model can never block an executor on
/// an exclusive file lock or a synchronous OAuth request.
pub fn read_codex_credentials(path: &Path) -> Result<CodexCredentials, CodexAuthError> {
    credentials_from_auth_value(&read_auth_value(path)?)
}

fn acquire_auth_file_lock(path: &Path) -> Result<std::fs::File, CodexAuthError> {
    use fs2::FileExt;

    if !path.is_file() {
        return Err(CodexAuthError::MissingFile(path.to_path_buf()));
    }
    let lock_path = path.with_extension("json.lock");
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(lock_path)?;
    lock.lock_exclusive()?;
    Ok(lock)
}

fn write_auth_value(path: &Path, data: &serde_json::Value) -> Result<(), CodexAuthError> {
    let tmp = path.with_extension("json.tmp");
    let body = serde_json::to_string_pretty(data)?;
    std::fs::write(&tmp, body + "\n")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
    }
    std::fs::rename(&tmp, path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

fn apply_rotated_tokens(
    data: &mut serde_json::Value,
    tokens: &CodexTokens,
    last_refresh: String,
) -> Result<(), CodexAuthError> {
    let tokens_value = data
        .as_object_mut()
        .and_then(|object| object.get_mut("tokens"))
        .and_then(serde_json::Value::as_object_mut);
    if let Some(object) = tokens_value {
        object.insert(
            "access_token".into(),
            serde_json::Value::String(tokens.access_token.clone()),
        );
        if let Some(refresh_token) = &tokens.refresh_token {
            object.insert(
                "refresh_token".into(),
                serde_json::Value::String(refresh_token.clone()),
            );
        }
        if let Some(id_token) = &tokens.id_token {
            object.insert(
                "id_token".into(),
                serde_json::Value::String(id_token.clone()),
            );
        }
        if let Some(account_id) = &tokens.account_id {
            object.insert(
                "account_id".into(),
                serde_json::Value::String(account_id.clone()),
            );
        }
    } else {
        data["tokens"] = serde_json::to_value(tokens)?;
    }
    data["last_refresh"] = serde_json::Value::String(last_refresh);
    Ok(())
}

fn jwt_exp(token: &str) -> Option<DateTime<Utc>> {
    let payload = token.split('.').nth(1)?;
    let padded = match payload.len() % 4 {
        2 => format!("{payload}=="),
        3 => format!("{payload}="),
        _ => payload.to_string(),
    };
    let bytes = base64_url_decode(&padded)?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let exp = v.get("exp")?.as_i64()?;
    DateTime::from_timestamp(exp, 0)
}

fn base64_url_decode(s: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(s)
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(s))
        .ok()
}

fn needs_refresh(access_token: &str) -> bool {
    match jwt_exp(access_token) {
        Some(exp) => {
            let now = Utc::now();
            exp - chrono::Duration::seconds(REFRESH_SKEW_SECS) <= now
        }
        // Unknown exp: refresh proactively only if we have a refresh token path.
        None => false,
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct RefreshResponse {
    access_token: Option<String>,
    refresh_token: Option<String>,
    id_token: Option<String>,
    expires_in: Option<u64>,
    error: Option<String>,
}

/// Refresh the access token via OpenAI OAuth (blocking HTTP) at `token_url`.
///
/// Production callers use [`CODEX_OAUTH_TOKEN_URL`]; tests inject a mock server URL.
pub(crate) fn refresh_access_token_at(
    token_url: &str,
    refresh_token: &str,
) -> Result<RefreshResponse, CodexAuthError> {
    let body = serde_json::json!({
        "client_id": CODEX_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    });
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| CodexAuthError::Refresh(e.to_string()))?;
    let resp = client
        .post(token_url)
        .json(&body)
        .send()
        .map_err(|e| CodexAuthError::Refresh(e.to_string()))?;
    let status = resp.status();
    let parsed: RefreshResponse = resp
        .json()
        .map_err(|e| CodexAuthError::Refresh(e.to_string()))?;
    if !status.is_success() {
        return Err(CodexAuthError::Refresh(format!(
            "HTTP {status}: {}",
            parsed.error.unwrap_or_else(|| "unknown error".into())
        )));
    }
    if parsed.access_token.is_none() {
        return Err(CodexAuthError::Refresh(
            "response missing access_token".into(),
        ));
    }
    Ok(parsed)
}

/// Production refresh against the real OpenAI token endpoint.
pub(crate) fn refresh_access_token(refresh_token: &str) -> Result<RefreshResponse, CodexAuthError> {
    refresh_access_token_at(CODEX_OAUTH_TOKEN_URL, refresh_token)
}

/// Async refresh at an explicit `token_url` (tests inject a mock).
pub(crate) async fn refresh_access_token_async_at(
    token_url: &str,
    refresh_token: &str,
) -> Result<RefreshResponse, CodexAuthError> {
    let body = serde_json::json!({
        "client_id": CODEX_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    });
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| CodexAuthError::Refresh(e.to_string()))?;
    let resp = client
        .post(token_url)
        .json(&body)
        .send()
        .await
        .map_err(|e| CodexAuthError::Refresh(e.to_string()))?;
    let status = resp.status();
    let parsed: RefreshResponse = resp
        .json()
        .await
        .map_err(|e| CodexAuthError::Refresh(e.to_string()))?;
    if !status.is_success() {
        return Err(CodexAuthError::Refresh(format!(
            "HTTP {status}: {}",
            parsed.error.unwrap_or_else(|| "unknown error".into())
        )));
    }
    if parsed.access_token.is_none() {
        return Err(CodexAuthError::Refresh(
            "response missing access_token".into(),
        ));
    }
    Ok(parsed)
}

/// Production async refresh against the real OpenAI token endpoint.
pub(crate) async fn refresh_access_token_async(
    refresh_token: &str,
) -> Result<RefreshResponse, CodexAuthError> {
    refresh_access_token_async_at(CODEX_OAUTH_TOKEN_URL, refresh_token).await
}

/// Load ChatGPT/Codex credentials, refreshing when near expiry.
///
/// `token_url` is the OAuth token endpoint (production: [`CODEX_OAUTH_TOKEN_URL`]).
fn apply_refresh_response(tokens: &mut CodexTokens, fresh: RefreshResponse) {
    if let Some(at) = fresh.access_token {
        tokens.access_token = at;
    }
    if let Some(id) = fresh.id_token {
        tokens.id_token = Some(id);
    }
    if let Some(rt_new) = fresh.refresh_token {
        tokens.refresh_token = Some(rt_new);
    }
}

pub(crate) fn load_codex_credentials_at(
    path: &Path,
    token_url: &str,
) -> Result<CodexCredentials, CodexAuthError> {
    let _auth_lock = acquire_auth_file_lock(path)?;
    let mut data = read_auth_value(path)?;
    require_chatgpt_mode(&data)?;
    let mut tokens = tokens_from_auth_value(&data)?;

    if needs_refresh(&tokens.access_token) {
        let rt = tokens
            .refresh_token
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                CodexAuthError::Refresh(
                    "access token expired and no refresh_token is available".into(),
                )
            })?;
        tracing::info!(%token_url, "codex auth: refreshing ChatGPT OAuth token");
        let fresh = refresh_access_token_at(token_url, rt)?;
        apply_refresh_response(&mut tokens, fresh);
        apply_rotated_tokens(&mut data, &tokens, Utc::now().to_rfc3339())?;
        write_auth_value(path, &data)?;
    }

    Ok(CodexCredentials {
        expires_at: jwt_exp(&tokens.access_token),
        access_token: tokens.access_token,
        account_id: tokens.account_id,
    })
}

/// Load ChatGPT/Codex credentials from `path`, refreshing against production OAuth.
pub fn load_codex_credentials(path: &Path) -> Result<CodexCredentials, CodexAuthError> {
    load_codex_credentials_at(path, CODEX_OAUTH_TOKEN_URL)
}

/// Async load with injectable token URL (tests inject a mock server).
pub(crate) async fn load_codex_credentials_async_at(
    path: &Path,
    token_url: &str,
) -> Result<CodexCredentials, CodexAuthError> {
    load_codex_credentials_async_at_with_rejected(path, token_url, None).await
}

async fn load_codex_credentials_async_at_with_rejected(
    path: &Path,
    token_url: &str,
    rejected_access_token: Option<&str>,
) -> Result<CodexCredentials, CodexAuthError> {
    let lock_path = path.to_path_buf();
    let _auth_lock = tokio::task::spawn_blocking(move || acquire_auth_file_lock(&lock_path))
        .await
        .map_err(|error| CodexAuthError::Refresh(format!("auth lock task failed: {error}")))??;
    let mut data = read_auth_value(path)?;
    require_chatgpt_mode(&data)?;
    let mut tokens = tokens_from_auth_value(&data)?;

    let rejected_token_is_current =
        rejected_access_token.is_some_and(|rejected| rejected == tokens.access_token);
    if rejected_token_is_current || needs_refresh(&tokens.access_token) {
        let rt = tokens
            .refresh_token
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                CodexAuthError::Refresh(
                    "access token expired and no refresh_token is available".into(),
                )
            })?;
        tracing::info!(%token_url, "codex auth: refreshing ChatGPT OAuth token");
        let fresh = refresh_access_token_async_at(token_url, rt).await?;
        apply_refresh_response(&mut tokens, fresh);
        apply_rotated_tokens(&mut data, &tokens, Utc::now().to_rfc3339())?;
        write_auth_value(path, &data)?;
    }

    Ok(CodexCredentials {
        expires_at: jwt_exp(&tokens.access_token),
        access_token: tokens.access_token,
        account_id: tokens.account_id,
    })
}

/// Async load against production OAuth.
pub async fn load_codex_credentials_async(path: &Path) -> Result<CodexCredentials, CodexAuthError> {
    load_codex_credentials_async_at(path, CODEX_OAUTH_TOKEN_URL).await
}

/// Recover after the Codex backend rejects an access token.
///
/// The auth file is re-read under the cross-process lock. If another process
/// has already rotated away from `rejected_access_token`, its fresh snapshot
/// is reused rather than rotating the refresh token a second time.
pub async fn recover_rejected_codex_credentials_async(
    path: &Path,
    rejected_access_token: Option<&str>,
) -> Result<CodexCredentials, CodexAuthError> {
    load_codex_credentials_async_at_with_rejected(
        path,
        CODEX_OAUTH_TOKEN_URL,
        rejected_access_token,
    )
    .await
}

/// CLI: `grok login --codex` — validate/refresh `~/.codex/auth.json`.
pub async fn run_cli_login_codex() -> anyhow::Result<()> {
    let path = default_auth_path();
    match load_codex_credentials_async(&path).await {
        Ok(creds) => {
            println!("Signed in to ChatGPT/Codex via {}", path.display());
            if let Some(aid) = &creds.account_id {
                println!("Account id: {aid}");
            }
            if let Some(exp) = creds.expires_at {
                println!("Access token expires at: {}", exp.to_rfc3339());
            }
            println!("Use models: codex-gpt-5.6-sol, codex-gpt-5.6-luna");
            Ok(())
        }
        Err(CodexAuthError::MissingFile(p)) => {
            anyhow::bail!(
                "No Codex auth at {}.\n\
                 Sign in with the Codex CLI first:\n  \
                 codex login\n\
                 Then re-run: grok login --codex",
                p.display()
            );
        }
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn reads_fixture_auth_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let mut f = std::fs::File::create(&path).unwrap();
        // Synthetic JWT: header.payload.sig with exp far in the future
        let payload = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            br#"{"exp":4102444800}"#, // 2100-01-01
        );
        let token = format!("eyJhbGciOiJub25lIn0.{payload}.sig");
        write!(
            f,
            r#"{{
              "auth_mode": "chatgpt",
              "tokens": {{
                "access_token": "{token}",
                "refresh_token": "rt-test",
                "account_id": "acct-123"
              }}
            }}"#
        )
        .unwrap();

        let creds = load_codex_credentials(&path).unwrap();
        assert_eq!(creds.account_id.as_deref(), Some("acct-123"));
        assert_eq!(creds.access_token, token);
    }

    #[test]
    fn rejects_wrong_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        std::fs::write(
            &path,
            r#"{"auth_mode":"apikey","tokens":{"access_token":"x"}}"#,
        )
        .unwrap();
        let err = load_codex_credentials(&path).unwrap_err();
        assert!(matches!(err, CodexAuthError::WrongMode(_)));
    }

    #[test]
    fn missing_file_errors() {
        let err = load_codex_credentials(Path::new("/no/such/codex-auth.json")).unwrap_err();
        assert!(matches!(err, CodexAuthError::MissingFile(_)));
    }

    fn make_jwt(exp_unix: i64) -> String {
        let payload = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            format!(r#"{{"exp":{exp_unix}}}"#).as_bytes(),
        );
        format!("hdr.{payload}.sig")
    }

    #[test]
    fn unexpired_token_skips_refresh() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let token = make_jwt(4_102_444_800); // year 2100
        std::fs::write(
            &path,
            format!(
                r#"{{"auth_mode":"chatgpt","tokens":{{"access_token":"{token}","refresh_token":"rt","account_id":"a"}}}}"#
            ),
        )
        .unwrap();
        // Point at a non-routable token URL — if refresh were wrongly invoked, this fails.
        let c = load_codex_credentials_at(&path, "http://127.0.0.1:1/oauth/token").unwrap();
        assert_eq!(c.access_token, token);
    }

    #[test]
    fn read_only_credentials_do_not_refresh_an_expired_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let token = make_jwt(1);
        std::fs::write(
            &path,
            format!(
                r#"{{"auth_mode":"chatgpt","tokens":{{"access_token":"{token}","refresh_token":"rt","account_id":"a"}}}}"#
            ),
        )
        .unwrap();

        let credentials = read_codex_credentials(&path).unwrap();
        assert_eq!(credentials.access_token, token);
    }

    #[tokio::test]
    async fn near_expiry_token_refreshes_against_mock_and_rewrites_auth_json() {
        use axum::{Json, Router, routing::post};
        use std::sync::{Arc, Mutex};

        let hits: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
        let hits_h = hits.clone();
        let app = Router::new().route(
            "/oauth/token",
            post(move |body: Json<serde_json::Value>| {
                let hits = hits_h.clone();
                async move {
                    hits.lock().unwrap().push(body.0.clone());
                    Json(serde_json::json!({
                        "access_token": "refreshed-access-token",
                        "refresh_token": "refreshed-rt",
                        "id_token": "refreshed-id",
                        "expires_in": 3600
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        // Yield so the server accepts connections.
        tokio::task::yield_now().await;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        // exp = now + 30s is inside the 120s refresh skew → must refresh.
        let near = Utc::now().timestamp() + 30;
        let old_token = make_jwt(near);
        std::fs::write(
            &path,
            format!(
                r#"{{
                  "auth_mode": "chatgpt",
                  "tokens": {{
                    "access_token": "{old_token}",
                    "refresh_token": "rt-old",
                    "account_id": "acct-preserve-me",
                    "id_token": "id-old"
                  }}
                }}"#
            ),
        )
        .unwrap();

        let token_url = format!("http://{addr}/oauth/token");
        let (first, second) = tokio::join!(
            load_codex_credentials_async_at(&path, &token_url),
            load_codex_credentials_async_at(&path, &token_url),
        );
        let creds = first.expect("first refresh should succeed");
        let concurrent = second.expect("concurrent load should reuse refreshed credentials");

        assert_eq!(creds.access_token, "refreshed-access-token");
        assert_eq!(creds.account_id.as_deref(), Some("acct-preserve-me"));
        assert_eq!(concurrent.access_token, "refreshed-access-token");
        assert_eq!(concurrent.account_id.as_deref(), Some("acct-preserve-me"));

        let stale_recovery =
            load_codex_credentials_async_at_with_rejected(&path, &token_url, Some(&old_token))
                .await
                .expect("a stale rejected token should reuse the rotation already on disk");
        assert_eq!(stale_recovery.access_token, "refreshed-access-token");

        let hits = hits.lock().unwrap();
        assert_eq!(hits.len(), 1, "exactly one refresh POST expected");
        assert_eq!(hits[0]["grant_type"], "refresh_token");
        assert_eq!(hits[0]["refresh_token"], "rt-old");
        assert_eq!(hits[0]["client_id"], CODEX_OAUTH_CLIENT_ID);
        drop(hits);

        // auth.json rewritten with new tokens; account_id preserved.
        let disk = read_auth_file(&path).unwrap();
        let t = disk.tokens.unwrap();
        assert_eq!(t.access_token, "refreshed-access-token");
        assert_eq!(t.refresh_token.as_deref(), Some("refreshed-rt"));
        assert_eq!(t.id_token.as_deref(), Some("refreshed-id"));
        assert_eq!(t.account_id.as_deref(), Some("acct-preserve-me"));
    }

    #[test]
    fn built_in_codex_models_are_in_default_catalog() {
        use crate::agent::config::{EndpointsConfig, default_model_entries, find_model_by_id};
        use xai_grok_sampling_types::{ApiBackend, is_codex_backend_url};

        let endpoints = EndpointsConfig::default();
        let models = default_model_entries(&endpoints);
        // Context windows match the Codex CLI account catalog
        // (`~/.codex/models_cache.json` / chatgpt backend-api/codex/models).
        let expected = [
            ("codex-gpt-5.6-sol", 272_000u64, "high"),
            ("codex-gpt-5.6-luna", 272_000, "high"),
        ];
        for (id, ctx, default_effort) in expected {
            let entry = find_model_by_id(&models, id).unwrap_or_else(|| panic!("missing {id}"));
            assert!(
                is_codex_backend_url(&entry.info().base_url),
                "{id} base_url={}",
                entry.info().base_url
            );
            assert_eq!(entry.info().api_backend, ApiBackend::Responses);
            assert_eq!(
                entry.info().agent_type,
                "grok-build-plan",
                "{id} must use the Grok Build harness, not the Codex compatibility harness"
            );
            assert_eq!(
                entry.info().context_window.get(),
                ctx,
                "{id} context_window"
            );
            assert!(entry.info().supports_reasoning_effort);
            assert_eq!(
                entry.info().auto_compact_threshold_percent,
                Some(90),
                "{id} Codex soft threshold"
            );
            let expected_limit = 244_800;
            assert_eq!(
                entry
                    .info()
                    .auto_compact_token_limit
                    .map(std::num::NonZeroU64::get),
                Some(expected_limit),
                "{id} Codex CLI auto-compaction limit"
            );
            let resolved = crate::util::config::resolve_auto_compact_threshold_percent_from_tiers(
                None, // no user [model.*]
                None, // no user [session]
                entry.info().auto_compact_threshold_percent,
                None, // no remote global
            );
            assert_eq!(resolved, 90, "{id} compact threshold");
            let default = entry
                .info()
                .reasoning_efforts
                .iter()
                .find(|e| e.default)
                .map(|e| e.id.as_str());
            assert_eq!(default, Some(default_effort), "{id} default reasoning");
            assert!(
                entry.info().reasoning_efforts.len() >= 4,
                "{id} should expose Codex-listed reasoning efforts"
            );
        }

        // Codex safety metadata must not alter Grok 4.5's existing policy.
        let grok = find_model_by_id(&models, "grok-4.5").expect("missing grok-4.5");
        assert_eq!(grok.info().context_window.get(), 500_000);
        assert_eq!(grok.info().auto_compact_threshold_percent, Some(80));
        assert_eq!(grok.info().auto_compact_token_limit, None);
        assert!(!is_codex_backend_url(&grok.info().base_url));
    }

    #[test]
    fn resolve_credentials_for_codex_uses_auth_file_and_account_header() {
        use crate::agent::config::{
            EndpointsConfig, default_model_entries, find_model_by_id, resolve_credentials,
        };
        use xai_grok_sampling_types::CHATGPT_ACCOUNT_ID_HEADER;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let payload = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            br#"{"exp":4102444800}"#,
        );
        let token = format!("hdr.{payload}.sig");
        std::fs::write(
            &path,
            format!(
                r#"{{"auth_mode":"chatgpt","tokens":{{"access_token":"{token}","account_id":"acct-xyz"}}}}"#
            ),
        )
        .unwrap();

        // Point CODEX_HOME at our fixture.
        // SAFETY: test-only env mutation; single-threaded unit test.
        unsafe {
            std::env::set_var(CODEX_HOME_ENV, dir.path());
        }

        let models = default_model_entries(&EndpointsConfig::default());
        let entry = find_model_by_id(&models, "codex-gpt-5.6-sol").expect("sol model");
        let creds = resolve_credentials(entry, Some("xai-session-should-not-win"));
        assert_eq!(creds.api_key.as_deref(), Some(token.as_str()));
        assert_eq!(
            creds
                .extra_headers
                .get(CHATGPT_ACCOUNT_ID_HEADER)
                .map(String::as_str),
            Some("acct-xyz")
        );

        unsafe {
            std::env::remove_var(CODEX_HOME_ENV);
        }
    }

    #[test]
    fn review_regression_legacy_chatgpt_auth_without_auth_mode_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let token = make_jwt(4_102_444_800);
        std::fs::write(
            &path,
            format!(r#"{{"tokens":{{"access_token":"{token}","refresh_token":"rt"}}}}"#),
        )
        .unwrap();
        let creds = read_codex_credentials(&path)
            .expect("legacy token-shaped ChatGPT auth without auth_mode must be accepted");
        assert_eq!(creds.access_token, token);
    }

    #[tokio::test]
    async fn review_regression_refresh_preserves_unmodeled_official_auth_fields() {
        use axum::{Json, Router, routing::post};

        let app = Router::new().route(
            "/oauth/token",
            post(|| async {
                Json(serde_json::json!({
                    "access_token": "refreshed-access-token",
                    "refresh_token": "refreshed-rt",
                    "id_token": "refreshed-id",
                    "expires_in": 3600
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::task::yield_now().await;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let near = Utc::now().timestamp() + 30;
        let old_token = make_jwt(near);
        std::fs::write(
            &path,
            format!(
                r#"{{
                  "auth_mode": "chatgpt",
                  "tokens": {{
                    "access_token": "{old_token}",
                    "refresh_token": "rt-old",
                    "account_id": "acct-preserve-me"
                  }},
                  "agent_identity": "workspace-preserve-me",
                  "personal_access_token": "pat-preserve-me"
                }}"#
            ),
        )
        .unwrap();

        let token_url = format!("http://{addr}/oauth/token");
        load_codex_credentials_async_at(&path, &token_url)
            .await
            .expect("refresh should succeed");
        let disk: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            disk.get("agent_identity")
                .and_then(serde_json::Value::as_str),
            Some("workspace-preserve-me")
        );
        assert_eq!(
            disk.get("personal_access_token")
                .and_then(serde_json::Value::as_str),
            Some("pat-preserve-me")
        );
    }

    #[tokio::test]
    async fn review_regression_refresh_reports_failure_when_rotated_token_cannot_be_persisted() {
        use axum::{Json, Router, routing::post};

        let app = Router::new().route(
            "/oauth/token",
            post(|| async {
                Json(serde_json::json!({
                    "access_token": "refreshed-access-token",
                    "refresh_token": "refreshed-rt",
                    "expires_in": 3600
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::task::yield_now().await;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let near = Utc::now().timestamp() + 30;
        let old_token = make_jwt(near);
        std::fs::write(
            &path,
            format!(
                r#"{{"auth_mode":"chatgpt","tokens":{{"access_token":"{old_token}","refresh_token":"rt-old"}}}}"#
            ),
        )
        .unwrap();
        // A directory at the tmp path makes the atomic write fail.
        std::fs::create_dir(path.with_extension("json.tmp")).unwrap();

        let token_url = format!("http://{addr}/oauth/token");
        let err = load_codex_credentials_async_at(&path, &token_url)
            .await
            .expect_err("a refreshed token is not usable until its rotation is durable");
        assert!(
            matches!(err, CodexAuthError::Io(_) | CodexAuthError::Refresh(_)),
            "persist failure must surface, got {err:?}"
        );
    }
}
