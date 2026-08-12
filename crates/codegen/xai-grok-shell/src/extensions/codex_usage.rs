//! `x.ai/codex/rate-limits` — ChatGPT/Codex account usage windows.
//!
//! Fetches the same `GET /backend-api/wham/usage` payload used by the official
//! Codex CLI and projects the primary (normally five-hour) and secondary
//! (normally weekly) windows onto a small, stable ACP response.

use std::path::Path;

use agent_client_protocol as acp;
use serde::{Deserialize, Serialize};

use super::{ExtResult, to_raw_response};

/// Serialize usage fetches so near-expiry OAuth refreshes cannot race on the
/// rotating refresh token or the shared `auth.json.tmp` rewrite path.
static CODEX_USAGE_FETCH_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn codex_usage_http_client() -> reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                // Never forward the custom ChatGPT account header to a redirect target.
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(std::time::Duration::from_secs(15))
                .pool_idle_timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build Codex usage HTTP client")
        })
        .clone()
}

/// One Codex account usage window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexRateLimitWindow {
    /// Percentage of this window already consumed (0–100).
    pub used_percent: f64,
    /// Rolling-window duration in minutes, when reported by the backend.
    pub window_duration_mins: Option<i64>,
    /// Unix timestamp (seconds) when this window resets.
    pub resets_at: Option<i64>,
}

/// Optional purchased-credit state returned beside the subscription windows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexCredits {
    pub has_credits: bool,
    pub unlimited: bool,
    pub balance: Option<String>,
}

/// Normalized Codex account quota response exposed to the pager.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexRateLimits {
    pub plan_type: Option<String>,
    pub primary: Option<CodexRateLimitWindow>,
    pub secondary: Option<CodexRateLimitWindow>,
    pub credits: Option<CodexCredits>,
}

#[derive(Debug, Deserialize)]
struct UsagePayload {
    #[serde(default)]
    plan_type: Option<String>,
    #[serde(default)]
    rate_limit: Option<RateLimitDetails>,
    #[serde(default)]
    credits: Option<CreditsDetails>,
}

#[derive(Debug, Deserialize)]
struct RateLimitDetails {
    #[serde(default)]
    primary_window: Option<WindowSnapshot>,
    #[serde(default)]
    secondary_window: Option<WindowSnapshot>,
}

#[derive(Debug, Deserialize)]
struct WindowSnapshot {
    used_percent: f64,
    #[serde(default)]
    limit_window_seconds: Option<i64>,
    #[serde(default)]
    reset_at: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct CreditsDetails {
    #[serde(default)]
    has_credits: bool,
    #[serde(default)]
    unlimited: bool,
    #[serde(default)]
    balance: Option<String>,
}

impl From<WindowSnapshot> for CodexRateLimitWindow {
    fn from(value: WindowSnapshot) -> Self {
        Self {
            used_percent: value.used_percent.clamp(0.0, 100.0),
            window_duration_mins: value
                .limit_window_seconds
                .filter(|seconds| *seconds > 0)
                .map(|seconds| (seconds + 59) / 60),
            resets_at: value.reset_at,
        }
    }
}

impl From<UsagePayload> for CodexRateLimits {
    fn from(value: UsagePayload) -> Self {
        let (primary, secondary) = value
            .rate_limit
            .map(|limits| {
                (
                    limits.primary_window.map(Into::into),
                    limits.secondary_window.map(Into::into),
                )
            })
            .unwrap_or_default();
        Self {
            plan_type: value.plan_type,
            primary,
            secondary,
            credits: value.credits.map(|credits| CodexCredits {
                has_credits: credits.has_credits,
                unlimited: credits.unlimited,
                balance: credits.balance,
            }),
        }
    }
}

#[tracing::instrument(skip_all, fields(method = %args.method))]
pub async fn handle(args: &acp::ExtRequest) -> ExtResult {
    match args.method.as_ref() {
        "x.ai/codex/rate-limits" => {
            let limits = fetch_codex_rate_limits().await.map_err(|error| {
                acp::Error::internal_error().data(format!("Failed to fetch Codex usage: {error}"))
            })?;
            to_raw_response(&limits)
        }
        _ => Err(acp::Error::method_not_found()),
    }
}

async fn fetch_codex_rate_limits() -> anyhow::Result<CodexRateLimits> {
    fetch_codex_rate_limits_at(
        xai_grok_sampling_types::CODEX_ACCOUNT_USAGE_URL,
        &crate::auth::codex_auth_path(),
    )
    .await
}

async fn fetch_codex_rate_limits_at(
    usage_url: &str,
    auth_path: &Path,
) -> anyhow::Result<CodexRateLimits> {
    let _fetch_guard = CODEX_USAGE_FETCH_LOCK.lock().await;
    let credentials = crate::auth::load_codex_credentials_async(auth_path).await?;
    let mut request = codex_usage_http_client()
        .get(usage_url)
        .bearer_auth(&credentials.access_token)
        .header(
            reqwest::header::USER_AGENT,
            format!("grok/{}", xai_grok_version::VERSION),
        )
        .timeout(std::time::Duration::from_secs(15));
    if let Some(account_id) = credentials.account_id {
        request = request.header(
            xai_grok_sampling_types::CHATGPT_ACCOUNT_ID_HEADER,
            account_id,
        );
    }

    let response = request.send().await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!("Codex usage service returned HTTP {status}: {body}");
    }
    let payload: UsagePayload = serde_json::from_str(&body)?;
    Ok(payload.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_primary_and_weekly_windows() {
        let payload: UsagePayload = serde_json::from_value(serde_json::json!({
            "plan_type": "plus",
            "rate_limit": {
                "allowed": true,
                "limit_reached": false,
                "primary_window": {
                    "used_percent": 12,
                    "limit_window_seconds": 18000,
                    "reset_after_seconds": 120,
                    "reset_at": 1_700_000_000
                },
                "secondary_window": {
                    "used_percent": 34.5,
                    "limit_window_seconds": 604800,
                    "reset_after_seconds": 3600,
                    "reset_at": 1_700_600_000
                }
            },
            "credits": { "has_credits": true, "unlimited": false, "balance": "9.99" }
        }))
        .unwrap();

        let limits = CodexRateLimits::from(payload);
        assert_eq!(limits.plan_type.as_deref(), Some("plus"));
        assert_eq!(
            limits.primary.as_ref().unwrap().window_duration_mins,
            Some(300)
        );
        assert_eq!(limits.secondary.as_ref().unwrap().used_percent, 34.5);
        assert_eq!(
            limits.secondary.as_ref().unwrap().window_duration_mins,
            Some(10_080)
        );
        assert_eq!(
            limits.credits.as_ref().unwrap().balance.as_deref(),
            Some("9.99")
        );
    }

    #[tokio::test]
    async fn fetch_uses_codex_bearer_and_account_header() {
        use axum::{Router, http::HeaderMap, routing::get};
        use std::sync::{Arc, Mutex};

        let seen = Arc::new(Mutex::new(None::<(String, String)>));
        let seen_handler = seen.clone();
        let app = Router::new().route(
            "/usage",
            get(move |headers: HeaderMap| {
                let seen = seen_handler.clone();
                async move {
                    let authorization = headers
                        .get(reqwest::header::AUTHORIZATION)
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default()
                        .to_string();
                    let account = headers
                        .get(xai_grok_sampling_types::CHATGPT_ACCOUNT_ID_HEADER)
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default()
                        .to_string();
                    *seen.lock().unwrap() = Some((authorization, account));
                    axum::Json(serde_json::json!({
                        "plan_type": "plus",
                        "rate_limit": {
                            "primary_window": {
                                "used_percent": 7,
                                "limit_window_seconds": 18000,
                                "reset_at": 1_700_000_000
                            },
                            "secondary_window": {
                                "used_percent": 19,
                                "limit_window_seconds": 604800,
                                "reset_at": 1_700_600_000
                            }
                        }
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let dir = tempfile::tempdir().unwrap();
        let auth_path = dir.path().join("auth.json");
        let payload = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            br#"{"exp":4102444800}"#,
        );
        let token = format!("header.{payload}.signature");
        std::fs::write(
            &auth_path,
            format!(
                r#"{{"auth_mode":"chatgpt","tokens":{{"access_token":"{token}","account_id":"acct-123"}}}}"#
            ),
        )
        .unwrap();

        let limits = fetch_codex_rate_limits_at(&format!("http://{address}/usage"), &auth_path)
            .await
            .unwrap();
        assert_eq!(limits.secondary.unwrap().used_percent, 19.0);
        assert_eq!(
            seen.lock().unwrap().clone(),
            Some((format!("Bearer {token}"), "acct-123".to_string()))
        );
    }

    #[test]
    fn sparse_payload_is_valid_and_percent_is_clamped() {
        let payload: UsagePayload = serde_json::from_value(serde_json::json!({
            "plan_type": "pro",
            "rate_limit": {
                "primary_window": { "used_percent": 120, "limit_window_seconds": 1 }
            }
        }))
        .unwrap();
        let limits = CodexRateLimits::from(payload);
        assert_eq!(limits.primary.unwrap().used_percent, 100.0);
        assert!(limits.secondary.is_none());
        assert!(limits.credits.is_none());
    }
}
