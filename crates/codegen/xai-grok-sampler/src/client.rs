//! HTTP client for the xAI sampling APIs.
//!
//! Owns the `reqwest::Client`, default request headers, and per-method
//! defaults. Talks to three backend shapes:
//!
//! * Chat Completions (`/chat/completions`)
//! * Responses API (`/responses`)
//! * Anthropic Messages API (`/messages`)
//!
//! All trace-upload and URL-based header injection is intentionally
//! *not* here. The session is responsible for putting any per-request
//! headers (proxy auth, OTel context, etc.)
//! into [`SamplerConfig::extra_headers`] before constructing the client.

use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use indexmap::IndexMap;
use reqwest::header::{
    ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue, USER_AGENT,
};
use serde::{Deserialize, Serialize};

use xai_grok_sampling_types::error::{try_parse_stream_error, user_facing_api_error_message};

use xai_grok_sampling_types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CodexCompactRequest,
    CodexCompactResponse, ConversationRequest, ConversationResponse, CreateResponseWrapper,
    DOOM_LOOP_CHECK_HEADER, MessagesRequestWrapper, ResponseModelMetadata, Result, SamplingError,
    TurnRoutingState, build_messages_request,
    conversation_request_to_codex_compact_request_for_origin, is_check_event, messages, rs,
};

use crate::config::{AuthScheme, OriginClientInfo, SamplerConfig};

// Re-export ApiBackend from the shared types crate for downstream callers.
pub use xai_grok_sampling_types::ApiBackend;

/// Process-level fallback for the `x-grok-client-identifier` header.
const DEFAULT_CLIENT_IDENTIFIER: &str = "grok-shell";

/// Product identifier baked into User-Agent strings.
const AGENT_PRODUCT: &str = "grok-shell";
const ANTHROPIC_DEFAULT_MAX_TOKENS: u32 = 128_000;

/// Per-request `x-grok-*` headers. Optional fields are skipped when empty/`None`.
struct GrokRequestHeaders<'a> {
    conv_id: &'a str,
    req_id: &'a str,
    model_id: &'a str,
    session_id: &'a str,
    turn_idx: Option<&'a str>,
    agent_id: &'a str,
    deployment_id: Option<&'a str>,
    user_id: Option<&'a str>,
}

impl GrokRequestHeaders<'_> {
    fn apply(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut b = builder
            .header("x-grok-conv-id", self.conv_id)
            .header("x-grok-req-id", self.req_id)
            .header("x-grok-model-override", self.model_id)
            .header("x-grok-session-id", self.session_id)
            .header("x-grok-agent-id", self.agent_id);
        if let Some(idx) = self.turn_idx {
            b = b.header("x-grok-turn-idx", idx);
        }
        if let Some(id) = self.deployment_id.filter(|s| !s.is_empty()) {
            b = b.header("x-grok-deployment-id", id);
        }
        if let Some(id) = self.user_id.filter(|s| !s.is_empty()) {
            b = b.header("x-grok-user-id", id);
        }
        b
    }
}

/// Parse the `Retry-After` response header as delta-seconds.
/// Our inference backends only emit integer seconds (never HTTP-date),
/// so we only handle that form. HTTP-dates silently return `None` and
/// the caller falls back to exponential backoff.
/// Capped at 120s to prevent absurdly long sleeps from a misbehaving upstream.
/// Deserialize a Responses API SSE event, with a fallback for xAI-specific
/// tool types (e.g., `x_search`) that `async_openai` can't parse.
///
/// The API echoes the request's `tools` array in `ResponseCompleted` and
/// `ResponseCreated` events. If we sent `{"type": "x_search"}`, the response
/// includes it, and `rs::Tool` deserialization fails. On failure, we strip
/// unrecognized tools from the raw JSON and retry.
///
/// On `response.completed` / `response.incomplete`, this also rewrites
/// `response.usage.total_tokens` in place to the live context length
/// (`context_details.input_tokens + context_details.output_tokens`)
/// when the API emits the xAI-specific `context_details` field.
/// Async-openai's typed `ResponseUsage` doesn't model `context_details`,
/// so we peek the raw JSON for it. The cumulative `input_tokens` /
/// `output_tokens` / `cached_tokens` continue to flow from the typed
/// `ResponseUsage` unchanged so billing telemetry stays correct. When
/// the API doesn't emit `context_details` (older deployments) `total_tokens`
/// passes through unchanged.
#[derive(Deserialize)]
struct RawOutputItemEvent {
    output_index: u32,
    item: serde_json::Value,
}

fn captured_output_item(
    output_index: u32,
    mut item: serde_json::Value,
    preserve_codex_metadata: bool,
) -> Result<xai_grok_sampling_types::CapturedResponseOutputItem> {
    let metadata = item
        .as_object_mut()
        .and_then(|object| object.remove("internal_chat_message_metadata_passthrough"))
        .map(serde_json::from_value)
        .transpose()
        .map_err(SamplingError::Serialization)?;
    let item_type = item
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("<missing>")
        .to_string();
    let value = if item_type == "function_call_output" {
        xai_grok_sampling_types::CapturedResponseOutputItemValue::FunctionCallOutput(
            serde_json::from_value(item).map_err(SamplingError::Serialization)?,
        )
    } else {
        let typed = serde_json::from_value(item).map_err(|error| {
            if preserve_codex_metadata && metadata.is_some() {
                SamplingError::serialization_message(format!(
                    "unsupported metadata-bearing Codex Responses output item `{item_type}`: {error}"
                ))
            } else {
                SamplingError::Serialization(error)
            }
        })?;
        xai_grok_sampling_types::CapturedResponseOutputItemValue::Typed(typed)
    };
    let captured = xai_grok_sampling_types::CapturedResponseOutputItem {
        output_index,
        value,
        internal_chat_message_metadata_passthrough: preserve_codex_metadata
            .then_some(metadata)
            .flatten(),
        metadata_origin: None,
    };
    if preserve_codex_metadata
        && captured
            .internal_chat_message_metadata_passthrough
            .is_some()
        && captured.kind().is_none()
    {
        return Err(SamplingError::serialization_message(format!(
            "unsupported metadata-bearing Codex Responses output item `{item_type}` cannot be replayed exactly"
        )));
    }
    Ok(captured)
}

fn capture_terminal_output(
    value: &mut serde_json::Value,
    preserve_codex_metadata: bool,
) -> Result<Option<Vec<xai_grok_sampling_types::CapturedResponseOutputItem>>> {
    let Some(output) = value
        .pointer_mut("/response/output")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return Ok(None);
    };
    let mut captured = Vec::with_capacity(output.len());
    for (output_index, item) in output.iter_mut().enumerate() {
        let decoded = captured_output_item(
            u32::try_from(output_index).map_err(|_| {
                SamplingError::serialization_message("Responses output index exceeds u32")
            })?,
            item.clone(),
            preserve_codex_metadata,
        )?;
        if let Some(object) = item.as_object_mut() {
            object.remove("internal_chat_message_metadata_passthrough");
        }
        captured.push(decoded);
    }
    // async-openai's response-side union omits function_call_output. The raw
    // captured list remains authoritative for conversion, while the typed
    // response retains all other response-level fields.
    output.retain(|item| {
        item.get("type").and_then(serde_json::Value::as_str) != Some("function_call_output")
    });
    Ok(Some(captured))
}

fn deserialize_response_event_with_metadata(
    data: &str,
    preserve_codex_metadata: bool,
) -> Result<xai_grok_sampling_types::DecodedResponseStreamEvent> {
    let mut value =
        serde_json::from_str::<serde_json::Value>(data).map_err(SamplingError::Serialization)?;
    match value.get("type").and_then(serde_json::Value::as_str) {
        Some("response.output_item.added") | Some("response.output_item.done") => {
            let done = value.get("type").and_then(serde_json::Value::as_str)
                == Some("response.output_item.done");
            let raw: RawOutputItemEvent =
                serde_json::from_value(value).map_err(SamplingError::Serialization)?;
            let item = captured_output_item(raw.output_index, raw.item, preserve_codex_metadata)?;
            return Ok(if done {
                xai_grok_sampling_types::DecodedResponseStreamEvent::OutputItemDone(item)
            } else {
                xai_grok_sampling_types::DecodedResponseStreamEvent::OutputItemAdded(item)
            });
        }
        _ => {}
    }

    let terminal_output = capture_terminal_output(&mut value, preserve_codex_metadata)?;
    // Strip tools that async_openai's rs::Tool can't deserialize (e.g.
    // xAI-specific `x_search`) without maintaining a hardcoded allowlist.
    if let Some(tools) = value
        .pointer_mut("/response/tools")
        .and_then(serde_json::Value::as_array_mut)
    {
        tools.retain(|tool| serde_json::from_value::<rs::Tool>(tool.clone()).is_ok());
    }
    let mut event = serde_json::from_value::<rs::ResponseStreamEvent>(value).map_err(|error| {
        tracing::error!(%error, raw_data = %data, "Failed to deserialize ResponseStreamEvent from stream");
        SamplingError::Serialization(error)
    })?;
    apply_terminal_event_overrides(&mut event, data);
    Ok(xai_grok_sampling_types::DecodedResponseStreamEvent::Event {
        event,
        terminal_output,
    })
}

#[cfg(test)]
fn deserialize_response_event(data: &str) -> Result<rs::ResponseStreamEvent> {
    match deserialize_response_event_with_metadata(data, false)? {
        xai_grok_sampling_types::DecodedResponseStreamEvent::Event { event, .. } => Ok(event),
        _ => Err(SamplingError::serialization_message(
            "test helper expected a non-output-item Responses event",
        )),
    }
}

fn deserialize_unary_response(
    bytes: &[u8],
    preserve_codex_metadata: bool,
) -> Result<xai_grok_sampling_types::DecodedResponse> {
    let mut value =
        serde_json::from_slice::<serde_json::Value>(bytes).map_err(SamplingError::Serialization)?;
    let output = value
        .get_mut("output")
        .and_then(serde_json::Value::as_array_mut)
        .map(|items| {
            items
                .iter_mut()
                .enumerate()
                .map(|(output_index, item)| {
                    let captured = captured_output_item(
                        u32::try_from(output_index).map_err(|_| {
                            SamplingError::serialization_message(
                                "Responses output index exceeds u32",
                            )
                        })?,
                        item.clone(),
                        preserve_codex_metadata,
                    )?;
                    if let Some(object) = item.as_object_mut() {
                        object.remove("internal_chat_message_metadata_passthrough");
                    }
                    Ok(captured)
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    if let Some(items) = value
        .get_mut("output")
        .and_then(serde_json::Value::as_array_mut)
    {
        items.retain(|item| {
            item.get("type").and_then(serde_json::Value::as_str) != Some("function_call_output")
        });
    }
    let response = serde_json::from_value(value).map_err(SamplingError::Serialization)?;
    Ok(xai_grok_sampling_types::DecodedResponse { response, output })
}

/// On terminal Responses API events (`response.completed` /
/// `response.incomplete`), rewrite `response.usage.total_tokens` to the
/// live context length when the wire includes
/// `response.usage.context_details.{input_tokens, output_tokens}`.
///
/// `total_tokens` drives the CLI's `/context` bar, the auto-compact
/// threshold, and `meta.totalTokens` on persisted sessions. Under
/// server-side multi-turn loops (e.g. `web_search`, `x_search`) the
/// wire's cumulative total inflates as the loop runs; `context_details`
/// reports the final turn's prompt + output tokens — the real live
/// context the model is sitting in. Billing fields
/// (`input_tokens`, `output_tokens`, `input_tokens_details.cached_tokens`,
/// `output_tokens_details.reasoning_tokens`) stay on the cumulative
/// wire values so telemetry is unaffected.
///
/// No-op when:
/// - the event is not terminal,
/// - `response.usage` is `None`,
/// - `context_details` is absent (older backends / non-loop responses),
/// - or either of `context_details.{input_tokens, output_tokens}` is
///   missing — we don't guess the missing half.
fn apply_terminal_event_overrides(event: &mut rs::ResponseStreamEvent, data: &str) {
    let response = match event {
        rs::ResponseStreamEvent::ResponseCompleted(e) => &mut e.response,
        rs::ResponseStreamEvent::ResponseIncomplete(e) => &mut e.response,
        _ => return,
    };
    // Re-parse for fields async_openai's types omit (context total, cost ticks).
    let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
        return;
    };
    // Stash cost ticks in metadata for stream_responses.
    if let Some(ticks) = xai_grok_sampling_types::reported_cost_ticks(
        value
            .pointer("/response/usage/cost_in_usd_ticks")
            .and_then(|v| v.as_i64()),
    ) {
        response
            .metadata
            .get_or_insert_with(Default::default)
            .insert(COST_USD_TICKS_METADATA_KEY.to_owned(), ticks.to_string());
    }
    let Some(usage) = response.usage.as_mut() else {
        return;
    };
    let Some(total) = extract_context_total(&value) else {
        return;
    };
    usage.total_tokens = total;
}

/// Metadata key for cost ticks past typed Response events.
pub(crate) const COST_USD_TICKS_METADATA_KEY: &str = "xai.cost_usd_ticks";

/// Read `response.usage.context_details.{input_tokens, output_tokens}`
/// from the parsed terminal-event JSON and return their sum. Returns `None`
/// if either field is missing or out of `u32` range.
fn extract_context_total(value: &serde_json::Value) -> Option<u32> {
    let cd = value.pointer("/response/usage/context_details")?;
    let i = u32::try_from(cd.get("input_tokens")?.as_u64()?).ok()?;
    let o = u32::try_from(cd.get("output_tokens")?.as_u64()?).ok()?;
    Some(i.saturating_add(o))
}

/// Record `success=false` + `error` on the active inference span when a stream
/// request fails before any response (transport/connect/TLS errors). Without
/// this the `#[instrument]` span closes with both fields Empty, so an outage
/// shows zero `success=false` and error-rate alerts never fire.
fn record_stream_request_failure(err: &reqwest::Error) {
    let span = tracing::Span::current();
    span.record("success", false);
    span.record("error", err.to_string().as_str());
}

fn extract_retry_after(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .map(|s| s.min(120))
}

fn extract_should_retry(headers: &reqwest::header::HeaderMap) -> Option<bool> {
    headers
        .get("x-should-retry")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| {
            if s.eq_ignore_ascii_case("true") {
                Some(true)
            } else if s.eq_ignore_ascii_case("false") {
                Some(false)
            } else {
                None // unknown value — treat as absent
            }
        })
}

fn extract_model_metadata(headers: &reqwest::header::HeaderMap) -> Option<ResponseModelMetadata> {
    let context_window = headers
        .get("x-grok-context-window")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let max_completion_tokens = headers
        .get("x-grok-max-completion-tokens")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok());

    let models_etag = headers
        .get("x-models-etag")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    if context_window.is_some() || max_completion_tokens.is_some() || models_etag.is_some() {
        Some(ResponseModelMetadata {
            context_window,
            max_completion_tokens,
            models_etag,
        })
    } else {
        None
    }
}

/// Wrapper for streaming chat completion requests that adds `stream` and
/// `stream_options` fields without modifying the original `ChatCompletionRequest`.
///
/// Uses `#[serde(flatten)]` to inline all fields from the inner request,
/// allowing single-pass serialization instead of the previous two-pass
/// approach (serialize to `Value`, mutate, serialize to bytes).
#[derive(Serialize)]
struct StreamingChatRequest<'a> {
    #[serde(flatten)]
    inner: &'a ChatCompletionRequest,
    stream: bool,
    stream_options: StreamOptions,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// Resolve `env_http_headers` (`header -> env var`) into `headers` via `getenv`, skipping unset/blank/invalid entries and trimming values.
fn apply_env_http_headers(
    env_http_headers: &IndexMap<String, String>,
    getenv: impl Fn(&str) -> Option<String>,
    headers: &mut HeaderMap,
) {
    for (key, env_var) in env_http_headers {
        let Some(value) = getenv(env_var) else {
            continue;
        };
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        let (Ok(name), Ok(header_value)) = (
            HeaderName::try_from(key.as_str()),
            HeaderValue::from_str(value),
        ) else {
            tracing::warn!(
                header = %key,
                env_var = %env_var,
                "skipping env_http_header with an invalid header name or value"
            );
            continue;
        };
        headers.insert(name, header_value);
    }
}

fn apply_config_http_headers(
    extra_headers: &IndexMap<String, String>,
    env_http_headers: &IndexMap<String, String>,
    getenv: impl Fn(&str) -> Option<String>,
    headers: &mut HeaderMap,
) -> Result<()> {
    for (key, value) in extra_headers {
        let header_name = HeaderName::try_from(key.as_str())
            .map_err(|_| SamplingError::InvalidConfiguration("Invalid extra header name"))?;
        let header_value = HeaderValue::from_str(value)
            .map_err(|_| SamplingError::InvalidConfiguration("Invalid extra header value"))?;
        headers.insert(header_name, header_value);
    }
    apply_env_http_headers(env_http_headers, getenv, headers);
    Ok(())
}

fn sampling_identity_from_effective_headers(
    protocol: &xai_grok_sampling_types::ProtocolIdentity,
    base_url: &str,
    model: &str,
    headers: &HeaderMap,
) -> xai_grok_sampling_types::SamplingIdentity {
    let chatgpt_account_id = headers
        .get(xai_grok_sampling_types::CHATGPT_ACCOUNT_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    xai_grok_sampling_types::SamplingIdentity::new_for_protocol(
        protocol,
        base_url,
        model,
        chatgpt_account_id,
    )
}

/// Resolve the runtime sampling identity exactly as the wire client does.
///
/// Static headers are installed first and environment-backed headers are then
/// applied with [`apply_env_http_headers`] semantics. Environment values remain
/// runtime-only and are represented in the result solely by the account-bound
/// identity metadata required for safe native-history replay.
pub fn resolve_runtime_sampling_identity(
    api_backend: ApiBackend,
    base_url: &str,
    model: &str,
    extra_headers: &IndexMap<String, String>,
    env_http_headers: &IndexMap<String, String>,
) -> Result<xai_grok_sampling_types::SamplingIdentity> {
    resolve_runtime_sampling_identity_for_provider(
        None,
        api_backend,
        base_url,
        model,
        extra_headers,
        env_http_headers,
    )
}

pub fn resolve_runtime_sampling_identity_for_provider(
    provider_id: Option<xai_grok_sampling_types::ProviderId>,
    api_backend: ApiBackend,
    base_url: &str,
    model: &str,
    extra_headers: &IndexMap<String, String>,
    env_http_headers: &IndexMap<String, String>,
) -> Result<xai_grok_sampling_types::SamplingIdentity> {
    resolve_runtime_sampling_identity_with_getenv(
        provider_id,
        api_backend,
        base_url,
        model,
        extra_headers,
        env_http_headers,
        |var| std::env::var(var).ok(),
    )
}

fn resolve_runtime_sampling_identity_with_getenv(
    provider_id: Option<xai_grok_sampling_types::ProviderId>,
    api_backend: ApiBackend,
    base_url: &str,
    model: &str,
    extra_headers: &IndexMap<String, String>,
    env_http_headers: &IndexMap<String, String>,
    getenv: impl Fn(&str) -> Option<String>,
) -> Result<xai_grok_sampling_types::SamplingIdentity> {
    let mut headers = HeaderMap::new();
    apply_config_http_headers(extra_headers, env_http_headers, getenv, &mut headers)?;
    let protocol =
        xai_grok_sampling_types::ProtocolIdentity::resolve(provider_id, api_backend, base_url);
    Ok(sampling_identity_from_effective_headers(
        &protocol, base_url, model, &headers,
    ))
}

/// HTTP client for sampling. Cheap to clone; carries an `Arc`-backed
/// `reqwest::Client` and the default headers/request-defaults computed from a
/// [`SamplerConfig`] at construction time.
#[derive(Clone)]
pub struct SamplingClient {
    http: reqwest::Client,
    default_headers: HeaderMap,
    base_url: String,
    chatgpt_account_id: Option<String>,
    provider: xai_grok_sampling_types::ResolvedProvider,
    responses_wire: crate::responses_wire::ResponsesWireAdapter,
    defaults: ClientDefaults,
    /// Optional 401-attribution hook. The shell wires this to emit a
    /// structured event at every UNAUTHORIZED arm so 401s can be
    /// bucketed by stale-snapshot vs. live-token-rejected. `None` for
    /// sampler-only callers and tests.
    attribution_callback: Option<crate::attribution::SharedAttributionCallback>,
    /// Per-request bearer override. See `SamplerConfig::bearer_resolver`.
    bearer_resolver: Option<crate::config::SharedBearerResolver>,
    /// Per-request header injection (OTel traceparent).
    header_injector: Option<crate::config::SharedHeaderInjector>,
    /// Endpoint URL builder, resolved once from `base_url` + `query_params`.
    endpoint: EndpointTemplate,
}

impl std::fmt::Debug for SamplingClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplingClient")
            .field("base_url", &self.base_url)
            .field("provider", &self.provider)
            .field("responses_wire", &self.responses_wire)
            .field("defaults", &self.defaults)
            .field(
                "has_attribution_callback",
                &self.attribution_callback.is_some(),
            )
            .field("has_bearer_resolver", &self.bearer_resolver.is_some())
            .finish()
    }
}

#[derive(Clone, Debug, Default)]
struct ClientDefaults {
    model: String,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    api_backend: ApiBackend,
    auth_scheme: AuthScheme,
    stream_tool_calls: bool,
    doom_loop_recovery: Option<xai_grok_sampling_types::DoomLoopRecoveryPolicy>,
}

/// Endpoint URL builder, resolved once at client construction so each request
/// only appends its path.
#[derive(Clone, Debug)]
enum EndpointTemplate {
    /// No query params and no query on the base URL (or an unparseable base):
    /// append the path to the base verbatim.
    Plain(String),
    /// Query params configured: `{prefix}/{path}{suffix}`. `suffix` starts with
    /// `?` and folds any base-URL params, with a configured key winning over the
    /// same key in `base_url` (percent-encoded, no duplicates).
    WithQuery { prefix: String, suffix: String },
}

impl EndpointTemplate {
    fn new(base_url: &str, query_params: &IndexMap<String, String>) -> Self {
        let base = base_url.trim_end_matches('/').to_string();
        // The fast path is safe only when there is nothing to fold: no configured
        // params and no query already on the base (which would otherwise land
        // before the appended path).
        if query_params.is_empty() && !base.contains('?') {
            return Self::Plain(base);
        }
        let mut url = match reqwest::Url::parse(&base) {
            Ok(url) => url,
            Err(error) => {
                tracing::warn!(
                    url = %base,
                    %error,
                    "failed to parse base URL for endpoint; sending without folded query"
                );
                return Self::Plain(base);
            }
        };
        let overridden: std::collections::HashSet<&str> =
            query_params.keys().map(String::as_str).collect();
        let kept: Vec<(String, String)> = url
            .query_pairs()
            .filter(|(k, _)| !overridden.contains(k.as_ref()))
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        let prefix = {
            let mut prefix_url = url.clone();
            prefix_url.set_query(None);
            prefix_url.as_str().trim_end_matches('/').to_string()
        };
        {
            let mut pairs = url.query_pairs_mut();
            pairs.clear();
            for (key, value) in &kept {
                pairs.append_pair(key, value);
            }
            for (key, value) in query_params {
                pairs.append_pair(key, value);
            }
        }
        let suffix = url.query().map(|q| format!("?{q}")).unwrap_or_default();
        Self::WithQuery { prefix, suffix }
    }

    fn url_for_path(&self, path: &str) -> String {
        let path = path.trim_start_matches('/');
        match self {
            Self::Plain(base) => format!("{base}/{path}"),
            Self::WithQuery { prefix, suffix } => format!("{prefix}/{path}{suffix}"),
        }
    }
}

// =============================================================================
// User-Agent helpers
// =============================================================================

#[derive(Clone, Debug, Eq, PartialEq)]
struct PlatformInfo {
    os: String,
    arch: String,
}

impl PlatformInfo {
    fn current() -> Self {
        let os = match std::env::consts::OS {
            "macos" => "macos",
            "windows" => "windows",
            other => other,
        }
        .to_string();

        let arch = match std::env::consts::ARCH {
            "arm64" => "aarch64",
            "x86_64" => "x86_64",
            other => other,
        }
        .to_string();

        Self { os, arch }
    }
}

fn agent_version() -> String {
    xai_grok_version::VERSION.to_string()
}

/// Render a User-Agent string for the given origin client.
///
/// Mirrors the shell's `user_agent_string_for` but uses sampler-local
/// constants. The session typically owns the canonical User-Agent
/// rendering for process-wide HTTP clients; this helper is for
/// per-session sampling clients that want to override it.
pub fn user_agent_string_for(origin: &OriginClientInfo) -> String {
    let agent_version = agent_version();
    let platform = PlatformInfo::current();

    if origin.product == AGENT_PRODUCT && origin.version.as_deref() == Some(agent_version.as_str())
    {
        return format!(
            "{}/{} ({}; {})",
            AGENT_PRODUCT, agent_version, platform.os, platform.arch
        );
    }

    match origin.version.as_deref() {
        Some(origin_version) => format!(
            "{}/{} {}/{} ({}; {})",
            origin.product,
            origin_version,
            AGENT_PRODUCT,
            agent_version,
            platform.os,
            platform.arch
        ),
        None => format!(
            "{} {}/{} ({}; {})",
            origin.product, AGENT_PRODUCT, agent_version, platform.os, platform.arch
        ),
    }
}

// =============================================================================
// SamplingClient
// =============================================================================

impl SamplingClient {
    /// Construct a sampling client from a [`SamplerConfig`].
    ///
    /// Grabs the process-wide shared `reqwest::Client` (HTTP/2 by
    /// default, HTTP/1.1 when `config.force_http1` is set) and
    /// pre-computes the default request headers. This does not perform
    /// any network I/O.
    pub fn new(config: SamplerConfig) -> Result<Self> {
        let provider = xai_grok_sampling_types::resolve_provider(
            config.provider_id,
            config.api_backend.clone(),
            &config.base_url,
        );
        let responses_wire = crate::responses_wire::ResponsesWireAdapter::new(
            provider.capabilities().responses_wire_protocol(),
            provider.capabilities().turn_routing_policy(),
        );
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(ref api_key) = config.api_key {
            match config.auth_scheme {
                AuthScheme::XApiKey => {
                    let header_value = HeaderValue::from_str(api_key).map_err(|_| {
                        tracing::debug!(
                            api_key = %api_key,
                            "Invalid api_key: cannot be converted to a valid HTTP header"
                        );
                        SamplingError::Auth(
                            "Invalid api_key: cannot be converted to a valid HTTP header"
                                .to_string(),
                        )
                    })?;
                    headers.insert(HeaderName::from_static("x-api-key"), header_value);
                }
                AuthScheme::Bearer => {
                    let bearer = format!("Bearer {}", api_key);
                    let header_value = HeaderValue::from_str(&bearer).map_err(|_| {
                        tracing::debug!(
                            api_key = %api_key,
                            "Invalid api_key: cannot be converted to a valid HTTP Authorization header"
                        );
                        SamplingError::Auth(
                            "Invalid api_key: cannot be converted to a valid HTTP Authorization header"
                                .to_string(),
                        )
                    })?;
                    headers.insert(AUTHORIZATION, header_value);
                }
            }
        }

        // Apply static headers first, then environment overrides. Resolve the
        // account identity only after this finalized effective ordering so the
        // history guard and origin stamps agree with the wire request.
        apply_config_http_headers(
            &config.extra_headers,
            &config.env_http_headers,
            |var| std::env::var(var).ok(),
            &mut headers,
        )?;
        let chatgpt_account_id = sampling_identity_from_effective_headers(
            provider.protocol(),
            &config.base_url,
            &config.model,
            &headers,
        )
        .chatgpt_account_id;

        // Add x-grok-client-version header for version gating at the proxy.
        if let Some(client_version) = config.client_version.as_ref()
            && let Ok(header_value) = HeaderValue::from_str(client_version)
        {
            headers.insert(
                HeaderName::from_static("x-grok-client-version"),
                header_value,
            );
        }

        if let Some(deployment_id) = config.deployment_id.as_ref()
            && let Ok(header_value) = HeaderValue::from_str(deployment_id)
        {
            headers.insert(
                HeaderName::from_static("x-grok-deployment-id"),
                header_value,
            );
        }

        if let Some(user_id) = config.user_id.as_ref()
            && let Ok(header_value) = HeaderValue::from_str(user_id)
        {
            headers.insert(HeaderName::from_static("x-grok-user-id"), header_value);
        }

        {
            let client_id = config
                .client_identifier
                .clone()
                .unwrap_or_else(|| DEFAULT_CLIENT_IDENTIFIER.to_string());
            if let Ok(header_value) = HeaderValue::from_str(&client_id) {
                headers.insert(
                    HeaderName::from_static("x-grok-client-identifier"),
                    header_value,
                );
            }
        }

        // Always set User-Agent: per-session origin if available, else fallback.
        {
            let ua_string = match config.origin_client.as_ref() {
                Some(origin) => user_agent_string_for(origin),
                None => user_agent_string_for(&OriginClientInfo {
                    product: AGENT_PRODUCT.to_string(),
                    version: Some(agent_version()),
                }),
            };
            if let Ok(v) = HeaderValue::from_str(&ua_string) {
                headers.insert(USER_AGENT, v);
            }
        }

        let http = if config.force_http1 {
            tracing::info!("Using HTTP/1.1 for sampling client (force_http1=true)");
            crate::shared_http::client_http1().map_err(SamplingError::Http)?
        } else {
            crate::shared_http::client().map_err(SamplingError::Http)?
        };

        tracing::info!(
            target: crate::sampling_log::TARGET,
            event = "client_new",
            base_url = %config.base_url,
            model = %config.model,
            api_backend = ?config.api_backend,
            auth_scheme = ?config.auth_scheme,
            // "unset" (not "none"): `ReasoningEffort::None` is a real wire value;
            // logging the absent Option as "none" looked like we were sending it.
            reasoning_effort = config.reasoning_effort.map_or("unset", |e| e.as_str()),
            has_api_key = config.api_key.is_some(),
            has_bearer_resolver = config.bearer_resolver.is_some(),
            has_authorization_header = headers.get(AUTHORIZATION).is_some(),
            has_x_api_key_header = headers.get(HeaderName::from_static("x-api-key")).is_some(),
        );

        let defaults = ClientDefaults {
            model: config.model,
            max_completion_tokens: config.max_completion_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            api_backend: config.api_backend,
            auth_scheme: config.auth_scheme,
            stream_tool_calls: config.stream_tool_calls,
            doom_loop_recovery: config.doom_loop_recovery,
        };

        let endpoint = EndpointTemplate::new(&config.base_url, &config.query_params);

        Ok(Self {
            http,
            default_headers: headers,
            base_url: config.base_url,
            chatgpt_account_id,
            provider,
            responses_wire,
            defaults,
            attribution_callback: config.attribution_callback,
            bearer_resolver: config.bearer_resolver,
            header_injector: config.header_injector,
            endpoint,
        })
    }

    /// The configured API backend for this client.
    pub fn api_backend(&self) -> ApiBackend {
        self.defaults.api_backend.clone()
    }

    /// Return the immutable runtime identity snapshot used by this client for
    /// history guards and provider-origin metadata for `model`.
    ///
    /// The account comes from the finalized effective header map captured at
    /// client construction, including any valid environment-header override.
    pub fn sampling_identity_for_model(
        &self,
        model: &str,
    ) -> xai_grok_sampling_types::SamplingIdentity {
        xai_grok_sampling_types::SamplingIdentity::new_for_protocol(
            self.provider.protocol(),
            self.base_url.clone(),
            model,
            self.chatgpt_account_id.clone(),
        )
    }

    /// POST with default headers. When a bearer_resolver is wired it is the
    /// sole auth source: a missing live bearer strips default Authorization /
    /// x-api-key so a hard-expired seed key cannot ride on the wire.
    fn post(&self, url: impl reqwest::IntoUrl) -> reqwest::RequestBuilder {
        let mut headers = self.default_headers.clone();
        if let Some(resolver) = &self.bearer_resolver {
            headers.remove(AUTHORIZATION);
            headers.remove(HeaderName::from_static("x-api-key"));
            if let Some(fresh) = resolver.current_bearer() {
                match self.defaults.auth_scheme {
                    AuthScheme::XApiKey => {
                        if let Ok(v) = HeaderValue::from_str(&fresh) {
                            headers.insert(HeaderName::from_static("x-api-key"), v);
                        }
                    }
                    AuthScheme::Bearer => {
                        if let Ok(v) = HeaderValue::from_str(&format!("Bearer {fresh}")) {
                            headers.insert(AUTHORIZATION, v);
                        }
                    }
                }
            }
        }
        {
            let auth_prefix = headers
                .get(AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.chars().take(20).collect::<String>());
            let x_api_key_prefix = headers
                .get(HeaderName::from_static("x-api-key"))
                .and_then(|v| v.to_str().ok())
                .map(|s| s.chars().take(12).collect::<String>());
            tracing::info!(
                target: crate::sampling_log::TARGET,
                event = "client_post",
                base_url = %self.base_url,
                model = %self.defaults.model,
                api_backend = ?self.defaults.api_backend,
                auth_scheme = ?self.defaults.auth_scheme,
                has_bearer_resolver = self.bearer_resolver.is_some(),
                has_authorization_header = headers.get(AUTHORIZATION).is_some(),
                has_x_api_key_header = headers.get(HeaderName::from_static("x-api-key")).is_some(),
                auth_header_prefix = auth_prefix.as_deref().unwrap_or("none"),
                x_api_key_prefix = x_api_key_prefix.as_deref().unwrap_or("none"),
            );
        }
        if let Some(injector) = &self.header_injector {
            injector.inject(&mut headers);
        }
        self.http.post(url).headers(headers)
    }

    /// Bearer prefix for 401 attribution. When a resolver is wired it is
    /// authoritative (including `None` ⇒ nothing was sent). Without a resolver,
    /// fall back to construction-time default headers.
    fn current_sent_bearer_prefix(&self) -> Option<String> {
        if self.bearer_resolver.is_some() {
            return self
                .bearer_resolver
                .as_ref()
                .and_then(|r| r.current_bearer())
                .map(|mut s| {
                    s.truncate(crate::attribution::SENT_BEARER_PREFIX_LEN.min(s.len()));
                    s
                });
        }
        self.extract_sent_bearer()
    }

    /// Extract the bearer from `default_headers`, truncated to prefix length.
    /// Reads `x-api-key` (Anthropic Messages API) or `Authorization` (OpenAI-completions).
    fn extract_sent_bearer(&self) -> Option<String> {
        let raw = match self.defaults.auth_scheme {
            AuthScheme::XApiKey => self
                .default_headers
                .get(HeaderName::from_static("x-api-key"))
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string()),
            AuthScheme::Bearer => self
                .default_headers
                .get(AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer "))
                .map(|s| s.to_string()),
        };
        raw.map(|mut s| {
            // Truncate in-place so we never materialize a heap-resident
            // copy of the full bearer outside the local stack of this
            // function. `String::truncate` operates on byte indices and
            // panics on a non-char-boundary cut; bearer tokens are
            // ASCII (per the `Authorization` and `x-api-key` header
            // grammars) so the byte index is always safe.
            s.truncate(crate::attribution::SENT_BEARER_PREFIX_LEN.min(s.len()));
            s
        })
    }

    /// Invoke the optional 401 attribution callback for one logical
    /// 401 response. Each of the six UNAUTHORIZED arms in this file
    /// calls this helper immediately before returning
    /// `SamplingError::Auth(...)`. Emit happens at the lowest layer
    /// that saw the status, so higher layers that react to a 401 must
    /// not emit a duplicate event.
    ///
    /// The bearer passed to the callback is already truncated to
    /// [`crate::attribution::SENT_BEARER_PREFIX_LEN`] characters by
    /// [`Self::extract_sent_bearer`]; the trait contract guarantees
    /// that callers downstream of this crate never see the full
    /// bearer.
    fn record_401_attribution(&self, consumer: crate::attribution::SamplingConsumer) {
        if let Some(cb) = self.attribution_callback.as_ref() {
            let sent_prefix = self.current_sent_bearer_prefix();
            cb.record_401(consumer, sent_prefix.as_deref());
        }
    }

    pub fn auth_info(&self) -> crate::sampling_log::AuthInfo {
        let auth_prefix = self.current_sent_bearer_prefix();
        let auth_type = match (&self.defaults.auth_scheme, &auth_prefix) {
            (AuthScheme::XApiKey, Some(_)) => "x-api-key",
            (AuthScheme::Bearer, Some(_)) => "bearer",
            (_, None) => "none",
        };
        crate::sampling_log::AuthInfo {
            auth_type,
            auth_prefix,
        }
    }

    /// Check if a header name contains sensitive information that should be redacted.
    fn is_sensitive_header(name: &str) -> bool {
        let lower = name.to_lowercase();
        lower.contains("authorization")
            || lower.contains("api-key")
            || lower.contains("apikey")
            || lower.contains("token")
            || lower.contains("secret")
    }

    /// Short lossy body snippet for error logs (never user-facing).
    fn body_preview(bytes: &[u8]) -> String {
        String::from_utf8_lossy(bytes).chars().take(500).collect()
    }

    /// Log all headers from a request at debug level (redacting sensitive values).
    fn log_request_headers(request: &reqwest::Request, endpoint_name: &str) {
        for (name, value) in request.headers().iter() {
            let value_str = if Self::is_sensitive_header(name.as_str()) {
                "[REDACTED]"
            } else {
                value.to_str().unwrap_or("[non-utf8]")
            };
            tracing::debug!(
                header_name = %name,
                header_value = %value_str,
                "Request header ({})",
                endpoint_name
            );
        }
    }

    fn endpoint(&self, path: &str) -> String {
        self.endpoint.url_for_path(path)
    }

    fn apply_defaults(&self, mut request: ChatCompletionRequest) -> Result<ChatCompletionRequest> {
        if request.model.is_none() {
            request.model = Some(self.defaults.model.clone());
        }

        if request.max_tokens.is_none() {
            request.max_tokens = self.defaults.max_completion_tokens;
        }

        if request.temperature.is_none() {
            request.temperature = self.defaults.temperature;
        }

        if request.top_p.is_none() {
            request.top_p = self.defaults.top_p;
        }

        Ok(request)
    }

    async fn handle_response(&self, response: reqwest::Response) -> Result<ChatCompletionResponse> {
        let status = response.status();
        let model_metadata = extract_model_metadata(response.headers());
        let retry_after_secs = extract_retry_after(response.headers());
        let should_retry = extract_should_retry(response.headers());
        let bytes = response.bytes().await?;

        if !status.is_success() {
            if status == reqwest::StatusCode::UNAUTHORIZED {
                self.record_401_attribution(crate::attribution::SamplingConsumer::ChatCompletions);
                let server_message = user_facing_api_error_message(status, bytes.as_ref());
                return Err(SamplingError::Auth(format!(
                    "Unauthorized (401): {server_message}"
                )));
            }
            let message = user_facing_api_error_message(status, bytes.as_ref());
            return Err(SamplingError::Api {
                status,
                message,
                model_metadata,
                retry_after_secs,
                should_retry,
            });
        }

        let completion = serde_json::from_slice::<ChatCompletionResponse>(&bytes).map_err(|e| {
            let raw_body = String::from_utf8_lossy(&bytes);
            tracing::error!(
                error = %e,
                raw_body = %raw_body,
                "Failed to deserialize ChatCompletionResponse"
            );
            SamplingError::Serialization(e)
        })?;
        Ok(completion)
    }

    // =========================================================================
    // Chat Completions API
    // =========================================================================

    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let payload = self.apply_defaults(request)?;
        let x_grok_conv_id = &payload.x_grok_conv_id.clone().unwrap_or_default();
        let x_grok_req_id = &payload.x_grok_req_id.clone().unwrap_or_default();
        let model_id = payload.model.clone().unwrap_or_default();

        tracing::debug!(
            base_url = %self.base_url,
            model_id = %model_id,
            "Sending chat completion request"
        );

        let grok_headers = GrokRequestHeaders {
            conv_id: x_grok_conv_id,
            req_id: x_grok_req_id,
            model_id: &model_id,
            session_id: payload.x_grok_session_id.as_deref().unwrap_or_default(),
            turn_idx: payload.x_grok_turn_idx.as_deref(),
            agent_id: payload.x_grok_agent_id.as_deref().unwrap_or_default(),
            deployment_id: payload.x_grok_deployment_id.as_deref(),
            user_id: payload.x_grok_user_id.as_deref(),
        };
        let http_request = grok_headers
            .apply(self.post(self.endpoint("chat/completions")))
            .json(&payload);

        let response = http_request.send().await.map_err(|e| {
            // Log at debug level; errors are surfaced to the caller.
            tracing::debug!("HTTP request failed: {}", e);
            e
        })?;

        self.handle_response(response).await
    }

    /// Start a streaming chat completion request. Returns a stream of typed chunks.
    #[tracing::instrument(
        name = "http.chat_completion_stream",
        skip_all,
        fields(
            endpoint = %self.endpoint("chat/completions"),
            model_id = request.model.as_deref().unwrap_or(""),
            status_code = tracing::field::Empty,
            success = tracing::field::Empty,
            error = tracing::field::Empty,
        )
    )]
    pub async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<(
        BoxStream<'static, Result<ChatCompletionChunk>>,
        Option<ResponseModelMetadata>,
    )> {
        let payload = self.apply_defaults(request)?;
        let x_grok_conv_id = &payload.x_grok_conv_id.clone().unwrap_or_default();
        let x_grok_req_id = &payload.x_grok_req_id.clone().unwrap_or_default();
        let model_id = payload.model.clone().unwrap_or_default();

        // Wrap the request with streaming fields and serialize once.
        // Previously this path serialized twice: first to serde_json::Value
        // (to inject `stream` and `stream_options`), then to HTTP body bytes.
        let streaming_request = StreamingChatRequest {
            inner: &payload,
            stream: true,
            stream_options: StreamOptions {
                include_usage: true,
            },
        };

        let grok_headers = GrokRequestHeaders {
            conv_id: x_grok_conv_id,
            req_id: x_grok_req_id,
            model_id: &model_id,
            session_id: payload.x_grok_session_id.as_deref().unwrap_or_default(),
            turn_idx: payload.x_grok_turn_idx.as_deref(),
            agent_id: payload.x_grok_agent_id.as_deref().unwrap_or_default(),
            deployment_id: payload.x_grok_deployment_id.as_deref(),
            user_id: payload.x_grok_user_id.as_deref(),
        };
        let http_request = grok_headers
            .apply(self.post(self.endpoint("chat/completions")))
            .header(ACCEPT, HeaderValue::from_static("text/event-stream"))
            .json(&streaming_request);

        let built_request = http_request.build().map_err(|e| {
            tracing::error!("Failed to build HTTP request: {}", e);
            SamplingError::Http(e)
        })?;

        tracing::debug!(
            url = %built_request.url(),
            method = %built_request.method(),
            "Sending chat/completions request"
        );
        Self::log_request_headers(&built_request, "chat/completions");

        let response = self.http.execute(built_request).await.map_err(|e| {
            tracing::debug!("HTTP request failed: {}", e);
            record_stream_request_failure(&e);
            e
        })?;

        let status = response.status();
        let span = tracing::Span::current();
        span.record("status_code", status.as_u16() as i64);
        span.record("success", status.is_success());
        let model_metadata = extract_model_metadata(response.headers());
        let retry_after_secs = extract_retry_after(response.headers());
        let should_retry = extract_should_retry(response.headers());
        if !status.is_success() {
            if status == reqwest::StatusCode::UNAUTHORIZED {
                span.record("error", "unauthorized (401)");
                self.record_401_attribution(
                    crate::attribution::SamplingConsumer::ChatCompletionsStream,
                );
                let endpoint = self.endpoint("chat/completions");
                let body = response.bytes().await.unwrap_or_default();
                let server_message = user_facing_api_error_message(status, body.as_ref());
                return Err(SamplingError::Auth(format!(
                    "Unauthorized (401) from {endpoint}: {server_message}"
                )));
            }

            let bytes = response.bytes().await?;
            let message = user_facing_api_error_message(status, bytes.as_ref());
            span.record("error", message.as_str());
            tracing::error!(
                status = %status,
                error_message = %message,
                body_preview = %Self::body_preview(bytes.as_ref()),
                model_id = %model_id,
                "chat/completions API error"
            );
            return Err(SamplingError::Api {
                status,
                message,
                model_metadata,
                retry_after_secs,
                should_retry,
            });
        }

        // Strip UTF-8 BOM if present: eventsource-stream 0.2.3 incorrectly slices BOM at byte 1 instead of 3.
        const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];
        let mut is_first = true;
        let byte_stream = response.bytes_stream().map(move |result| {
            result.map(|bytes| {
                if is_first {
                    is_first = false;
                    if bytes.starts_with(UTF8_BOM) {
                        return bytes.slice(UTF8_BOM.len()..);
                    }
                }
                bytes
            })
        });

        // Turn raw bytes into SSE events
        let event_stream = byte_stream.eventsource();

        // Map SSE events into ChatCompletionChunk.
        // Uses `scan` so that `[DONE]` and transport errors both terminate the
        // stream (`None`). The first transport error is emitted to the consumer,
        // then subsequent polls return `None` -- preventing an infinite busy-loop
        // when the HTTP/2 connection drops and h2 keeps producing errors.
        let chunks = event_stream
            .scan(false, |had_transport_error, event_res| {
                if *had_transport_error {
                    return std::future::ready(None);
                }
                let item = match event_res {
                    Ok(event) => {
                        let data = &event.data;
                        if data == "[DONE]" {
                            return std::future::ready(None);
                        }

                        tracing::info!(
                            target: crate::sampling_log::TARGET,
                            event = "sse_chunk",
                            backend = "chat_completions",
                            data = %data,
                        );

                        if let Some(stream_error) = try_parse_stream_error(data) {
                            Some(Err(stream_error))
                        } else {
                            Some(
                                serde_json::from_str::<ChatCompletionChunk>(data).map_err(|e| {
                                    tracing::error!(
                                        error = %e,
                                        raw_data = %data,
                                        "Failed to deserialize ChatCompletionChunk from stream"
                                    );
                                    SamplingError::Serialization(e)
                                }),
                            )
                        }
                    }
                    Err(e) => {
                        *had_transport_error = true;
                        Some(Err(SamplingError::EventStreamError(e.to_string())))
                    }
                };
                std::future::ready(item)
            })
            .boxed();

        Ok((chunks, model_metadata))
    }

    // =========================================================================
    // Responses API
    // =========================================================================

    /// Apply default configuration to a Responses API request.
    fn apply_response_defaults(&self, request: &mut CreateResponseWrapper) -> Result<()> {
        // Apply model default if not specified
        if request.inner.model.is_none() {
            request.inner.model = Some(self.defaults.model.clone());
        }

        // Apply temperature default if not specified
        if request.inner.temperature.is_none() {
            request.inner.temperature = self.defaults.temperature;
        }

        // Apply top_p default if not specified
        if request.inner.top_p.is_none() {
            request.inner.top_p = self.defaults.top_p;
        }

        // Apply max_output_tokens default if not specified
        if request.inner.max_output_tokens.is_none() {
            request.inner.max_output_tokens = self.defaults.max_completion_tokens;
        }

        // Set store to false if not specified (default is true, but that breaks ZDR compliance)
        if request.inner.store.is_none() {
            request.inner.store = Some(false);
        }

        // Include encrypted reasoning content if not specified
        let includes = request.inner.include.get_or_insert_with(Vec::new);
        if !includes.contains(&rs::IncludeEnum::ReasoningEncryptedContent) {
            includes.push(rs::IncludeEnum::ReasoningEncryptedContent);
        }

        Ok(())
    }

    /// Apply backend-specific rules after generic defaults and immediately
    /// before wire serialization. Ordering is significant for Codex because
    /// generic defaults may populate fields that its API rejects.
    fn normalize_response_for_backend(&self, request: &mut CreateResponseWrapper) {
        self.responses_wire.normalize_create_response(request);
    }

    fn apply_responses_sideband(
        sideband: crate::responses_wire::ResponsesSideband,
        turn_routing_state: Option<&TurnRoutingState>,
    ) {
        if let (Some(state), Some(value)) = (turn_routing_state, sideband.turn_routing_value) {
            state.capture_first(value);
        }
    }

    /// Compact a Codex Responses conversation using the native unary endpoint.
    ///
    /// Unlike ordinary `/responses` sampling, this returns provider-authored
    /// structured replacement history and never enables streaming. The body is
    /// represented by [`CodexCompactRequest`], whose type cannot express the
    /// unsupported temperature/top-p/output-token controls.
    #[tracing::instrument(
        name = "http.codex_compact_response",
        skip_all,
        fields(
            endpoint = %self.endpoint("responses/compact"),
            model_id = request.model.as_deref().unwrap_or(""),
            status_code = tracing::field::Empty,
            success = tracing::field::Empty,
            error = tracing::field::Empty,
        )
    )]
    pub async fn conversation_compact_responses(
        &self,
        mut request: ConversationRequest,
    ) -> Result<CodexCompactResponse> {
        if self.provider.capabilities().native_compaction_kind()
            != xai_grok_sampling_types::NativeCompactionKind::Codex
        {
            return Err(SamplingError::InvalidConfiguration(
                "native Codex compaction requires a Codex Responses sampling target",
            ));
        }
        self.apply_conversation_defaults(&mut request)?;
        self.reject_incompatible_native_history(&request, "Responses")?;
        let tracking = GrokRequestHeaders {
            conv_id: request.x_grok_conv_id.as_deref().unwrap_or_default(),
            req_id: request.x_grok_req_id.as_deref().unwrap_or_default(),
            model_id: request.model.as_deref().unwrap_or_default(),
            session_id: request.x_grok_session_id.as_deref().unwrap_or_default(),
            turn_idx: request.x_grok_turn_idx.as_deref(),
            agent_id: request.x_grok_agent_id.as_deref().unwrap_or_default(),
            deployment_id: request.x_grok_deployment_id.as_deref(),
            user_id: request.x_grok_user_id.as_deref(),
        };
        let origin = self.response_metadata_origin(request.model.as_deref().unwrap_or_default());
        let payload: CodexCompactRequest =
            conversation_request_to_codex_compact_request_for_origin(&request, origin.as_ref())
                .map_err(SamplingError::serialization_message)?;
        let mut request_body = serde_json::to_value(&payload).map_err(|error| {
            tracing::error!(%error, "failed to serialize Codex compact request");
            SamplingError::Serialization(error)
        })?;
        xai_grok_sampling_types::patch_reasoning_text_types(&mut request_body);
        let turn_routing_state = request.turn_routing_state.as_ref();
        let compact_builder = tracking.apply(self.post(self.endpoint("responses/compact")));
        let compact_builder = self
            .responses_wire
            .apply_turn_routing(compact_builder, turn_routing_state);
        let built_request = compact_builder
            .json(&request_body)
            .build()
            .map_err(SamplingError::Http)?;
        Self::log_request_headers(&built_request, "responses/compact");
        let response = self.http.execute(built_request).await.map_err(|error| {
            record_stream_request_failure(&error);
            error
        })?;
        let status = response.status();
        self.responses_wire
            .capture_turn_routing(response.headers(), turn_routing_state);
        let span = tracing::Span::current();
        span.record("status_code", status.as_u16() as i64);
        span.record("success", status.is_success());
        let model_metadata = extract_model_metadata(response.headers());
        let retry_after_secs = extract_retry_after(response.headers());
        let should_retry = extract_should_retry(response.headers());
        let bytes = response.bytes().await?;
        if !status.is_success() {
            let message = user_facing_api_error_message(status, bytes.as_ref());
            span.record("error", message.as_str());
            if status == reqwest::StatusCode::UNAUTHORIZED {
                self.record_401_attribution(crate::attribution::SamplingConsumer::ResponsesCompact);
                return Err(SamplingError::Auth(format!(
                    "Unauthorized (401) from {}: {message}",
                    self.endpoint("responses/compact")
                )));
            }
            return Err(SamplingError::Api {
                status,
                message,
                model_metadata,
                retry_after_secs,
                should_retry,
            });
        }
        serde_json::from_slice::<CodexCompactResponse>(&bytes).map_err(|error| {
            tracing::error!(
                %error,
                body_preview = %Self::body_preview(bytes.as_ref()),
                "failed to deserialize Codex compact replacement history"
            );
            let unsupported = serde_json::from_slice::<serde_json::Value>(&bytes)
                .ok()
                .and_then(|value| value.get("output")?.as_array().cloned())
                .and_then(|items| {
                    items.into_iter().find_map(|item| {
                        let kind = item.get("type")?.as_str()?;
                        (!matches!(kind, "message" | "reasoning" | "compaction" | "compaction_summary"))
                            .then(|| kind.to_string())
                    })
                });
            if let Some(kind) = unsupported {
                SamplingError::serialization_message(format!(
                    "unsupported Codex compact output variant `{kind}`; replacement history was not installed"
                ))
            } else {
                SamplingError::serialization_message(format!(
                    "unsupported or malformed Codex compact output fields; replacement history was not installed: {error}"
                ))
            }
        })
    }

    /// Create a response using the Responses API (non-streaming).
    ///
    /// This uses the Responses API format which provides a simpler interface
    /// for multi-turn conversations and tool calling.
    pub async fn create_response(
        &self,
        mut request: CreateResponseWrapper,
    ) -> Result<xai_grok_sampling_types::DecodedResponse> {
        self.apply_response_defaults(&mut request)?;
        self.normalize_response_for_backend(&mut request);

        let x_grok_conv_id = request.x_grok_conv_id.as_deref().unwrap_or_default();
        let x_grok_req_id = request.x_grok_req_id.as_deref().unwrap_or_default();
        let model_id = request.inner.model.clone().unwrap_or_default();

        // The trace field is process-local: it is consumed by upstream
        // session code (which may upload a payload artifact) and is not
        // forwarded by the sampler. Drop it before we send.
        request.trace.take();

        tracing::debug!("create_response: {:?}", &request);
        tracing::debug!("endpoint: {:?}", self.endpoint("responses"));

        let grok_headers = GrokRequestHeaders {
            conv_id: x_grok_conv_id,
            req_id: x_grok_req_id,
            model_id: &model_id,
            session_id: request.x_grok_session_id.as_deref().unwrap_or_default(),
            turn_idx: request.x_grok_turn_idx.as_deref(),
            agent_id: request.x_grok_agent_id.as_deref().unwrap_or_default(),
            deployment_id: request.x_grok_deployment_id.as_deref(),
            user_id: request.x_grok_user_id.as_deref(),
        };
        let mut request_body = serde_json::to_value(&request.inner).map_err(|e| {
            tracing::error!("Failed to serialize responses request: {}", e);
            SamplingError::Serialization(e)
        })?;
        // async-openai's ReasoningTextContent struct omits the `type`
        // discriminator that the Responses API requires on input. Patch
        // it in post-serialize. This is the last surviving piece of the
        // old raw_output machinery.
        xai_grok_sampling_types::patch_reasoning_text_types(&mut request_body);
        xai_grok_sampling_types::patch_response_message_item_ids(
            &mut request_body,
            &request.response_message_item_ids,
        );
        xai_grok_sampling_types::patch_response_item_metadata_passthrough(
            &mut request_body,
            &request.response_item_metadata_passthrough,
        )
        .map_err(SamplingError::serialization_message)?;
        let turn_routing_state = request.turn_routing_state.as_ref();
        let http_request = grok_headers.apply(self.post(self.endpoint("responses")));
        let http_request = self
            .responses_wire
            .apply_turn_routing(http_request, turn_routing_state)
            .json(&request_body);

        let response = http_request.send().await.map_err(|e| {
            tracing::debug!("HTTP request failed: {}", e);
            e
        })?;

        let status = response.status();
        self.responses_wire
            .capture_turn_routing(response.headers(), turn_routing_state);
        let model_metadata = extract_model_metadata(response.headers());
        let retry_after_secs = extract_retry_after(response.headers());
        let should_retry = extract_should_retry(response.headers());
        let bytes = response.bytes().await?;

        if !status.is_success() {
            if status == reqwest::StatusCode::UNAUTHORIZED {
                self.record_401_attribution(crate::attribution::SamplingConsumer::Responses);
                let endpoint = self.endpoint("responses");
                let server_message = user_facing_api_error_message(status, bytes.as_ref());
                return Err(SamplingError::Auth(format!(
                    "Unauthorized (401) from {endpoint}: {server_message}"
                )));
            }

            let message = user_facing_api_error_message(status, bytes.as_ref());
            tracing::warn!(
                status = %status,
                error_message = %message,
                body_preview = %Self::body_preview(bytes.as_ref()),
                model_id = %model_id,
                "responses API error"
            );
            return Err(SamplingError::Api {
                status,
                message,
                model_metadata,
                retry_after_secs,
                should_retry,
            });
        }

        let mut decoded = deserialize_unary_response(
            &bytes,
            self.responses_wire.preserves_response_metadata(),
        )
        .map_err(|error| {
            let raw_body = String::from_utf8_lossy(&bytes);
            tracing::error!(%error, raw_body = %raw_body, "Failed to deserialize Responses response");
            error
        })?;
        decoded.set_metadata_origin(request.response_metadata_origin.as_ref());
        Ok(decoded)
    }

    /// Create a streaming response using the Responses API.
    ///
    /// Returns a stream of `rs::ResponseStreamEvent` which includes events like:
    /// - `response.created` - Initial response object
    /// - `response.output_text.delta` - Text content deltas
    /// - `response.function_call_arguments.delta` - Function call argument deltas
    /// - `response.completed` - Final response with all output
    ///
    /// The third tuple element is a per-request doom-loop signal collector,
    /// `Some` only when `SamplerConfig::doom_loop_recovery` is set — the same
    /// gate that adds the opt-in `x-grok-doom-loop-check` request header, so
    /// header and parse protection cannot drift apart. It is filled by the
    /// SSE decoder as the server reports triggers and is meant to be handed
    /// to `stream_responses` so the signals land on the final
    /// `ConversationResponse`.
    #[tracing::instrument(
        name = "http.create_response_stream",
        skip_all,
        fields(
            endpoint = %self.endpoint("responses"),
            model_id = request.inner.model.as_deref().unwrap_or(""),
            status_code = tracing::field::Empty,
            success = tracing::field::Empty,
            error = tracing::field::Empty,
        )
    )]
    #[allow(clippy::type_complexity)]
    pub async fn create_response_stream(
        &self,
        mut request: CreateResponseWrapper,
    ) -> Result<(
        BoxStream<'static, Result<xai_grok_sampling_types::DecodedResponseStreamEvent>>,
        Option<ResponseModelMetadata>,
        Option<crate::doom_loop::DoomLoopSignalCollector>,
    )> {
        self.apply_response_defaults(&mut request)?;
        self.normalize_response_for_backend(&mut request);

        // Enable streaming
        request.inner.stream = Some(true);

        let x_grok_conv_id = request.x_grok_conv_id.as_deref().unwrap_or_default();
        let x_grok_req_id = request.x_grok_req_id.as_deref().unwrap_or_default();
        let model_id = request.inner.model.clone().unwrap_or_default();

        // Drop process-local trace data (see note in `create_response`).
        request.trace.take();

        tracing::debug!(
            base_url = %self.base_url,
            model_id = model_id.as_str(),
            "Sending responses API stream request"
        );

        let grok_headers = GrokRequestHeaders {
            conv_id: x_grok_conv_id,
            req_id: x_grok_req_id,
            model_id: &model_id,
            session_id: request.x_grok_session_id.as_deref().unwrap_or_default(),
            turn_idx: request.x_grok_turn_idx.as_deref(),
            agent_id: request.x_grok_agent_id.as_deref().unwrap_or_default(),
            deployment_id: request.x_grok_deployment_id.as_deref(),
            user_id: request.x_grok_user_id.as_deref(),
        };
        let extra_tool_entries = std::mem::take(&mut request.extra_tool_entries);
        let mut request_body = serde_json::to_value(&request.inner).map_err(|e| {
            tracing::error!("Failed to serialize responses request: {}", e);
            SamplingError::Serialization(e)
        })?;
        // Inject xAI-specific fields not in async-openai's CreateResponse type.
        if self.defaults.stream_tool_calls {
            request_body["stream_tool_calls"] = serde_json::json!(true);
        }
        // Inject xAI-specific tools (e.g., x_search) that can't be expressed
        // via async_openai's rs::Tool enum.
        if !extra_tool_entries.is_empty() {
            if let Some(tools) = request_body.get_mut("tools").and_then(|v| v.as_array_mut()) {
                tools.extend(extra_tool_entries);
            } else {
                request_body["tools"] = serde_json::Value::Array(extra_tool_entries);
            }
        }
        xai_grok_sampling_types::patch_reasoning_text_types(&mut request_body);
        xai_grok_sampling_types::patch_response_message_item_ids(
            &mut request_body,
            &request.response_message_item_ids,
        );
        xai_grok_sampling_types::patch_response_item_metadata_passthrough(
            &mut request_body,
            &request.response_item_metadata_passthrough,
        )
        .map_err(SamplingError::serialization_message)?;
        // Fresh per attempt so signals never leak across retries; `None`
        // (check disabled) sends no header and does no peek work per event.
        let doom_loop = self
            .defaults
            .doom_loop_recovery
            .map(crate::doom_loop::DoomLoopSignalCollector::new);
        let turn_routing_state = request.turn_routing_state.as_ref();
        let http_request = grok_headers.apply(self.post(self.endpoint("responses")));
        let mut http_request = self
            .responses_wire
            .apply_turn_routing(http_request, turn_routing_state)
            .header(ACCEPT, HeaderValue::from_static("text/event-stream"));
        if doom_loop.is_some() {
            // Presence opts in; the server ignores the value.
            http_request = http_request.header(DOOM_LOOP_CHECK_HEADER, "true");
        }
        let http_request = http_request.json(&request_body);

        let built_request = http_request.build().map_err(|e| {
            tracing::error!("Failed to build HTTP request: {}", e);
            SamplingError::Http(e)
        })?;

        tracing::debug!(
            url = %built_request.url(),
            method = %built_request.method(),
            "Sending responses API stream request"
        );
        Self::log_request_headers(&built_request, "responses");

        let response = self.http.execute(built_request).await.map_err(|e| {
            tracing::debug!("HTTP request failed: {}", e);
            record_stream_request_failure(&e);
            e
        })?;

        let status = response.status();
        self.responses_wire
            .capture_turn_routing(response.headers(), turn_routing_state);
        let span = tracing::Span::current();
        span.record("status_code", status.as_u16() as i64);
        span.record("success", status.is_success());
        if !status.is_success() {
            if status == reqwest::StatusCode::UNAUTHORIZED {
                span.record("error", "unauthorized (401)");
                self.record_401_attribution(crate::attribution::SamplingConsumer::ResponsesStream);
                let endpoint = self.endpoint("responses");
                let body = response.bytes().await.unwrap_or_default();
                let server_message = user_facing_api_error_message(status, body.as_ref());
                return Err(SamplingError::Auth(format!(
                    "Unauthorized (401) from {endpoint}: {server_message}"
                )));
            }
            let model_metadata = extract_model_metadata(response.headers());
            let retry_after_secs = extract_retry_after(response.headers());
            let should_retry = extract_should_retry(response.headers());
            let bytes = response.bytes().await?;
            let message = user_facing_api_error_message(status, bytes.as_ref());
            span.record("error", message.as_str());
            tracing::error!(
                status = %status,
                error_message = %message,
                body_preview = %Self::body_preview(bytes.as_ref()),
                model_id = %model_id,
                "responses API error"
            );
            return Err(SamplingError::Api {
                status,
                message,
                model_metadata,
                retry_after_secs,
                should_retry,
            });
        }

        let model_metadata = extract_model_metadata(response.headers());

        // Strip UTF-8 BOM if present
        const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];
        let mut is_first = true;
        let byte_stream = response.bytes_stream().map(move |result| {
            result.map(|bytes| {
                if is_first {
                    is_first = false;
                    if bytes.starts_with(UTF8_BOM) {
                        return bytes.slice(UTF8_BOM.len()..);
                    }
                }
                bytes
            })
        });

        // Turn raw bytes into SSE events
        let event_stream = byte_stream.eventsource();

        let doom_loop_for_stream = doom_loop.clone();
        let preserve_codex_metadata = self.responses_wire.preserves_response_metadata();
        let responses_wire = self.responses_wire;
        let turn_routing_state_for_stream = turn_routing_state.cloned();
        let response_metadata_origin = request.response_metadata_origin.clone();

        // The scan item is an `Option`: `Some(None)` skips an absorbed
        // doom-loop event without terminating the stream (`filter_map`
        // below), while an outer `None` still ends it.
        let events = event_stream
            .scan(false, move |had_transport_error, event_res| {
                if *had_transport_error {
                    return std::future::ready(None);
                }
                let item =
                    match event_res {
                        Ok(event) => {
                            let data = &event.data;
                            if data == "[DONE]" {
                                return std::future::ready(None);
                            }

                            // Give the resolved provider wire adapter first refusal
                            // before async-openai's closed standard event enum.
                            let swallow_doom_loop = match &doom_loop_for_stream {
                                Some(collector) => collector.absorb(&event.event, data),
                                None => is_check_event(&event.event, data),
                            };
                            if swallow_doom_loop {
                                Some(None)
                            } else {
                                match responses_wire.decode_sideband(data) {
                                    Ok(Some(sideband)) => {
                                        Self::apply_responses_sideband(
                                            sideband,
                                            turn_routing_state_for_stream.as_ref(),
                                        );
                                        tracing::debug!(
                                            target: crate::sampling_log::TARGET,
                                            event = "sse_sideband",
                                            backend = "responses",
                                            "Consumed provider Responses sideband"
                                        );
                                        Some(None)
                                    }
                                    Err(error) => Some(Some(Err(error))),
                                    Ok(None) => {
                                        tracing::info!(
                                            target: crate::sampling_log::TARGET,
                                            event = "sse_chunk",
                                            backend = "responses",
                                            data = %data,
                                        );
                                        if let Some(stream_error) = try_parse_stream_error(data) {
                                            Some(Some(Err(stream_error)))
                                        } else {
                                            Some(Some(
                                                deserialize_response_event_with_metadata(
                                                    data,
                                                    preserve_codex_metadata,
                                                )
                                                .map(|event| {
                                                    event.with_metadata_origin(
                                                        response_metadata_origin.as_ref(),
                                                    )
                                                }),
                                            ))
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            *had_transport_error = true;
                            Some(Some(Err(SamplingError::EventStreamError(e.to_string()))))
                        }
                    };
                std::future::ready(item)
            })
            .filter_map(std::future::ready)
            .boxed();

        Ok((events, model_metadata, doom_loop))
    }

    // =========================================================================
    // Anthropic Messages API
    // =========================================================================

    /// Apply default configuration to a Messages API request.
    fn apply_message_defaults(&self, request: &mut MessagesRequestWrapper) -> Result<()> {
        // Apply model default if not specified
        if request.inner.model.is_empty() {
            request.inner.model = self.defaults.model.clone();
        }

        if request.inner.max_tokens == 0 {
            request.inner.max_tokens = self
                .defaults
                .max_completion_tokens
                .unwrap_or(ANTHROPIC_DEFAULT_MAX_TOKENS);
        }

        // Apply temperature default if not specified
        if request.inner.temperature.is_none() {
            request.inner.temperature = self.defaults.temperature;
        }

        // Apply top_p default if not specified
        if request.inner.top_p.is_none() {
            request.inner.top_p = self.defaults.top_p;
        }

        Ok(())
    }

    /// Create a message using the Anthropic Messages API (non-streaming).
    pub async fn create_message(
        &self,
        mut request: MessagesRequestWrapper,
    ) -> Result<messages::MessagesResponse> {
        self.apply_message_defaults(&mut request)?;

        let x_grok_conv_id = request.x_grok_conv_id.as_deref().unwrap_or_default();
        let x_grok_req_id = request.x_grok_req_id.as_deref().unwrap_or_default();
        let model_id = request.inner.model.clone();

        // Drop process-local trace data.
        request.trace.take();

        tracing::debug!("create_message: {:?}", &request.inner);
        tracing::debug!("endpoint: {:?}", self.endpoint("messages"));

        let grok_headers = GrokRequestHeaders {
            conv_id: x_grok_conv_id,
            req_id: x_grok_req_id,
            model_id: &model_id,
            session_id: request.x_grok_session_id.as_deref().unwrap_or_default(),
            turn_idx: request.x_grok_turn_idx.as_deref(),
            agent_id: request.x_grok_agent_id.as_deref().unwrap_or_default(),
            deployment_id: request.x_grok_deployment_id.as_deref(),
            user_id: request.x_grok_user_id.as_deref(),
        };
        let http_request = grok_headers
            .apply(self.post(self.endpoint("messages")))
            .json(&request.inner);

        let response = http_request.send().await.map_err(|e| {
            tracing::debug!("HTTP request failed: {}", e);
            e
        })?;

        let status = response.status();
        let model_metadata = extract_model_metadata(response.headers());
        let retry_after_secs = extract_retry_after(response.headers());
        let should_retry = extract_should_retry(response.headers());
        let bytes = response.bytes().await?;

        if !status.is_success() {
            if status == reqwest::StatusCode::UNAUTHORIZED {
                self.record_401_attribution(crate::attribution::SamplingConsumer::Messages);
                let endpoint = self.endpoint("messages");
                let server_message = user_facing_api_error_message(status, bytes.as_ref());
                return Err(SamplingError::Auth(format!(
                    "Unauthorized (401) from {endpoint}: {server_message}"
                )));
            }

            let message = user_facing_api_error_message(status, bytes.as_ref());
            tracing::warn!(
                status = %status,
                error_message = %message,
                body_preview = %Self::body_preview(bytes.as_ref()),
                model_id = %model_id,
                "messages API error"
            );
            return Err(SamplingError::Api {
                status,
                message,
                model_metadata,
                retry_after_secs,
                should_retry,
            });
        }

        let response_obj =
            serde_json::from_slice::<messages::MessagesResponse>(&bytes).map_err(|e| {
                let raw_body = String::from_utf8_lossy(&bytes);
                tracing::error!(
                    error = %e,
                    raw_body = %raw_body,
                    "Failed to deserialize MessagesResponse"
                );
                SamplingError::Serialization(e)
            })?;
        Ok(response_obj)
    }

    /// Create a streaming message using the Anthropic Messages API.
    ///
    /// Returns a stream of `MessageStreamEvent` which includes events like:
    /// - `message_start` - Initial message object
    /// - `content_block_start` / `content_block_delta` / `content_block_stop` - Content blocks
    /// - `message_delta` / `message_stop` - Final message with stop reason
    #[tracing::instrument(
        name = "http.create_message_stream",
        skip_all,
        fields(
            endpoint = %self.endpoint("messages"),
            model_id = request.inner.model.as_str(),
            status_code = tracing::field::Empty,
            success = tracing::field::Empty,
            error = tracing::field::Empty,
        )
    )]
    pub async fn create_message_stream(
        &self,
        mut request: MessagesRequestWrapper,
    ) -> Result<(
        BoxStream<'static, Result<messages::MessageStreamEvent>>,
        Option<ResponseModelMetadata>,
    )> {
        self.apply_message_defaults(&mut request)?;

        // Enable streaming
        request.inner.stream = Some(true);

        let x_grok_conv_id = request.x_grok_conv_id.as_deref().unwrap_or_default();
        let x_grok_req_id = request.x_grok_req_id.as_deref().unwrap_or_default();
        let model_id = request.inner.model.clone();

        // Drop process-local trace data.
        request.trace.take();

        tracing::debug!(
            base_url = %self.base_url,
            model_id = model_id.as_str(),
            "Sending Messages API stream request"
        );

        let grok_headers = GrokRequestHeaders {
            conv_id: x_grok_conv_id,
            req_id: x_grok_req_id,
            model_id: &model_id,
            session_id: request.x_grok_session_id.as_deref().unwrap_or_default(),
            turn_idx: request.x_grok_turn_idx.as_deref(),
            agent_id: request.x_grok_agent_id.as_deref().unwrap_or_default(),
            deployment_id: request.x_grok_deployment_id.as_deref(),
            user_id: request.x_grok_user_id.as_deref(),
        };
        let http_request = grok_headers
            .apply(self.post(self.endpoint("messages")))
            .header(ACCEPT, HeaderValue::from_static("text/event-stream"))
            .json(&request.inner);

        let built_request = http_request.build().map_err(|e| {
            tracing::error!("Failed to build HTTP request: {}", e);
            SamplingError::Http(e)
        })?;

        tracing::debug!(
            url = %built_request.url(),
            method = %built_request.method(),
            "Sending messages API stream request"
        );
        Self::log_request_headers(&built_request, "messages");

        let response = self.http.execute(built_request).await.map_err(|e| {
            tracing::debug!("HTTP request failed: {}", e);
            record_stream_request_failure(&e);
            e
        })?;

        let status = response.status();
        let span = tracing::Span::current();
        span.record("status_code", status.as_u16() as i64);
        span.record("success", status.is_success());
        if !status.is_success() {
            if status == reqwest::StatusCode::UNAUTHORIZED {
                span.record("error", "unauthorized (401)");
                self.record_401_attribution(crate::attribution::SamplingConsumer::MessagesStream);
                let endpoint = self.endpoint("messages");
                let body = response.bytes().await.unwrap_or_default();
                let server_message = user_facing_api_error_message(status, body.as_ref());
                return Err(SamplingError::Auth(format!(
                    "Unauthorized (401) from {endpoint}: {server_message}"
                )));
            }
            let model_metadata = extract_model_metadata(response.headers());
            let retry_after_secs = extract_retry_after(response.headers());
            let should_retry = extract_should_retry(response.headers());
            let bytes = response.bytes().await?;
            let message = user_facing_api_error_message(status, bytes.as_ref());
            span.record("error", message.as_str());
            tracing::error!(
                status = %status,
                error_message = %message,
                body_preview = %Self::body_preview(bytes.as_ref()),
                model_id = %model_id,
                "messages API error"
            );
            return Err(SamplingError::Api {
                status,
                message,
                model_metadata,
                retry_after_secs,
                should_retry,
            });
        }

        let model_metadata = extract_model_metadata(response.headers());

        // Strip UTF-8 BOM if present
        const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];
        let mut is_first = true;
        let byte_stream = response.bytes_stream().map(move |result| {
            result.map(|bytes| {
                if is_first {
                    is_first = false;
                    if bytes.starts_with(UTF8_BOM) {
                        return bytes.slice(UTF8_BOM.len()..);
                    }
                }
                bytes
            })
        });

        // Turn raw bytes into SSE events
        let event_stream = byte_stream.eventsource();

        // Map SSE events into MessageStreamEvent.
        // Uses `scan` so transport errors terminate the stream after the first
        // error (same pattern as `chat_completion_stream`).
        let events = event_stream
            .scan(false, |had_transport_error, event_res| {
                if *had_transport_error {
                    return std::future::ready(None);
                }
                let item = match event_res {
                    Ok(event) => {
                        let data = &event.data;
                        if data == "[DONE]" {
                            return std::future::ready(None);
                        }

                        tracing::info!(
                            target: crate::sampling_log::TARGET,
                            event = "sse_chunk",
                            backend = "messages",
                            data = %data,
                        );

                        if let Some(stream_error) = try_parse_stream_error(data) {
                            Some(Err(stream_error))
                        } else {
                            Some(
                                serde_json::from_str::<messages::MessageStreamEvent>(data).map_err(
                                    |e| {
                                        tracing::error!(
                                            error = %e,
                                            raw_data = %data,
                                            "Failed to deserialize MessageStreamEvent from stream"
                                        );
                                        SamplingError::Serialization(e)
                                    },
                                ),
                            )
                        }
                    }
                    Err(e) => {
                        *had_transport_error = true;
                        Some(Err(SamplingError::EventStreamError(e.to_string())))
                    }
                };
                std::future::ready(item)
            })
            .boxed();

        Ok((events, model_metadata))
    }

    // =========================================================================
    // Unified Conversation API
    // =========================================================================

    /// Apply default configuration to a ConversationRequest.
    fn apply_conversation_defaults(&self, request: &mut ConversationRequest) -> Result<()> {
        if request.model.is_none() {
            request.model = Some(self.defaults.model.clone());
        }

        if request.temperature.is_none() {
            request.temperature = self.defaults.temperature;
        }

        if request.top_p.is_none() {
            request.top_p = self.defaults.top_p;
        }

        if request.max_output_tokens.is_none() {
            request.max_output_tokens = self.defaults.max_completion_tokens;
        }

        Ok(())
    }

    fn response_metadata_origin(
        &self,
        model: &str,
    ) -> Option<xai_grok_sampling_types::ResponseMetadataOrigin> {
        let identity = self.sampling_identity_for_model(model);
        xai_grok_sampling_types::ResponseMetadataOrigin::codex(
            &identity.base_url,
            &identity.model,
            identity.chatgpt_account_id,
        )
    }

    fn reject_incompatible_native_history(
        &self,
        request: &ConversationRequest,
        api: &'static str,
    ) -> Result<()> {
        let api_backend = match api {
            "Responses" => xai_grok_sampling_types::ApiBackend::Responses,
            "Messages" => xai_grok_sampling_types::ApiBackend::Messages,
            _ => xai_grok_sampling_types::ApiBackend::ChatCompletions,
        };
        let model = request.model.as_deref().unwrap_or_default();
        let identity = if api_backend == self.defaults.api_backend {
            self.sampling_identity_for_model(model)
        } else {
            xai_grok_sampling_types::SamplingIdentity::new(
                api_backend,
                self.base_url.clone(),
                model,
                self.chatgpt_account_id.clone(),
            )
        };
        match xai_grok_sampling_types::validate_history_for_sampling_identity(
            &request.items,
            &identity,
        ) {
            Ok(()) => Ok(()),
            Err(xai_grok_sampling_types::SamplingIdentityHistoryError::MalformedNativeHistory(
                _,
            )) => Err(SamplingError::InvalidConfiguration(
                "native Codex compaction history has missing or malformed durable identity metadata; history was not modified",
            )),
            Err(
                xai_grok_sampling_types::SamplingIdentityHistoryError::IncompatibleNativeHistory,
            ) => Err(SamplingError::InvalidConfiguration(
                "this session contains identity-bound native Codex compaction history; backend, API, model, and ChatGPT account must exactly match the compaction origin (history was not modified)",
            )),
        }
    }

    /// Send a conversation request using the Chat Completions API (streaming).
    ///
    /// Converts the `ConversationRequest` to `ChatCompletionRequest` internally.
    /// Returns the stream and any model metadata extracted from response headers.
    pub async fn conversation_stream(
        &self,
        mut request: ConversationRequest,
    ) -> Result<(
        BoxStream<'static, Result<ChatCompletionChunk>>,
        Option<ResponseModelMetadata>,
    )> {
        self.apply_conversation_defaults(&mut request)?;
        self.reject_incompatible_native_history(&request, "Chat Completions")?;

        let trace = request.trace.take();
        let mut chat_request: ChatCompletionRequest = request.into();
        if let Some(trace) = trace {
            chat_request.trace = Some(trace);
        }

        self.chat_completion_stream(chat_request).await
    }

    /// Send a conversation request using the Chat Completions API (non-streaming).
    ///
    /// Converts the `ConversationRequest` to `ChatCompletionRequest` internally.
    pub async fn conversation(
        &self,
        mut request: ConversationRequest,
    ) -> Result<ChatCompletionResponse> {
        self.apply_conversation_defaults(&mut request)?;
        self.reject_incompatible_native_history(&request, "Chat Completions")?;

        let trace = request.trace.take();
        let mut chat_request: ChatCompletionRequest = request.into();
        if let Some(trace) = trace {
            chat_request.trace = Some(trace);
        }

        self.chat_completion(chat_request).await
    }

    /// Send a conversation request using the Responses API (streaming).
    ///
    /// Converts the `ConversationRequest` to Responses API format internally.
    /// The third tuple element is the per-request doom-loop signal collector
    /// (see [`Self::create_response_stream`]); callers that don't consume the
    /// signals can ignore it.
    #[allow(clippy::type_complexity)]
    pub async fn conversation_stream_responses(
        &self,
        mut request: ConversationRequest,
    ) -> Result<(
        BoxStream<'static, Result<xai_grok_sampling_types::DecodedResponseStreamEvent>>,
        Option<ResponseModelMetadata>,
        Option<crate::doom_loop::DoomLoopSignalCollector>,
    )> {
        self.apply_conversation_defaults(&mut request)?;
        self.reject_incompatible_native_history(&request, "Responses")?;

        let trace = request.trace.take();
        let x_grok_conv_id = request.x_grok_conv_id.clone();
        let x_grok_req_id = request.x_grok_req_id.clone();
        let x_grok_session_id = request.x_grok_session_id.clone();
        let x_grok_turn_idx = request.x_grok_turn_idx.clone();
        let x_grok_agent_id = request.x_grok_agent_id.clone();

        let origin = self.response_metadata_origin(request.model.as_deref().unwrap_or_default());
        let mut wrapper = self
            .responses_wire
            .prepare_create_response(&request, origin)?;
        wrapper.x_grok_conv_id = x_grok_conv_id;
        wrapper.x_grok_req_id = x_grok_req_id;
        wrapper.x_grok_session_id = x_grok_session_id;
        wrapper.x_grok_turn_idx = x_grok_turn_idx;
        wrapper.x_grok_agent_id = x_grok_agent_id;
        wrapper.turn_routing_state = request.turn_routing_state.clone();

        if let Some(trace) = trace {
            wrapper.trace = Some(trace);
        }

        self.create_response_stream(wrapper).await
    }

    /// Send a conversation request using the Responses API (non-streaming).
    ///
    /// Converts the `ConversationRequest` to Responses API format internally.
    pub async fn conversation_responses(
        &self,
        mut request: ConversationRequest,
    ) -> Result<xai_grok_sampling_types::DecodedResponse> {
        self.apply_conversation_defaults(&mut request)?;
        self.reject_incompatible_native_history(&request, "Responses")?;

        let trace = request.trace.take();
        let x_grok_conv_id = request.x_grok_conv_id.clone();
        let x_grok_req_id = request.x_grok_req_id.clone();
        let x_grok_session_id = request.x_grok_session_id.clone();
        let x_grok_turn_idx = request.x_grok_turn_idx.clone();
        let x_grok_agent_id = request.x_grok_agent_id.clone();

        let origin = self.response_metadata_origin(request.model.as_deref().unwrap_or_default());
        let mut wrapper = self
            .responses_wire
            .prepare_create_response(&request, origin)?;
        wrapper.x_grok_conv_id = x_grok_conv_id;
        wrapper.x_grok_req_id = x_grok_req_id;
        wrapper.x_grok_session_id = x_grok_session_id;
        wrapper.x_grok_turn_idx = x_grok_turn_idx;
        wrapper.x_grok_agent_id = x_grok_agent_id;
        wrapper.turn_routing_state = request.turn_routing_state.clone();

        if let Some(trace) = trace {
            wrapper.trace = Some(trace);
        }

        self.create_response(wrapper).await
    }

    /// Send a conversation request using the Anthropic Messages API (streaming).
    ///
    /// Converts the `ConversationRequest` to Messages API format internally.
    pub async fn conversation_stream_messages(
        &self,
        mut request: ConversationRequest,
    ) -> Result<(
        BoxStream<'static, Result<messages::MessageStreamEvent>>,
        Option<ResponseModelMetadata>,
    )> {
        self.apply_conversation_defaults(&mut request)?;
        self.reject_incompatible_native_history(&request, "Messages")?;

        let trace = request.trace.take();
        let x_grok_conv_id = request.x_grok_conv_id.clone();
        let x_grok_req_id = request.x_grok_req_id.clone();
        let x_grok_session_id = request.x_grok_session_id.clone();
        let x_grok_turn_idx = request.x_grok_turn_idx.clone();
        let x_grok_agent_id = request.x_grok_agent_id.clone();

        let messages_request = build_messages_request(&request);

        let mut wrapper = MessagesRequestWrapper::new(messages_request);
        wrapper.x_grok_conv_id = x_grok_conv_id;
        wrapper.x_grok_req_id = x_grok_req_id;
        wrapper.x_grok_session_id = x_grok_session_id;
        wrapper.x_grok_turn_idx = x_grok_turn_idx;
        wrapper.x_grok_agent_id = x_grok_agent_id;

        if let Some(trace) = trace {
            wrapper.trace = Some(trace);
        }

        self.create_message_stream(wrapper).await
    }

    /// Send a conversation request using the Anthropic Messages API (non-streaming).
    ///
    /// Converts the `ConversationRequest` to Messages API format internally.
    pub async fn conversation_messages(
        &self,
        mut request: ConversationRequest,
    ) -> Result<messages::MessagesResponse> {
        self.apply_conversation_defaults(&mut request)?;
        self.reject_incompatible_native_history(&request, "Messages")?;

        let trace = request.trace.take();
        let x_grok_conv_id = request.x_grok_conv_id.clone();
        let x_grok_req_id = request.x_grok_req_id.clone();
        let x_grok_session_id = request.x_grok_session_id.clone();
        let x_grok_turn_idx = request.x_grok_turn_idx.clone();
        let x_grok_agent_id = request.x_grok_agent_id.clone();

        let messages_request = build_messages_request(&request);

        let mut wrapper = MessagesRequestWrapper::new(messages_request);
        wrapper.x_grok_conv_id = x_grok_conv_id;
        wrapper.x_grok_req_id = x_grok_req_id;
        wrapper.x_grok_session_id = x_grok_session_id;
        wrapper.x_grok_turn_idx = x_grok_turn_idx;
        wrapper.x_grok_agent_id = x_grok_agent_id;

        if let Some(trace) = trace {
            wrapper.trace = Some(trace);
        }

        self.create_message(wrapper).await
    }

    /// Backend-aware streaming call that collects the full response.
    pub async fn conversation_collect(
        &self,
        request: ConversationRequest,
    ) -> Result<ConversationResponse> {
        let request_id = crate::types::RequestId::random();
        let idle_timeout = std::time::Duration::from_secs(300);
        let result = match self.api_backend() {
            ApiBackend::ChatCompletions => {
                let (raw, meta) = self.conversation_stream(request).await?;
                let events =
                    crate::stream::stream_chat_completions(raw, meta, request_id, idle_timeout);
                crate::stream::collect_response(events).await
            }
            ApiBackend::Responses => {
                let (raw, meta, doom_loop) = self.conversation_stream_responses(request).await?;
                let events =
                    crate::stream::stream_responses(raw, meta, request_id, idle_timeout, doom_loop);
                crate::stream::collect_response(events).await
            }
            ApiBackend::Messages => {
                let (raw, meta) = self.conversation_stream_messages(request).await?;
                let events = crate::stream::stream_messages(raw, meta, request_id, idle_timeout);
                crate::stream::collect_response(events).await
            }
        };
        result
            .map(|(response, _metrics)| response)
            .map_err(|info| SamplingError::Api {
                status: info
                    .status_code
                    .and_then(|c| reqwest::StatusCode::from_u16(c).ok())
                    .unwrap_or(reqwest::StatusCode::INTERNAL_SERVER_ERROR),
                message: info.message,
                model_metadata: info.model_metadata,
                retry_after_secs: info.retry_after_secs,
                should_retry: None,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use xai_grok_sampling_types::ConversationItem;
    use xai_grok_sampling_types::types::ChatRequestMessage;

    fn minimal_config() -> SamplerConfig {
        SamplerConfig {
            api_key: Some("test-key".to_string()),
            base_url: "https://example.test".to_string(),
            model: "test-model".to_string(),
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            api_backend: ApiBackend::ChatCompletions,
            provider_id: None,
            auth_scheme: AuthScheme::Bearer,
            extra_headers: IndexMap::new(),
            query_params: IndexMap::new(),
            env_http_headers: IndexMap::new(),
            context_window: 8192,
            force_http1: false,
            max_retries: None,
            stream_tool_calls: false,
            idle_timeout_secs: None,
            reasoning_effort: None,
            origin_client: None,
            client_identifier: None,
            deployment_id: None,
            user_id: None,
            client_version: None,
            attribution_callback: None,
            bearer_resolver: None,
            supports_backend_search: false,
            compactions_remaining: None,
            compaction_at_tokens: None,
            doom_loop_recovery: None,
            header_injector: None,
        }
    }

    #[test]
    fn native_history_is_only_compatible_with_codex_responses() {
        let mut compatibility = xai_grok_sampling_types::NativeCompactionCompatibility::codex(
            "gpt-test",
            Some("acct-test".into()),
        );
        compatibility.replacement_segment_items = 1;
        compatibility.item_metadata = vec![xai_grok_sampling_types::NativeCompactionItemMetadata {
            input_index: 0,
            kind: xai_grok_sampling_types::NativeCompactionItemKind::Compaction,
            item_id: Some("cmp_test".into()),
            internal_chat_message_metadata_passthrough: None,
        }];
        let request = ConversationRequest {
            items: vec![
                xai_grok_sampling_types::ConversationItem::native_compaction_metadata(
                    compatibility,
                ),
                xai_grok_sampling_types::ConversationItem::encrypted_compaction(
                    rs::CompactionSummaryItemParam {
                        id: Some("cmp_test".into()),
                        encrypted_content: "opaque".into(),
                    },
                ),
            ],
            model: Some("gpt-test".into()),
            ..Default::default()
        };
        let ordinary = SamplingClient::new(minimal_config()).expect("client");
        for api in ["Chat Completions", "Messages", "Responses"] {
            let error = ordinary
                .reject_incompatible_native_history(&request, api)
                .expect_err("ordinary backends must reject opaque history");
            assert!(error.to_string().contains("identity-bound native Codex"));
        }

        let mut config = minimal_config();
        config.base_url = xai_grok_sampling_types::CODEX_BACKEND_BASE_URL.to_string();
        config.model = "gpt-test".into();
        config.api_backend = ApiBackend::Responses;
        config.extra_headers.insert(
            xai_grok_sampling_types::CHATGPT_ACCOUNT_ID_HEADER.into(),
            "acct-test".into(),
        );
        let wrong_account = {
            let mut wrong = config.clone();
            wrong.extra_headers.insert(
                xai_grok_sampling_types::CHATGPT_ACCOUNT_ID_HEADER.into(),
                "acct-other".into(),
            );
            SamplingClient::new(wrong).expect("wrong-account Codex client")
        };
        let codex = SamplingClient::new(config).expect("Codex client");
        codex
            .reject_incompatible_native_history(&request, "Responses")
            .expect("Codex Responses replay remains lossless");
        let mut wrong_model_request = request.clone();
        wrong_model_request.model = Some("gpt-other".into());
        assert!(
            codex
                .reject_incompatible_native_history(&wrong_model_request, "Responses")
                .is_err(),
            "opaque history must remain pinned to its exact model"
        );
        assert!(
            wrong_account
                .reject_incompatible_native_history(&request, "Responses")
                .is_err(),
            "opaque history must remain pinned to its exact account"
        );
        assert!(
            codex
                .reject_incompatible_native_history(&request, "Messages")
                .is_err()
        );
        assert!(
            matches!(
                request.items[1],
                xai_grok_sampling_types::ConversationItem::Provider(ref provider)
                    if provider.is_encrypted_compaction()
            ),
            "guard must not mutate history"
        );
    }

    #[test]
    fn turn_routing_state_is_first_value_wins_and_replayed() {
        const TURN_HEADER: &str = "x-codex-turn-state";
        let adapter = crate::responses_wire::ResponsesWireAdapter::new(
            xai_grok_sampling_types::ResponsesWireProtocol::Codex,
            xai_grok_sampling_types::TurnRoutingPolicy::FirstValueWinsHeader(TURN_HEADER),
        );
        let state = TurnRoutingState::fresh();
        let mut first = HeaderMap::new();
        first.insert(TURN_HEADER, HeaderValue::from_static("turn-one"));
        adapter.capture_turn_routing(&first, Some(&state));
        let mut later = HeaderMap::new();
        later.insert(TURN_HEADER, HeaderValue::from_static("turn-two"));
        adapter.capture_turn_routing(&later, Some(&state));
        SamplingClient::apply_responses_sideband(
            crate::responses_wire::ResponsesSideband {
                turn_routing_value: Some("turn-three".into()),
            },
            Some(&state),
        );
        assert_eq!(state.value(), Some("turn-one"));

        let stream_only = TurnRoutingState::fresh();
        SamplingClient::apply_responses_sideband(
            crate::responses_wire::ResponsesSideband {
                turn_routing_value: Some("stream-state".into()),
            },
            Some(&stream_only),
        );
        assert_eq!(stream_only.value(), Some("stream-state"));

        let request = adapter
            .apply_turn_routing(
                reqwest::Client::new().post("https://example.test/responses"),
                Some(&state),
            )
            .build()
            .unwrap();
        assert_eq!(
            request
                .headers()
                .get(TURN_HEADER)
                .and_then(|value| value.to_str().ok()),
            Some("turn-one")
        );
        assert!(TurnRoutingState::default().value().is_none());
    }

    #[test]
    fn backend_capability_gates_turn_routing_send_and_capture() {
        const TURN_HEADER: &str = "x-codex-turn-state";
        let mut supported_config = minimal_config();
        supported_config.api_backend = ApiBackend::Responses;
        supported_config.provider_id = Some(xai_grok_sampling_types::ProviderId::Codex);
        supported_config.base_url = "http://127.0.0.1:3210/v1".into();
        let supported = SamplingClient::new(supported_config).unwrap();
        let mut unsupported_config = minimal_config();
        unsupported_config.api_backend = ApiBackend::Responses;
        unsupported_config.provider_id =
            Some(xai_grok_sampling_types::ProviderId::OpenAiCompatible);
        unsupported_config.base_url = xai_grok_sampling_types::CODEX_BACKEND_BASE_URL.into();
        let unsupported = SamplingClient::new(unsupported_config).unwrap();
        assert_eq!(
            supported.responses_wire,
            crate::responses_wire::ResponsesWireAdapter::new(
                xai_grok_sampling_types::ResponsesWireProtocol::Codex,
                xai_grok_sampling_types::TurnRoutingPolicy::FirstValueWinsHeader(TURN_HEADER),
            )
        );
        assert_eq!(
            unsupported.responses_wire,
            crate::responses_wire::ResponsesWireAdapter::new(
                xai_grok_sampling_types::ResponsesWireProtocol::Standard,
                xai_grok_sampling_types::TurnRoutingPolicy::None,
            )
        );

        let populated = TurnRoutingState::fresh();
        assert!(populated.capture_first("client-state".to_string()));
        let supported_request = supported
            .responses_wire
            .apply_turn_routing(
                reqwest::Client::new().post("https://example.test/responses"),
                Some(&populated),
            )
            .build()
            .unwrap();
        let unsupported_request = unsupported
            .responses_wire
            .apply_turn_routing(
                reqwest::Client::new().post("https://example.test/responses"),
                Some(&populated),
            )
            .build()
            .unwrap();
        assert!(supported_request.headers().contains_key(TURN_HEADER));
        assert!(!unsupported_request.headers().contains_key(TURN_HEADER));

        let fresh = TurnRoutingState::fresh();
        let mut response_headers = HeaderMap::new();
        response_headers.insert(TURN_HEADER, HeaderValue::from_static("server-state"));
        unsupported
            .responses_wire
            .capture_turn_routing(&response_headers, Some(&fresh));
        assert!(fresh.value().is_none());
    }

    #[tokio::test]
    async fn explicit_codex_proxy_consumes_metadata_sideband_and_captures_routing() {
        let app = axum::Router::new().route(
            "/v1/responses",
            axum::routing::post(|| async {
                (
                    [(reqwest::header::CONTENT_TYPE.as_str(), "text/event-stream")],
                    concat!(
                        "data: {\"type\":\"response.metadata\",\"headers\":{\"x-codex-turn-state\":\"stream-state\"}}\n\n",
                        "data: [DONE]\n\n"
                    ),
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let mut config = minimal_config();
        config.api_backend = ApiBackend::Responses;
        config.provider_id = Some(xai_grok_sampling_types::ProviderId::Codex);
        config.base_url = format!("http://{address}/v1");
        let client = SamplingClient::new(config).unwrap();
        let routing = TurnRoutingState::fresh();
        let (stream, _, _) = client
            .conversation_stream_responses(ConversationRequest {
                items: vec![ConversationItem::user("hello")],
                model: Some("test-model".into()),
                turn_routing_state: Some(routing.clone()),
                ..Default::default()
            })
            .await
            .unwrap();
        let events = stream.collect::<Vec<_>>().await;

        assert!(events.is_empty(), "sideband must not become model output");
        assert_eq!(routing.value(), Some("stream-state"));
    }

    #[tokio::test]
    async fn native_compact_rejects_non_codex_resolved_provider_before_http() {
        let mut config = minimal_config();
        config.api_backend = ApiBackend::Responses;
        config.provider_id = Some(xai_grok_sampling_types::ProviderId::Custom);
        config.base_url = "http://127.0.0.1:9/v1".into();
        let client = SamplingClient::new(config).unwrap();
        let error = client
            .conversation_compact_responses(ConversationRequest::default())
            .await
            .unwrap_err();
        assert!(
            matches!(error, SamplingError::InvalidConfiguration(_)),
            "{error}"
        );
    }

    #[test]
    fn explicit_custom_provider_overrides_codex_url_and_preserves_standard_controls() {
        let mut config = minimal_config();
        config.api_backend = ApiBackend::Responses;
        config.provider_id = Some(xai_grok_sampling_types::ProviderId::Custom);
        config.base_url = xai_grok_sampling_types::CODEX_BACKEND_BASE_URL.into();
        config.temperature = Some(0.6);
        config.top_p = Some(0.9);
        config.max_completion_tokens = Some(2048);
        let client = SamplingClient::new(config).expect("Responses client");
        assert_eq!(
            client.responses_wire,
            crate::responses_wire::ResponsesWireAdapter::new(
                xai_grok_sampling_types::ResponsesWireProtocol::Standard,
                xai_grok_sampling_types::TurnRoutingPolicy::None,
            )
        );
        let created: rs::CreateResponse = (&ConversationRequest {
            items: vec![xai_grok_sampling_types::ConversationItem::user("hello")],
            ..Default::default()
        })
            .into();
        let mut wrapper = CreateResponseWrapper::new(created);

        client.apply_response_defaults(&mut wrapper).unwrap();
        client.normalize_response_for_backend(&mut wrapper);

        assert_eq!(wrapper.inner.temperature, Some(0.6));
        assert_eq!(wrapper.inner.top_p, Some(0.9));
        assert_eq!(wrapper.inner.max_output_tokens, Some(2048));
    }

    #[test]
    fn codex_wire_normalization_runs_after_response_defaults() {
        let mut config = minimal_config();
        config.base_url = xai_grok_sampling_types::CODEX_BACKEND_BASE_URL.to_string();
        config.api_backend = ApiBackend::Responses;
        config.temperature = Some(0.8);
        config.top_p = Some(0.9);
        config.max_completion_tokens = Some(4096);
        let client = SamplingClient::new(config).expect("client");
        let created: rs::CreateResponse = (&ConversationRequest {
            items: vec![xai_grok_sampling_types::ConversationItem::user("summarize")],
            model: Some("gpt-5.6-sol".into()),
            ..Default::default()
        })
            .into();
        let mut wrapper = CreateResponseWrapper::new(created);

        client.apply_response_defaults(&mut wrapper).unwrap();
        assert!(
            wrapper.inner.temperature.is_some(),
            "generic defaults must populate first"
        );
        assert!(wrapper.inner.top_p.is_some());
        assert!(wrapper.inner.max_output_tokens.is_some());
        client.normalize_response_for_backend(&mut wrapper);

        let wire = serde_json::to_value(&wrapper.inner).expect("serialize exact wire body");
        for field in ["temperature", "top_p", "max_output_tokens"] {
            assert!(
                wire.get(field).is_none() || wire[field].is_null(),
                "last-mile Codex body must omit {field}: {wire:#}"
            );
        }
    }

    /// Verify the serialized shape of StreamingChatRequest matches the
    /// expected wire format: all ChatCompletionRequest fields flattened at
    /// top level, plus `stream: true` and `stream_options.include_usage: true`.
    #[test]
    fn streaming_chat_request_serializes_correctly() {
        let request = ChatCompletionRequest {
            model: Some("test-model".into()),
            messages: vec![ChatRequestMessage::user("hello")],
            temperature: Some(0.7),
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            user: None,
            tools: None,
            tool_choice: None,
            search_parameters: None,
            response_format: None,
            reasoning_effort: None,
            x_grok_conv_id: None,
            x_grok_req_id: None,
            x_grok_session_id: None,
            x_grok_turn_idx: None,
            x_grok_agent_id: None,
            x_grok_deployment_id: None,
            x_grok_user_id: None,
            trace: None,
        };

        let wrapper = StreamingChatRequest {
            inner: &request,
            stream: true,
            stream_options: StreamOptions {
                include_usage: true,
            },
        };

        let json: serde_json::Value = serde_json::to_value(&wrapper).unwrap();
        let obj = json.as_object().unwrap();

        assert_eq!(obj.get("stream").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(
            obj.get("stream_options")
                .and_then(|v| v.get("include_usage"))
                .and_then(|v| v.as_bool()),
            Some(true)
        );

        assert!(
            obj.get("inner").is_none(),
            "inner field should be flattened"
        );
        assert_eq!(
            obj.get("model").and_then(|v| v.as_str()),
            Some("test-model")
        );
        assert!(obj.get("messages").is_some());
        let temp = obj.get("temperature").and_then(|v| v.as_f64()).unwrap();
        assert!((temp - 0.7).abs() < 0.001, "temperature should be ~0.7");

        assert!(obj.get("max_tokens").is_none());
        assert!(obj.get("tools").is_none());
    }

    #[test]
    fn extract_retry_after_parses_seconds() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "30".parse().unwrap());
        assert_eq!(extract_retry_after(&headers), Some(30));
    }

    #[test]
    fn extract_retry_after_caps_at_120() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "3600".parse().unwrap());
        assert_eq!(extract_retry_after(&headers), Some(120));
    }

    #[test]
    fn extract_retry_after_zero_is_valid() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "0".parse().unwrap());
        assert_eq!(extract_retry_after(&headers), Some(0));
    }

    #[test]
    fn extract_retry_after_ignores_http_date() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            "Fri, 31 Dec 2025 23:59:59 GMT".parse().unwrap(),
        );
        assert_eq!(extract_retry_after(&headers), None);
    }

    #[test]
    fn extract_retry_after_none_when_missing() {
        let headers = reqwest::header::HeaderMap::new();
        assert_eq!(extract_retry_after(&headers), None);
    }

    #[test]
    fn extract_should_retry_true() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-should-retry", "true".parse().unwrap());
        assert_eq!(extract_should_retry(&headers), Some(true));
    }

    #[test]
    fn extract_should_retry_true_case_insensitive() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-should-retry", "TRUE".parse().unwrap());
        assert_eq!(extract_should_retry(&headers), Some(true));
    }

    #[test]
    fn extract_should_retry_false() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-should-retry", "false".parse().unwrap());
        assert_eq!(extract_should_retry(&headers), Some(false));
    }

    #[test]
    fn extract_should_retry_unknown_value_is_none() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-should-retry", "banana".parse().unwrap());
        assert_eq!(extract_should_retry(&headers), None);
    }

    #[test]
    fn extract_should_retry_absent_is_none() {
        let headers = reqwest::header::HeaderMap::new();
        assert_eq!(extract_should_retry(&headers), None);
    }

    #[test]
    fn new_with_minimal_config_succeeds() {
        let client = SamplingClient::new(minimal_config()).expect("client should construct");
        assert_eq!(client.api_backend(), ApiBackend::ChatCompletions);
    }

    #[test]
    fn new_applies_extra_headers() {
        let mut cfg = minimal_config();
        cfg.extra_headers
            .insert("x-test-header".to_string(), "test-value".to_string());
        cfg.extra_headers
            .insert("x-XAI-token-auth".to_string(), "xai-grok-cli".to_string());
        let _client = SamplingClient::new(cfg).expect("client with extra headers should construct");
    }

    #[test]
    fn apply_env_http_headers_resolves_trims_skips_and_overrides() {
        let mut map = IndexMap::new();
        map.insert("x-tenant-token".to_string(), "TENANT".to_string());
        map.insert("x-blank".to_string(), "BLANK".to_string());
        map.insert("x-missing".to_string(), "MISSING".to_string());
        map.insert("x-override".to_string(), "OVERRIDE".to_string());
        map.insert("x invalid".to_string(), "INVALID".to_string());

        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-override"),
            HeaderValue::from_static("static"),
        );

        apply_env_http_headers(
            &map,
            |var| match var {
                // Leading space + trailing newline exercises trimming.
                "TENANT" => Some(" tenant-secret\n".to_string()),
                "BLANK" => Some("   ".to_string()),
                "OVERRIDE" => Some("from-env".to_string()),
                "INVALID" => Some("value".to_string()),
                _ => None,
            },
            &mut headers,
        );

        assert_eq!(headers.get("x-tenant-token").unwrap(), "tenant-secret");
        assert!(headers.get("x-blank").is_none());
        assert!(headers.get("x-missing").is_none());
        // A resolved env value overrides an existing header of the same name.
        assert_eq!(headers.get("x-override").unwrap(), "from-env");
        // An invalid header name is skipped rather than panicking.
        assert!(headers.get("x invalid").is_none());
    }

    #[test]
    fn runtime_account_identity_matches_effective_header_ordering() {
        let account_header = xai_grok_sampling_types::CHATGPT_ACCOUNT_ID_HEADER.to_string();
        let resolve = |extra_headers: &IndexMap<String, String>,
                       env_http_headers: &IndexMap<String, String>,
                       value: Option<&str>| {
            resolve_runtime_sampling_identity_with_getenv(
                None,
                ApiBackend::Responses,
                xai_grok_sampling_types::CODEX_BACKEND_BASE_URL,
                "gpt-account-test",
                extra_headers,
                env_http_headers,
                |_| value.map(str::to_owned),
            )
            .unwrap()
            .chatgpt_account_id
        };

        let mut static_headers = IndexMap::new();
        static_headers.insert(account_header.clone(), "acct-static".into());
        assert_eq!(
            resolve(&static_headers, &IndexMap::new(), None).as_deref(),
            Some("acct-static")
        );

        let mut env_headers = IndexMap::new();
        env_headers.insert("chatgpt-account-id".into(), "ACCOUNT_ENV".into());
        assert_eq!(
            resolve(&IndexMap::new(), &env_headers, Some(" acct-env ")).as_deref(),
            Some("acct-env")
        );
        assert_eq!(
            resolve(&static_headers, &env_headers, Some("acct-override")).as_deref(),
            Some("acct-override")
        );
        assert_eq!(
            resolve(&static_headers, &env_headers, Some("   ")).as_deref(),
            Some("acct-static")
        );
        assert_eq!(
            resolve(&static_headers, &env_headers, Some("bad\nvalue")).as_deref(),
            Some("acct-static")
        );
    }

    #[test]
    fn runtime_identity_helper_agrees_with_sampling_client() {
        let mut config = minimal_config();
        config.api_backend = ApiBackend::Responses;
        config.base_url = xai_grok_sampling_types::CODEX_BACKEND_BASE_URL.into();
        config.model = "gpt-account-test".into();
        config.extra_headers.insert(
            xai_grok_sampling_types::CHATGPT_ACCOUNT_ID_HEADER.into(),
            "acct-static".into(),
        );
        let identity = resolve_runtime_sampling_identity(
            config.api_backend.clone(),
            &config.base_url,
            &config.model,
            &config.extra_headers,
            &config.env_http_headers,
        )
        .unwrap();
        let client = SamplingClient::new(config).unwrap();
        assert_eq!(identity.chatgpt_account_id, client.chatgpt_account_id);
    }

    #[test]
    fn endpoint_appends_path_before_a_base_url_query_without_configured_params() {
        let template =
            EndpointTemplate::new("https://gateway.example/v1?api-version=x", &IndexMap::new());
        let url = template.url_for_path("responses");
        assert!(
            url.starts_with("https://gateway.example/v1/responses?"),
            "url: {url}"
        );
        assert!(url.contains("api-version=x"), "url: {url}");
        assert!(!url.contains("x/responses"), "url: {url}");
    }

    #[test]
    fn messages_plus_anthropic_api_key_uses_x_api_key_and_not_authorization() {
        let cfg = SamplerConfig {
            api_key: Some("anthropic-key-abc123".to_string()),
            api_backend: ApiBackend::Messages,
            auth_scheme: AuthScheme::XApiKey,
            ..minimal_config()
        };
        let client = SamplingClient::new(cfg).expect("client should build");
        assert!(
            client
                .default_headers
                .get(HeaderName::from_static("x-api-key"))
                .is_some()
        );
        assert!(client.default_headers.get(AUTHORIZATION).is_none());
    }

    #[test]
    fn messages_plus_bearer_uses_authorization_and_not_x_api_key() {
        let cfg = SamplerConfig {
            api_key: Some("bearer-key-abc123".to_string()),
            api_backend: ApiBackend::Messages,
            auth_scheme: AuthScheme::Bearer,
            ..minimal_config()
        };
        let client = SamplingClient::new(cfg).expect("client should build");
        assert!(client.default_headers.get(AUTHORIZATION).is_some());
        assert!(
            client
                .default_headers
                .get(HeaderName::from_static("x-api-key"))
                .is_none()
        );
    }

    // Regression: a past change dropped User-Agent from sampling requests.
    #[test]
    fn sampling_client_always_has_user_agent() {
        let client = SamplingClient::new(minimal_config()).expect("build");
        assert!(client.default_headers.contains_key(USER_AGENT));
    }

    // Regression: a past change dropped HeaderInjector (traceparent) from sampling requests.
    #[test]
    fn header_injector_is_called_in_post() {
        #[derive(Debug)]
        struct TestInjector;
        impl crate::config::HeaderInjector for TestInjector {
            fn inject(&self, headers: &mut HeaderMap) {
                headers.insert(
                    HeaderName::from_static("traceparent"),
                    HeaderValue::from_static("00-test-trace-id-00"),
                );
            }
        }

        let mut config = minimal_config();
        config.header_injector = Some(std::sync::Arc::new(TestInjector));
        let client = SamplingClient::new(config).expect("build");
        let req = client
            .post("http://localhost/test")
            .build()
            .expect("build request");
        assert!(
            req.headers().contains_key("traceparent"),
            "HeaderInjector should inject traceparent into post() requests"
        );
    }

    #[test]
    fn user_agent_includes_origin_and_agent_product() {
        let origin = OriginClientInfo {
            product: "my-client".to_string(),
            version: Some("1.2.3".to_string()),
        };
        let ua = user_agent_string_for(&origin);
        assert!(ua.contains("my-client/1.2.3"));
        assert!(ua.contains(AGENT_PRODUCT));
    }

    #[test]
    fn user_agent_omits_origin_version_when_absent() {
        let origin = OriginClientInfo {
            product: "my-client".to_string(),
            version: None,
        };
        let ua = user_agent_string_for(&origin);
        // No slash between product and the grok-shell agent product.
        assert!(ua.starts_with("my-client grok-shell/"));
    }

    #[test]
    fn user_agent_collapses_when_origin_matches_agent() {
        let agent_version = xai_grok_version::VERSION.to_string();
        let origin = OriginClientInfo {
            product: AGENT_PRODUCT.to_string(),
            version: Some(agent_version.clone()),
        };
        let ua = user_agent_string_for(&origin);
        // Single product/version slot when the origin and agent match.
        assert!(ua.starts_with(&format!("{}/{}", AGENT_PRODUCT, agent_version)));
    }

    /// Counts callbacks for assertions in the tests below.
    #[derive(Default, Debug)]
    struct CountingCallback {
        invocations: std::sync::Mutex<Vec<(crate::attribution::SamplingConsumer, Option<String>)>>,
    }

    #[derive(Debug)]
    struct StaticBearerResolver(&'static str);

    impl crate::config::BearerResolver for StaticBearerResolver {
        fn current_bearer(&self) -> Option<String> {
            Some(self.0.to_string())
        }
    }

    impl crate::attribution::Auth401AttributionCallback for CountingCallback {
        fn record_401(
            &self,
            consumer: crate::attribution::SamplingConsumer,
            sent_bearer: Option<&str>,
        ) {
            self.invocations
                .lock()
                .unwrap()
                .push((consumer, sent_bearer.map(|s| s.to_string())));
        }
    }

    /// `extract_sent_bearer` strips the `"Bearer "` prefix off
    /// `Authorization` for OpenAI-completions backends and truncates the
    /// remaining bearer to the cross-crate prefix length.
    #[test]
    fn extract_sent_bearer_strips_bearer_prefix_for_openai_compat() {
        let cfg = SamplerConfig {
            api_key: Some("test-bearer-1234567890".to_string()),
            api_backend: ApiBackend::ChatCompletions,
            ..minimal_config()
        };
        let client = SamplingClient::new(cfg).expect("client should build");
        let bearer = client.extract_sent_bearer();
        // Bearer is truncated at the crate boundary -- callers
        // downstream of this method only ever see the prefix.
        assert_eq!(bearer.as_deref(), Some("test-bearer-"));
        assert_eq!(
            bearer.as_deref().map(str::len),
            Some(crate::attribution::SENT_BEARER_PREFIX_LEN),
        );
    }

    /// `extract_sent_bearer` reads `x-api-key` for Anthropic Messages API
    /// and truncates the value to the cross-crate prefix length.
    #[test]
    fn extract_sent_bearer_reads_x_api_key_for_messages() {
        let cfg = SamplerConfig {
            api_key: Some("anthropic-key-abc123".to_string()),
            api_backend: ApiBackend::Messages,
            auth_scheme: AuthScheme::XApiKey,
            ..minimal_config()
        };
        let client = SamplingClient::new(cfg).expect("client should build");
        let bearer = client.extract_sent_bearer();
        assert_eq!(bearer.as_deref(), Some("anthropic-ke"));
        assert_eq!(
            bearer.as_deref().map(str::len),
            Some(crate::attribution::SENT_BEARER_PREFIX_LEN),
        );
    }

    /// `extract_sent_bearer` returns `None` when no auth header is set.
    #[test]
    fn extract_sent_bearer_returns_none_when_no_header() {
        let cfg = SamplerConfig {
            api_key: None,
            api_backend: ApiBackend::ChatCompletions,
            ..minimal_config()
        };
        let client = SamplingClient::new(cfg).expect("client should build");
        assert!(client.extract_sent_bearer().is_none());
    }

    #[test]
    fn live_bearer_resolver_uses_authorization_for_messages_plus_bearer() {
        let cfg = SamplerConfig {
            api_key: Some("stale-bearer".to_string()),
            api_backend: ApiBackend::Messages,
            auth_scheme: AuthScheme::Bearer,
            bearer_resolver: Some(std::sync::Arc::new(StaticBearerResolver("fresh-bearer"))),
            ..minimal_config()
        };
        let client = SamplingClient::new(cfg).expect("client should build");
        let request = client
            .post("https://example.test/v1/messages")
            .build()
            .expect("request should build");
        let auth = request
            .headers()
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok());
        assert_eq!(auth, Some("Bearer fresh-bearer"));
        assert!(request.headers().get("x-api-key").is_none());
    }

    /// Regression: when `api_key` (which seeds `default_headers` with an
    /// `Authorization: Bearer ...`) AND a `bearer_resolver` are both set,
    /// `post()` must produce **exactly one** `Authorization` header on the
    /// wire. The pre-fix code used `RequestBuilder::header(AUTHORIZATION, ...)`
    /// which appends rather than replaces, causing two identical
    /// `Authorization` headers and a 400 from cli-chat-proxy.
    #[test]
    fn post_emits_single_authorization_with_api_key_and_bearer_resolver() {
        let cfg = SamplerConfig {
            api_key: Some("stale-bearer".to_string()),
            api_backend: ApiBackend::Responses,
            auth_scheme: AuthScheme::Bearer,
            bearer_resolver: Some(std::sync::Arc::new(StaticBearerResolver("fresh-bearer"))),
            ..minimal_config()
        };
        let client = SamplingClient::new(cfg).expect("client should build");
        let request = client
            .post("https://example.test/v1/responses")
            .build()
            .expect("request should build");
        let auth_count = request.headers().get_all(AUTHORIZATION).iter().count();
        assert_eq!(
            auth_count, 1,
            "expected exactly one Authorization header, got {auth_count}"
        );
        assert_eq!(
            request
                .headers()
                .get(AUTHORIZATION)
                .and_then(|v| v.to_str().ok()),
            Some("Bearer fresh-bearer"),
        );
    }

    #[test]
    fn live_bearer_resolver_uses_x_api_key_for_messages_plus_anthropic_api_key() {
        let cfg = SamplerConfig {
            api_key: Some("stale-anthropic".to_string()),
            api_backend: ApiBackend::Messages,
            auth_scheme: AuthScheme::XApiKey,
            bearer_resolver: Some(std::sync::Arc::new(StaticBearerResolver("fresh-anthropic"))),
            ..minimal_config()
        };
        let client = SamplingClient::new(cfg).expect("client should build");
        let request = client
            .post("https://example.test/v1/messages")
            .build()
            .expect("request should build");
        let api_key = request
            .headers()
            .get("x-api-key")
            .and_then(|v| v.to_str().ok());
        assert_eq!(api_key, Some("fresh-anthropic"));
        assert!(request.headers().get(AUTHORIZATION).is_none());
    }

    /// Bearers shorter than the prefix length pass through unchanged.
    /// Defensive against the truncation logic inadvertently widening
    /// short bearers (no panics, no zero-padding).
    #[test]
    fn extract_sent_bearer_short_bearer_passes_through_unchanged() {
        let cfg = SamplerConfig {
            api_key: Some("abc".to_string()),
            api_backend: ApiBackend::ChatCompletions,
            ..minimal_config()
        };
        let client = SamplingClient::new(cfg).expect("client should build");
        assert_eq!(client.extract_sent_bearer().as_deref(), Some("abc"));
    }

    /// `record_401_attribution` invokes the wired callback with the
    /// expected `consumer` and the truncated bearer prefix that the
    /// wire would carry. The key assertion is that the callback
    /// receives the prefix only -- the full bearer never crosses the
    /// crate boundary.
    #[test]
    fn record_401_attribution_invokes_callback_with_extracted_bearer() {
        let cb = std::sync::Arc::new(CountingCallback::default());
        let cb_dyn: crate::attribution::SharedAttributionCallback = cb.clone();
        let cfg = SamplerConfig {
            api_key: Some("the-bearer-1234567890-extra-tail".to_string()),
            api_backend: ApiBackend::ChatCompletions,
            attribution_callback: Some(cb_dyn),
            bearer_resolver: None,
            ..minimal_config()
        };
        let client = SamplingClient::new(cfg).expect("client should build");
        client.record_401_attribution(crate::attribution::SamplingConsumer::ChatCompletionsStream);
        let calls = cb.invocations.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].0,
            crate::attribution::SamplingConsumer::ChatCompletionsStream
        );
        // Prefix-only -- the `extra-tail` portion of the bearer is
        // dropped by `extract_sent_bearer` before the callback fires.
        assert_eq!(calls[0].1.as_deref(), Some("the-bearer-1"));
        assert_eq!(
            calls[0].1.as_deref().map(str::len),
            Some(crate::attribution::SENT_BEARER_PREFIX_LEN),
        );
    }

    /// When a bearer_resolver is wired but returns `None`, attribution must
    /// report no sent bearer (not the construction-time default header seed).
    #[test]
    fn bearer_resolver_none_attribution_ignores_default_headers() {
        #[derive(Debug)]
        struct EmptyResolver;
        impl crate::config::BearerResolver for EmptyResolver {
            fn current_bearer(&self) -> Option<String> {
                None
            }
        }

        let cfg = SamplerConfig {
            api_key: Some("stale-seed-token".to_string()),
            api_backend: ApiBackend::Responses,
            bearer_resolver: Some(std::sync::Arc::new(EmptyResolver)),
            ..minimal_config()
        };
        let client = SamplingClient::new(cfg).expect("client should build");
        assert_eq!(
            client.current_sent_bearer_prefix(),
            None,
            "resolver None must not attribute a stripped default seed"
        );
    }

    /// When a bearer_resolver is wired but returns `None` (hard-expired
    /// session with no live AT), default Authorization / x-api-key must be
    /// stripped so a stale seed key cannot ride the wire.
    #[test]
    fn bearer_resolver_none_strips_default_authorization() {
        #[derive(Debug)]
        struct EmptyResolver;
        impl crate::config::BearerResolver for EmptyResolver {
            fn current_bearer(&self) -> Option<String> {
                None
            }
        }

        let cfg = SamplerConfig {
            api_key: Some("stale-token".to_string()),
            api_backend: ApiBackend::Responses,
            bearer_resolver: Some(std::sync::Arc::new(EmptyResolver)),
            ..minimal_config()
        };
        let client = SamplingClient::new(cfg).expect("client should build");
        let request = client
            .post("https://example.test/v1/responses")
            .body("")
            .build()
            .expect("request should build");
        assert!(
            request.headers().get(AUTHORIZATION).is_none(),
            "stale default Authorization must not be sent when resolver is empty"
        );
    }

    /// Regression test: when a bearer_resolver is wired, `post()` must
    /// *replace* the Authorization header from `default_headers`, not
    /// append a second one. Duplicate Authorization headers cause
    /// Cloudflare to return 400 Bad Request.
    #[test]
    fn bearer_resolver_replaces_authorization_header() {
        #[derive(Debug)]
        struct StaticResolver(String);
        impl crate::config::BearerResolver for StaticResolver {
            fn current_bearer(&self) -> Option<String> {
                Some(self.0.clone())
            }
        }

        let resolver: crate::config::SharedBearerResolver =
            std::sync::Arc::new(StaticResolver("fresh-token".to_string()));
        let cfg = SamplerConfig {
            api_key: Some("stale-token".to_string()),
            api_backend: ApiBackend::Responses,
            bearer_resolver: Some(resolver),
            ..minimal_config()
        };
        let client = SamplingClient::new(cfg).expect("client should build");

        // Build a request to inspect the final headers.
        let builder = client.post("https://example.test/v1/responses");
        let request = builder.body("").build().expect("request should build");

        let auth_values: Vec<_> = request.headers().get_all(AUTHORIZATION).iter().collect();
        assert_eq!(
            auth_values.len(),
            1,
            "expected exactly one Authorization header, got {}: {:?}",
            auth_values.len(),
            auth_values
        );
        assert_eq!(
            auth_values[0].to_str().unwrap(),
            "Bearer fresh-token",
            "Authorization header should contain the resolver's fresh token"
        );
    }

    /// `record_401_attribution` is a no-op when `attribution_callback`
    /// is `None` (the BYOK / sampler-only path). The previous tests
    /// in this module construct clients without a callback and rely
    /// on this property holding.
    #[test]
    fn record_401_attribution_is_noop_without_callback() {
        let cfg = SamplerConfig {
            api_key: Some("bearer".to_string()),
            api_backend: ApiBackend::ChatCompletions,
            attribution_callback: None,
            bearer_resolver: None,
            ..minimal_config()
        };
        let client = SamplingClient::new(cfg).expect("client should build");
        // Must not panic.
        client.record_401_attribution(crate::attribution::SamplingConsumer::ChatCompletions);
    }

    /// `response.completed` carrying
    /// `usage.context_details.{input_tokens, output_tokens}` rewrites
    /// `usage.total_tokens` in place to the live context length
    /// (`ctx.input + ctx.output`). Billing fields stay on the wire's
    /// cumulative values.
    #[test]
    fn deserialize_response_event_overrides_total_tokens_from_context_details() {
        let sse = r#"{
            "type": "response.completed",
            "sequence_number": 0,
            "response": {
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": "grok-build",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 6003,
                    "input_tokens_details": { "cached_tokens": 1984 },
                    "output_tokens": 711,
                    "output_tokens_details": { "reasoning_tokens": 388 },
                    "total_tokens": 6714,
                    "context_details": {
                        "input_tokens": 5022,
                        "output_tokens": 571
                    }
                }
            }
        }"#;
        let event = deserialize_response_event(sse).expect("parse");
        let rs::ResponseStreamEvent::ResponseCompleted(e) = event else {
            panic!("expected ResponseCompleted");
        };
        let usage = e.response.usage.expect("usage present");
        // Billing fields stay cumulative — unchanged by context_details.
        assert_eq!(usage.input_tokens, 6003);
        assert_eq!(usage.output_tokens, 711);
        assert_eq!(usage.input_tokens_details.cached_tokens, 1984);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 388);
        // total_tokens rewritten to ctx.input + ctx.output (5022 + 571).
        // NOT the wire's cumulative total (6714).
        assert_eq!(usage.total_tokens, 5_593);
    }

    #[test]
    fn deserialize_response_event_stashes_cost_in_metadata() {
        let make = |ticks: i64| {
            format!(
                r#"{{
                "type": "response.completed",
                "sequence_number": 0,
                "response": {{
                    "id": "resp_1", "object": "response", "created_at": 0,
                    "model": "grok-build", "status": "completed", "output": [],
                    "usage": {{
                        "input_tokens": 10,
                        "input_tokens_details": {{ "cached_tokens": 0 }},
                        "output_tokens": 5,
                        "output_tokens_details": {{ "reasoning_tokens": 0 }},
                        "total_tokens": 15,
                        "cost_in_usd_ticks": {ticks}
                    }}
                }}
            }}"#
            )
        };

        let event = deserialize_response_event(&make(78)).expect("parse");
        let rs::ResponseStreamEvent::ResponseCompleted(e) = event else {
            panic!("expected ResponseCompleted");
        };
        assert_eq!(
            e.response
                .metadata
                .as_ref()
                .and_then(|m| m.get(COST_USD_TICKS_METADATA_KEY))
                .map(String::as_str),
            Some("78")
        );

        // The REST mapper backfills 0 for unbilled requests: no stash.
        let event = deserialize_response_event(&make(0)).expect("parse");
        let rs::ResponseStreamEvent::ResponseCompleted(e) = event else {
            panic!("expected ResponseCompleted");
        };
        assert!(e.response.metadata.is_none());
    }

    #[test]
    fn deserialize_response_event_total_tokens_unchanged_when_context_details_absent() {
        // Older / non-Responses backends omit `context_details`.
        // `total_tokens` passes through from the wire unchanged.
        let sse = r#"{
            "type": "response.completed",
            "sequence_number": 0,
            "response": {
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": "grok-build",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 10000,
                    "input_tokens_details": { "cached_tokens": 0 },
                    "output_tokens": 100,
                    "output_tokens_details": { "reasoning_tokens": 0 },
                    "total_tokens": 10100
                }
            }
        }"#;
        let event = deserialize_response_event(sse).expect("parse");
        let rs::ResponseStreamEvent::ResponseCompleted(e) = event else {
            panic!("expected ResponseCompleted");
        };
        let usage = e.response.usage.expect("usage present");
        assert_eq!(usage.total_tokens, 10_100);
    }

    #[test]
    fn deserialize_response_event_total_tokens_unchanged_when_context_details_partial() {
        // Defensive: if the backend ever ships only one of the two
        // context_details fields, we don't have a complete picture of
        // the live context size, so leave `total_tokens` on the wire's
        // cumulative value instead of guessing (treating the missing
        // half as 0 would silently under-report).
        let sse = r#"{
            "type": "response.completed",
            "sequence_number": 0,
            "response": {
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": "grok-build",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 6003,
                    "input_tokens_details": { "cached_tokens": 1984 },
                    "output_tokens": 711,
                    "output_tokens_details": { "reasoning_tokens": 388 },
                    "total_tokens": 6714,
                    "context_details": {
                        "input_tokens": 5022
                    }
                }
            }
        }"#;
        let event = deserialize_response_event(sse).expect("parse");
        let rs::ResponseStreamEvent::ResponseCompleted(e) = event else {
            panic!("expected ResponseCompleted");
        };
        let usage = e.response.usage.expect("usage present");
        assert_eq!(usage.total_tokens, 6_714);
    }

    #[test]
    fn raw_output_item_events_capture_distinct_codex_metadata_before_typed_decode() {
        use xai_grok_sampling_types::{
            CapturedResponseOutputItemValue, DecodedResponseStreamEvent, ResponseOutputItemKind,
        };

        let cases = [
            (
                ResponseOutputItemKind::Message,
                "turn-message",
                serde_json::json!({
                    "type": "message", "id": "msg_1", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text", "text": "ok", "annotations": []}]
                }),
            ),
            (
                ResponseOutputItemKind::Reasoning,
                "turn-reasoning",
                serde_json::json!({
                    "type": "reasoning", "id": "rs_1", "summary": [],
                    "encrypted_content": "cipher"
                }),
            ),
            (
                ResponseOutputItemKind::FunctionCall,
                "turn-call",
                serde_json::json!({
                    "type": "function_call", "id": "fc_1", "call_id": "call_1",
                    "name": "read_file", "arguments": "{}"
                }),
            ),
            (
                ResponseOutputItemKind::FunctionCallOutput,
                "turn-output",
                serde_json::json!({
                    "type": "function_call_output", "id": "fco_1", "call_id": "call_1",
                    "output": "contents"
                }),
            ),
        ];

        for (output_index, (kind, turn_id, mut item)) in cases.into_iter().enumerate() {
            item["internal_chat_message_metadata_passthrough"] =
                serde_json::json!({"turn_id": turn_id});
            for event_type in ["response.output_item.added", "response.output_item.done"] {
                let raw = serde_json::json!({
                    "type": event_type,
                    "sequence_number": output_index,
                    "output_index": output_index,
                    "item": item,
                });
                let decoded = deserialize_response_event_with_metadata(&raw.to_string(), true)
                    .expect("raw item must decode");
                let captured = match decoded {
                    DecodedResponseStreamEvent::OutputItemAdded(item)
                    | DecodedResponseStreamEvent::OutputItemDone(item) => item,
                    other => panic!("expected raw output item, got {other:?}"),
                };
                assert_eq!(captured.kind(), Some(kind));
                assert_eq!(
                    captured
                        .internal_chat_message_metadata_passthrough
                        .as_ref()
                        .and_then(|metadata| metadata.turn_id.as_deref()),
                    Some(turn_id)
                );
                if kind == ResponseOutputItemKind::FunctionCallOutput {
                    assert!(matches!(
                        captured.value,
                        CapturedResponseOutputItemValue::FunctionCallOutput(_)
                    ));
                }
            }
        }
    }

    #[test]
    fn unary_raw_output_metadata_survives_durable_conversion() {
        let body = serde_json::json!({
            "id": "resp_1", "object": "response", "created_at": 0,
            "model": "gpt-codex", "status": "completed",
            "output": [{
                "type": "message", "id": "msg_unary", "role": "assistant",
                "status": "completed", "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-unary"}
            }]
        });
        let mut decoded = deserialize_unary_response(body.to_string().as_bytes(), true).unwrap();
        let origin = xai_grok_sampling_types::ResponseMetadataOrigin::codex(
            xai_grok_sampling_types::CODEX_BACKEND_BASE_URL,
            "gpt-codex",
            None,
        )
        .unwrap();
        decoded.set_metadata_origin(Some(&origin));
        let durable = decoded.into_conversation_items().unwrap();
        assert!(matches!(
            &durable[0],
            ConversationItem::Provider(provider)
                if provider.as_response_output_metadata().is_some_and(|metadata| {
                    metadata.items[0]
                        .internal_chat_message_metadata_passthrough
                        .as_ref()
                        .and_then(|value| value.turn_id.as_deref())
                        == Some("turn-unary")
                        && metadata.items[0].item_id.as_deref() == Some("msg_unary")
                })
        ));
        assert!(matches!(&durable[1], ConversationItem::Assistant(_)));
    }

    #[test]
    fn raw_streaming_empty_completed_output_replays_exact_interleaving_after_cold_load() {
        let origin = xai_grok_sampling_types::ResponseMetadataOrigin::codex(
            xai_grok_sampling_types::CODEX_BACKEND_BASE_URL,
            "gpt-codex",
            None,
        )
        .unwrap();
        let output = [
            serde_json::json!({
                "type": "reasoning", "id": "rs-s0", "summary": [],
                "encrypted_content": "cipher-s0", "status": "completed",
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-s0"}
            }),
            serde_json::json!({
                "type": "function_call", "id": "fc-s1", "call_id": "call-s1",
                "name": "read_file", "arguments": "{}",
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-s1"}
            }),
            serde_json::json!({
                "type": "reasoning", "id": "rs-s2", "summary": [],
                "encrypted_content": "cipher-s2", "status": "completed",
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-s2"}
            }),
            serde_json::json!({
                "type": "function_call", "id": "fc-s3", "call_id": "call-s3",
                "name": "grep", "arguments": "{}",
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-s3"}
            }),
            serde_json::json!({
                "type": "message", "id": "msg-s4", "role": "assistant",
                "status": "completed", "content": [
                    {"type": "output_text", "text": "streamed", "annotations": []}
                ], "internal_chat_message_metadata_passthrough": {"turn_id": "turn-s4"}
            }),
        ];
        let mut accumulator = xai_grok_sampling_types::ResponsesStreamAccumulator::default();
        for (output_index, item) in output.into_iter().enumerate() {
            let raw = serde_json::json!({
                "type": "response.output_item.done",
                "sequence_number": output_index,
                "output_index": output_index,
                "item": item,
            });
            let decoded = deserialize_response_event_with_metadata(&raw.to_string(), true)
                .unwrap()
                .with_metadata_origin(Some(&origin));
            let xai_grok_sampling_types::DecodedResponseStreamEvent::OutputItemDone(item) = decoded
            else {
                panic!("raw output item event")
            };
            accumulator.note_captured_output_item_done(item);
        }
        let completed = serde_json::json!({
            "type": "response.completed", "sequence_number": 99,
            "response": {
                "id": "resp-stream-order", "object": "response", "created_at": 0,
                "model": "gpt-codex", "status": "completed", "output": []
            }
        });
        let decoded = deserialize_response_event_with_metadata(&completed.to_string(), true)
            .unwrap()
            .with_metadata_origin(Some(&origin));
        let xai_grok_sampling_types::DecodedResponseStreamEvent::Event {
            event: rs::ResponseStreamEvent::ResponseCompleted(completed),
            terminal_output,
        } = decoded
        else {
            panic!("raw completed event")
        };
        let captured = accumulator.terminal_output(terminal_output);
        let items = xai_grok_sampling_types::captured_response_to_conversation_items(
            completed.response,
            captured,
        )
        .unwrap();
        let items: Vec<ConversationItem> =
            serde_json::from_slice(&serde_json::to_vec(&items).unwrap()).unwrap();
        let request = xai_grok_sampling_types::ConversationRequest {
            items,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        let mut wire = serde_json::to_value(
            xai_grok_sampling_types::conversation_request_to_codex_create_response(&request),
        )
        .unwrap();
        xai_grok_sampling_types::patch_response_message_item_ids(
            &mut wire,
            &xai_grok_sampling_types::response_message_item_ids(&request),
        );
        let metadata = xai_grok_sampling_types::response_item_metadata_passthrough_for_origin(
            &request,
            Some(&origin),
        )
        .unwrap();
        xai_grok_sampling_types::patch_response_item_metadata_passthrough(&mut wire, &metadata)
            .unwrap();
        let input = wire["input"].as_array().unwrap();
        assert_eq!(
            input
                .iter()
                .map(|item| (
                    item["type"].as_str().unwrap(),
                    item.get("id").and_then(serde_json::Value::as_str),
                ))
                .collect::<Vec<_>>(),
            [
                ("reasoning", Some("rs-s0")),
                ("function_call", Some("fc-s1")),
                ("reasoning", Some("rs-s2")),
                ("function_call", Some("fc-s3")),
                ("message", Some("msg-s4")),
            ]
        );
        assert_eq!(input[1]["call_id"], "call-s1");
        assert_eq!(input[3]["call_id"], "call-s3");
        for (index, turn_id) in ["turn-s0", "turn-s1", "turn-s2", "turn-s3", "turn-s4"]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                input[index]["internal_chat_message_metadata_passthrough"]["turn_id"],
                turn_id
            );
        }
    }

    #[test]
    fn unary_interleaved_output_replays_in_exact_provider_order() {
        let body = serde_json::json!({
            "id": "resp-unary-order", "object": "response", "created_at": 0,
            "model": "gpt-codex", "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs-u0", "summary": [],
                 "encrypted_content": "cipher-u0", "status": "completed"},
                {"type": "function_call", "id": "fc-u1", "call_id": "call-u1",
                 "name": "read_file", "arguments": "{}"},
                {"type": "reasoning", "id": "rs-u2", "summary": [],
                 "encrypted_content": "cipher-u2", "status": "completed"},
                {"type": "function_call", "id": "fc-u3", "call_id": "call-u3",
                 "name": "grep", "arguments": "{}"},
                {"type": "message", "id": "msg-u4", "role": "assistant",
                 "status": "completed", "content": [
                    {"type": "output_text", "text": "unary", "annotations": []}
                 ]}
            ]
        });
        let mut decoded = deserialize_unary_response(body.to_string().as_bytes(), true).unwrap();
        let origin = xai_grok_sampling_types::ResponseMetadataOrigin::codex(
            xai_grok_sampling_types::CODEX_BACKEND_BASE_URL,
            "gpt-codex",
            None,
        )
        .unwrap();
        decoded.set_metadata_origin(Some(&origin));
        let items: Vec<ConversationItem> = serde_json::from_slice(
            &serde_json::to_vec(&decoded.into_conversation_items().unwrap()).unwrap(),
        )
        .unwrap();
        let request = xai_grok_sampling_types::ConversationRequest {
            items,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        let mut wire = serde_json::to_value(
            xai_grok_sampling_types::conversation_request_to_codex_create_response(&request),
        )
        .unwrap();
        xai_grok_sampling_types::patch_response_message_item_ids(
            &mut wire,
            &xai_grok_sampling_types::response_message_item_ids(&request),
        );
        let metadata = xai_grok_sampling_types::response_item_metadata_passthrough_for_origin(
            &request,
            Some(&origin),
        )
        .unwrap();
        xai_grok_sampling_types::patch_response_item_metadata_passthrough(&mut wire, &metadata)
            .unwrap();
        let input = wire["input"].as_array().unwrap();
        assert_eq!(
            input
                .iter()
                .map(|item| (
                    item["type"].as_str().unwrap(),
                    item.get("id").and_then(serde_json::Value::as_str),
                ))
                .collect::<Vec<_>>(),
            [
                ("reasoning", Some("rs-u0")),
                ("function_call", Some("fc-u1")),
                ("reasoning", Some("rs-u2")),
                ("function_call", Some("fc-u3")),
                ("message", Some("msg-u4")),
            ]
        );
    }

    #[tokio::test]
    async fn production_compact_request_restores_interleaved_response_order() {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let sink = std::sync::Arc::clone(&captured);
        let app = axum::Router::new().route(
            "/v1/responses/compact",
            axum::routing::post(move |request: axum::extract::Request| {
                let sink = std::sync::Arc::clone(&sink);
                async move {
                    let bytes = axum::body::to_bytes(request.into_body(), usize::MAX)
                        .await
                        .unwrap();
                    *sink.lock().unwrap() =
                        Some(serde_json::from_slice::<serde_json::Value>(&bytes).unwrap());
                    axum::Json(serde_json::json!({"output": []}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let body = serde_json::json!({
            "id": "resp-compact-order", "object": "response", "created_at": 0,
            "model": "gpt-codex", "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs-c0", "summary": [],
                 "encrypted_content": "cipher-c0", "status": "completed",
                 "internal_chat_message_metadata_passthrough": {"turn_id": "turn-c0"}},
                {"type": "function_call", "id": "fc-c1", "call_id": "call-c1",
                 "name": "read_file", "arguments": "{}",
                 "internal_chat_message_metadata_passthrough": {"turn_id": "turn-c1"}},
                {"type": "reasoning", "id": "rs-c2", "summary": [],
                 "encrypted_content": "cipher-c2", "status": "completed",
                 "internal_chat_message_metadata_passthrough": {"turn_id": "turn-c2"}},
                {"type": "function_call", "id": "fc-c3", "call_id": "call-c3",
                 "name": "grep", "arguments": "{}",
                 "internal_chat_message_metadata_passthrough": {"turn_id": "turn-c3"}},
                {"type": "message", "id": "msg-c4", "role": "assistant",
                 "status": "completed", "content": [
                    {"type": "output_text", "text": "compact", "annotations": []}
                 ], "internal_chat_message_metadata_passthrough": {"turn_id": "turn-c4"}}
            ]
        });
        let mut decoded = deserialize_unary_response(body.to_string().as_bytes(), true).unwrap();
        let origin = xai_grok_sampling_types::ResponseMetadataOrigin::codex(
            xai_grok_sampling_types::CODEX_BACKEND_BASE_URL,
            "gpt-codex",
            None,
        )
        .unwrap();
        decoded.set_metadata_origin(Some(&origin));
        let items: Vec<ConversationItem> = serde_json::from_slice(
            &serde_json::to_vec(&decoded.into_conversation_items().unwrap()).unwrap(),
        )
        .unwrap();

        let config = SamplerConfig {
            api_key: Some("test-token".into()),
            base_url: format!("http://{address}/v1"),
            model: "gpt-codex".into(),
            api_backend: ApiBackend::Responses,
            provider_id: Some(xai_grok_sampling_types::ProviderId::Codex),
            ..SamplerConfig::default()
        };
        let client = SamplingClient::new(config).unwrap();
        client
            .conversation_compact_responses(xai_grok_sampling_types::ConversationRequest {
                items,
                model: Some("gpt-codex".into()),
                ..Default::default()
            })
            .await
            .unwrap();

        let body = captured.lock().unwrap().take().unwrap();
        let input = body["input"].as_array().unwrap();
        assert_eq!(
            input
                .iter()
                .map(|item| (
                    item["type"].as_str().unwrap(),
                    item.get("id").and_then(serde_json::Value::as_str),
                ))
                .collect::<Vec<_>>(),
            [
                ("reasoning", Some("rs-c0")),
                ("function_call", Some("fc-c1")),
                ("reasoning", Some("rs-c2")),
                ("function_call", Some("fc-c3")),
                ("message", Some("msg-c4")),
            ]
        );
        assert_eq!(input[1]["call_id"], "call-c1");
        assert_eq!(input[3]["call_id"], "call-c3");
        for (index, turn_id) in ["turn-c0", "turn-c1", "turn-c2", "turn-c3", "turn-c4"]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                input[index]["internal_chat_message_metadata_passthrough"]["turn_id"],
                turn_id
            );
        }
    }

    #[test]
    fn unsupported_metadata_variant_fails_closed_only_for_codex() {
        let raw = serde_json::json!({
            "type": "response.output_item.done", "sequence_number": 1, "output_index": 0,
            "item": {
                "type": "image_generation_call", "id": "ig_1", "status": "completed",
                "result": "image", "internal_chat_message_metadata_passthrough": {"turn_id": "turn-image"}
            }
        })
        .to_string();
        let error = deserialize_response_event_with_metadata(&raw, true).unwrap_err();
        assert!(error.to_string().contains("cannot be replayed exactly"));
        let non_codex = deserialize_response_event_with_metadata(&raw, false).unwrap();
        let xai_grok_sampling_types::DecodedResponseStreamEvent::OutputItemDone(item) = non_codex
        else {
            panic!("expected decoded non-Codex output item")
        };
        assert!(item.internal_chat_message_metadata_passthrough.is_none());
    }

    #[test]
    fn deserialize_response_event_ignores_context_details_on_non_terminal_events() {
        // Non-terminal events don't carry final usage; even if the backend ever
        // echoed `context_details` on one, we don't touch it.
        let sse = r#"{
            "type": "response.output_text.delta",
            "sequence_number": 0,
            "item_id": "item-1",
            "output_index": 0,
            "content_index": 0,
            "delta": "hello",
            "logprobs": []
        }"#;
        let event = deserialize_response_event(sse).expect("non-terminal event parses");
        assert!(matches!(
            event,
            rs::ResponseStreamEvent::ResponseOutputTextDelta(_)
        ));
    }
}
