//! Codex-native compaction policy and transport adapter.

use crate::session::helpers::session_compact::is_context_length_error;
use xai_grok_sampler::SamplingClient;
use xai_grok_sampling_types::{
    ConversationItem, HostedTool, ReasoningEffort, SamplingError, SamplingIdentity, ToolSpec,
    TurnRoutingState,
};

const MAX_RETRIES: u32 = 3;
const RETRY_DELAY_SECS: u64 = 3;
const TRUNCATED_TOOL_OUTPUT: &str = "Output exceeded the available model context and was truncated";

/// Compaction implementation selection. Native replacement history and local
/// summary input fitting are different implementations, not stages of one algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum CompactionStrategy {
    /// Unary Codex `/responses/compact`; installs provider replacement items.
    NativeCodex,
    /// Existing ordinary `/responses` summarization and local history rebuild.
    LocalSummary,
}

impl CompactionStrategy {
    pub(in super::super) fn as_str(self) -> &'static str {
        match self {
            Self::NativeCodex => "native_codex",
            Self::LocalSummary => "local_summary",
        }
    }

    pub(in super::super) fn uses_local_summary_pipeline(self) -> bool {
        self == Self::LocalSummary
    }
}

/// Resolve Codex's opt-in policy and the command-level compatibility contract.
/// `/compact <context>` remains local because the native endpoint has no field
/// for user-authored summary guidance; silently discarding it is unsafe.
pub(in super::super) fn select_compaction_strategy(
    capabilities: xai_grok_sampling_types::ProviderCapabilities,
    override_value: Option<&str>,
    user_context_supplied: bool,
) -> CompactionStrategy {
    if !capabilities.supports_native_compact() {
        return CompactionStrategy::LocalSummary;
    }

    let strategy = match override_value
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("native" | "native_codex") => CompactionStrategy::NativeCodex,
        Some("local" | "local_summary") | None | Some("") => CompactionStrategy::LocalSummary,
        Some(other) => {
            tracing::warn!(
                value = other,
                "unknown GROK_CODEX_COMPACTION_STRATEGY; native compaction requires explicit `native` or `native_codex`, using local summary"
            );
            CompactionStrategy::LocalSummary
        }
    };

    if user_context_supplied && strategy == CompactionStrategy::NativeCodex {
        CompactionStrategy::LocalSummary
    } else {
        strategy
    }
}

/// Owned snapshots needed by one native request lifecycle. In particular, no
/// session actor or `RefCell` borrow can be retained while the adapter awaits.
pub(in super::super) struct CodexCompactionInput {
    pub source_conversation: Vec<ConversationItem>,
    pub canonical_system_message: ConversationItem,
    pub tools: Vec<ToolSpec>,
    pub hosted_tools: Vec<HostedTool>,
    /// Exact effective identity snapshot held by the client performing this
    /// native request, including its finalized environment-header account.
    pub sampling_identity: SamplingIdentity,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub session_id: String,
    pub turn_routing_state: Option<TurnRoutingState>,
    pub context_window: u64,
    pub tool_tokens: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in super::super) struct NativeCompactionCounters {
    pub attempts: u32,
    pub transient_rejections: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum NativeCompactionFallbackReason {
    StructuralMinimumTooLarge,
    EndpointUnavailable,
    ContextOverflow,
}

#[derive(Debug)]
pub(in super::super) enum NativeCompactionFailureSource {
    Sampling(SamplingError),
    InvalidResponse(String),
}

impl NativeCompactionFailureSource {
    pub(in super::super) fn message(&self) -> String {
        match self {
            Self::Sampling(error) => format!("native compact failed: {error}"),
            Self::InvalidResponse(message) => {
                format!("native compact response invalid: {message}")
            }
        }
    }

    pub(in super::super) fn suppresses_auto_compaction(&self) -> bool {
        // Preserve the prior orchestration contract: terminal request errors
        // suppress another automatic attempt, while a successfully decoded HTTP
        // response that fails replacement validation returns immediately.
        matches!(self, Self::Sampling(_))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in super::super) struct NativeCompactionOutput {
    pub stop_reason: String,
    pub truncated: bool,
    pub ttft_ms: Option<u64>,
    pub stream_ms: Option<u64>,
    pub delta_count: u64,
    pub itl_max_ms: Option<u64>,
}

pub(in super::super) enum NativeCompactionOutcome {
    Success {
        replacement_history: Vec<ConversationItem>,
        output: NativeCompactionOutput,
        counters: NativeCompactionCounters,
    },
    LocalFallback {
        reason: NativeCompactionFallbackReason,
        counters: NativeCompactionCounters,
    },
    HardFailure {
        source: NativeCompactionFailureSource,
        counters: NativeCompactionCounters,
        estimated_input_tokens: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NativePreflightResult {
    rewritten_outputs: usize,
    initial_tokens: i128,
    final_tokens: i128,
}

fn compatibility_for_sampling_identity(
    identity: &SamplingIdentity,
) -> xai_grok_sampling_types::NativeCompactionCompatibility {
    xai_grok_sampling_types::NativeCompactionCompatibility::codex(
        identity.model.clone(),
        identity.chatgpt_account_id.clone(),
    )
}

/// Execute the complete Codex-native compaction lifecycle. The sampling client
/// is borrowed so generic orchestration can still move it into the local
/// summary sampler if this adapter explicitly requests fallback.
pub(in super::super) async fn run_native_compaction(
    mut input: CodexCompactionInput,
    client: &SamplingClient,
) -> NativeCompactionOutcome {
    let preflight = trim_tool_outputs_to_fit_context_window(
        &mut input.source_conversation,
        input.context_window,
        input.tool_tokens,
    );
    let estimated_input_tokens =
        xai_chat_state::estimate_conversation_tokens(&input.source_conversation);
    let client_identity = client.sampling_identity_for_model(&input.sampling_identity.model);
    if client_identity != input.sampling_identity {
        return NativeCompactionOutcome::HardFailure {
            source: NativeCompactionFailureSource::InvalidResponse(
                "native compact identity snapshot does not match the performing client".into(),
            ),
            counters: NativeCompactionCounters::default(),
            estimated_input_tokens,
        };
    }
    tracing::info!(
        rewritten_outputs = preflight.rewritten_outputs,
        initial_tokens = %preflight.initial_tokens,
        final_tokens = %preflight.final_tokens,
        context_window = input.context_window,
        "Prepared native Codex compaction input"
    );
    if preflight.final_tokens > i128::from(input.context_window) {
        tracing::warn!(
            final_tokens = %preflight.final_tokens,
            context_window = input.context_window,
            "Native compaction structural minimum still exceeds context; using fitted local-summary ladder without sending an oversized native request"
        );
        return NativeCompactionOutcome::LocalFallback {
            reason: NativeCompactionFallbackReason::StructuralMinimumTooLarge,
            counters: NativeCompactionCounters::default(),
        };
    }

    let compatibility = compatibility_for_sampling_identity(&input.sampling_identity);
    let has_tools = !input.tools.is_empty();
    let request = xai_grok_sampling_types::ConversationRequest {
        items: input.source_conversation,
        tools: input.tools,
        hosted_tools: input.hosted_tools,
        tool_choice: has_tools.then_some(xai_grok_sampling_types::ConversationToolChoice::Auto),
        model: Some(input.sampling_identity.model.clone()),
        reasoning_effort: input.reasoning_effort,
        x_grok_conv_id: Some(input.session_id.clone()),
        x_grok_req_id: Some(format!("xai-native-compact-{}", uuid::Uuid::new_v4())),
        x_grok_session_id: Some(input.session_id.clone()),
        x_grok_agent_id: Some(xai_grok_telemetry::id::agent_id()),
        prompt_cache_key: Some(input.session_id),
        turn_routing_state: input.turn_routing_state,
        ..Default::default()
    };
    let started = std::time::Instant::now();
    let mut counters = NativeCompactionCounters::default();

    loop {
        counters.attempts += 1;
        match client.conversation_compact_responses(request.clone()).await {
            Ok(response) => {
                let mut replacement =
                    match xai_grok_sampling_types::codex_compact_output_to_conversation(
                        response.output,
                        compatibility.clone(),
                    ) {
                        Ok(replacement) => replacement,
                        Err(message) => {
                            return NativeCompactionOutcome::HardFailure {
                                source: NativeCompactionFailureSource::InvalidResponse(message),
                                counters,
                                estimated_input_tokens,
                            };
                        }
                    };
                if replacement.is_empty()
                    || !replacement.iter().any(|item| {
                        matches!(item, ConversationItem::Provider(provider) if provider.is_encrypted_compaction())
                    })
                {
                    return NativeCompactionOutcome::HardFailure {
                        source: NativeCompactionFailureSource::InvalidResponse(
                            "replacement history must contain an encrypted compaction item".into(),
                        ),
                        counters,
                        estimated_input_tokens,
                    };
                }

                // Codex receives the leading System item via `instructions` and
                // does not echo it. Restore the canonical item so replay
                // reconstructs identical instructions without a local summary.
                replacement.insert(0, input.canonical_system_message);
                return NativeCompactionOutcome::Success {
                    replacement_history: replacement,
                    output: NativeCompactionOutput {
                        stop_reason: "native_replacement".to_string(),
                        truncated: false,
                        // Unary compaction has no first-token event. Its full
                        // request latency is the only native timing signal.
                        ttft_ms: None,
                        stream_ms: Some(started.elapsed().as_millis() as u64),
                        delta_count: 0,
                        itl_max_ms: None,
                    },
                    counters,
                };
            }
            Err(error) if endpoint_unavailable(&error) => {
                tracing::warn!(
                    %error,
                    "Codex compact endpoint unavailable; using explicit local-summary fallback"
                );
                return NativeCompactionOutcome::LocalFallback {
                    reason: NativeCompactionFallbackReason::EndpointUnavailable,
                    counters,
                };
            }
            Err(error) if context_overflow(&error) => {
                tracing::warn!(
                    %error,
                    "Codex compact endpoint rejected the safely rewritten input as oversized; using fitted local-summary ladder without retrying identical native input"
                );
                return NativeCompactionOutcome::LocalFallback {
                    reason: NativeCompactionFallbackReason::ContextOverflow,
                    counters,
                };
            }
            Err(error) if error.is_retryable() && counters.attempts < MAX_RETRIES => {
                counters.transient_rejections += 1;
                tracing::warn!(
                    attempt = counters.attempts,
                    %error,
                    "transient native Codex compaction failure; retrying native endpoint"
                );
                tokio::time::sleep(std::time::Duration::from_secs(RETRY_DELAY_SECS)).await;
            }
            Err(error) => {
                return NativeCompactionOutcome::HardFailure {
                    source: NativeCompactionFailureSource::Sampling(error),
                    counters,
                    estimated_input_tokens,
                };
            }
        }
    }
}

/// Preserve call/result topology while reducing only a contiguous suffix of
/// tool outputs. This mirrors Codex's remote-compaction preflight: walk from
/// newest to oldest, stop at the first ineligible item, and never let a
/// saturating public estimate conceal overflow.
fn trim_tool_outputs_to_fit_context_window(
    items: &mut Vec<ConversationItem>,
    context_window: u64,
    tool_tokens: u64,
) -> NativePreflightResult {
    let estimates = items
        .iter()
        .map(xai_chat_state::estimate_item_tokens)
        .collect::<Vec<_>>();
    let mut total = estimates
        .iter()
        .copied()
        .map(i128::from)
        .fold(i128::from(tool_tokens), i128::saturating_add);
    let initial_tokens = total;
    let mut rewritten = Vec::new();

    for (item, item_tokens) in items.iter().zip(estimates).rev() {
        if total <= i128::from(context_window) {
            break;
        }
        let ConversationItem::ToolResult(output) = item else {
            break;
        };
        let replacement = ConversationItem::ToolResult(xai_grok_sampling_types::ToolResultItem {
            tool_call_id: output.tool_call_id.clone(),
            content: TRUNCATED_TOOL_OUTPUT.into(),
            images: Vec::new(),
        });
        total = total
            .saturating_sub(i128::from(item_tokens))
            .saturating_add(i128::from(xai_chat_state::estimate_item_tokens(
                &replacement,
            )));
        rewritten.push(replacement);
    }

    let rewritten_outputs = rewritten.len();
    if rewritten_outputs > 0 {
        let retained = items.len() - rewritten_outputs;
        items.truncate(retained);
        items.extend(rewritten.into_iter().rev());
    }
    NativePreflightResult {
        rewritten_outputs,
        initial_tokens,
        final_tokens: total,
    }
}

/// Only the endpoint's explicit capability statuses permit local fallback.
fn endpoint_unavailable(error: &SamplingError) -> bool {
    matches!(
        error,
        SamplingError::Api { status, .. }
            if matches!(
                *status,
                reqwest::StatusCode::NOT_FOUND
                    | reqwest::StatusCode::METHOD_NOT_ALLOWED
                    | reqwest::StatusCode::NOT_IMPLEMENTED
            )
    )
}

fn context_overflow(error: &SamplingError) -> bool {
    match error {
        SamplingError::Api { message, .. } | SamplingError::StreamError { message, .. } => {
            is_context_length_error(message)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CodexCompactionInput, CompactionStrategy, NativeCompactionFailureSource,
        NativeCompactionFallbackReason, NativeCompactionOutcome, TRUNCATED_TOOL_OUTPUT,
        compatibility_for_sampling_identity, context_overflow, endpoint_unavailable,
        run_native_compaction, select_compaction_strategy, trim_tool_outputs_to_fit_context_window,
    };
    use axum::Json;
    use axum::Router;
    use axum::routing::post;
    use reqwest::StatusCode;
    use serde_json::json;
    use tokio::net::TcpListener;
    use xai_grok_sampler::{SamplerConfig, SamplingClient};
    use xai_grok_sampling_types::{
        ApiBackend, CODEX_BACKEND_BASE_URL, ConversationItem, ProviderCapabilities, ProviderId,
        SamplingError, resolve_provider,
    };

    fn provider_capabilities(provider_id: ProviderId) -> ProviderCapabilities {
        resolve_provider(
            Some(provider_id),
            ApiBackend::Responses,
            "http://127.0.0.1:3210/v1",
        )
        .capabilities()
    }

    fn api_error(status: StatusCode) -> SamplingError {
        SamplingError::Api {
            status,
            message: "fixture".into(),
            model_metadata: None,
            retry_after_secs: None,
            should_retry: None,
        }
    }

    #[test]
    fn native_requires_an_explicit_valid_opt_in() {
        for value in [None, Some(""), Some("local"), Some("bogus")] {
            assert_eq!(
                select_compaction_strategy(provider_capabilities(ProviderId::Codex), value, false,),
                CompactionStrategy::LocalSummary
            );
        }
        for value in [Some("native"), Some(" native_codex ")] {
            assert_eq!(
                select_compaction_strategy(provider_capabilities(ProviderId::Codex), value, false,),
                CompactionStrategy::NativeCodex
            );
        }
    }

    #[test]
    fn non_codex_and_manual_guidance_remain_local() {
        assert_eq!(
            select_compaction_strategy(
                provider_capabilities(ProviderId::OpenAiCompatible),
                Some("native"),
                false,
            ),
            CompactionStrategy::LocalSummary
        );
        assert_eq!(
            select_compaction_strategy(provider_capabilities(ProviderId::Codex), None, true,),
            CompactionStrategy::LocalSummary
        );
    }

    #[test]
    fn native_is_not_a_prefit_or_two_pass_local_summary_variant() {
        assert!(!CompactionStrategy::NativeCodex.uses_local_summary_pipeline());
        assert!(CompactionStrategy::LocalSummary.uses_local_summary_pipeline());
    }

    #[test]
    fn fallback_is_limited_to_explicit_native_policy() {
        for status in [
            StatusCode::NOT_FOUND,
            StatusCode::METHOD_NOT_ALLOWED,
            StatusCode::NOT_IMPLEMENTED,
        ] {
            assert!(endpoint_unavailable(&api_error(status)), "{status}");
        }
        for status in [
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::REQUEST_TIMEOUT,
            StatusCode::CONFLICT,
            StatusCode::PAYLOAD_TOO_LARGE,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
            StatusCode::SERVICE_UNAVAILABLE,
        ] {
            assert!(!endpoint_unavailable(&api_error(status)), "{status}");
        }
        let overflow = SamplingError::Api {
            status: StatusCode::BAD_REQUEST,
            message: "input exceeds the context window".into(),
            model_metadata: None,
            retry_after_secs: None,
            should_retry: Some(false),
        };
        assert!(!endpoint_unavailable(&overflow));
        assert!(context_overflow(&overflow));
        assert!(!context_overflow(&api_error(StatusCode::BAD_REQUEST)));
        assert!(!endpoint_unavailable(&SamplingError::Auth(
            "expired".into()
        )));
        assert!(!endpoint_unavailable(
            &SamplingError::serialization_message("malformed compact response")
        ));
    }

    #[test]
    fn native_preflight_rewrites_multiple_trailing_outputs_and_preserves_ids() {
        let mut items = vec![
            ConversationItem::user("prompt"),
            ConversationItem::tool_result("call-a", &"a".repeat(4_000)),
            ConversationItem::tool_result("call-b", &"b".repeat(4_000)),
        ];
        let result = trim_tool_outputs_to_fit_context_window(&mut items, 200, 0);
        assert_eq!(result.rewritten_outputs, 2);
        assert!(result.final_tokens <= 200);
        for (item, expected_id) in items[1..].iter().zip(["call-a", "call-b"]) {
            let ConversationItem::ToolResult(output) = item else {
                panic!("tool-result topology must be retained")
            };
            assert_eq!(output.tool_call_id, expected_id);
            assert_eq!(&*output.content, TRUNCATED_TOOL_OUTPUT);
        }
    }

    #[test]
    fn native_preflight_does_not_rewrite_when_fitting_or_cross_ineligible_suffix() {
        let fitting = vec![ConversationItem::tool_result("call-a", "small")];
        let mut untouched = fitting.clone();
        let result = trim_tool_outputs_to_fit_context_window(&mut untouched, 100, 0);
        assert_eq!(result.rewritten_outputs, 0);
        assert_eq!(
            xai_chat_state::estimate_conversation_tokens(&untouched),
            xai_chat_state::estimate_conversation_tokens(&fitting)
        );

        let mut blocked = vec![
            ConversationItem::tool_result("old-large", &"x".repeat(8_000)),
            ConversationItem::assistant("suffix blocks backward rewriting"),
        ];
        let result = trim_tool_outputs_to_fit_context_window(&mut blocked, 10, 0);
        assert_eq!(result.rewritten_outputs, 0);
        let ConversationItem::ToolResult(output) = &blocked[0] else {
            panic!()
        };
        assert_eq!(output.content.len(), 8_000);
        assert!(result.final_tokens > 10);
    }

    #[test]
    fn native_preflight_reports_structural_minimum_over_limit() {
        let mut items = vec![ConversationItem::tool_result("call-a", &"x".repeat(8_000))];
        let result = trim_tool_outputs_to_fit_context_window(&mut items, 1, 100);
        assert_eq!(result.rewritten_outputs, 1);
        assert!(result.final_tokens > 1);
    }

    fn adapter_input(
        source_conversation: Vec<ConversationItem>,
        client: &SamplingClient,
    ) -> CodexCompactionInput {
        CodexCompactionInput {
            source_conversation,
            canonical_system_message: ConversationItem::system("canonical instructions"),
            tools: Vec::new(),
            hosted_tools: Vec::new(),
            sampling_identity: client.sampling_identity_for_model("gpt-test"),
            reasoning_effort: None,
            session_id: "session-adapter".into(),
            turn_routing_state: None,
            context_window: 128_000,
            tool_tokens: 0,
        }
    }

    fn adapter_client(base_url: String) -> SamplingClient {
        let mut config = SamplerConfig {
            api_key: Some("test-token".into()),
            base_url,
            model: "gpt-test".into(),
            api_backend: xai_grok_sampling_types::ApiBackend::Responses,
            ..Default::default()
        };
        config.extra_headers.insert(
            xai_grok_sampling_types::CHATGPT_ACCOUNT_ID_HEADER.into(),
            "acct-adapter".into(),
        );
        SamplingClient::new(config).unwrap()
    }

    async fn spawn_compact_server(app: Router) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{addr}/v1")
    }

    fn native_replacement_for_identity(
        identity: &xai_grok_sampling_types::SamplingIdentity,
    ) -> Vec<ConversationItem> {
        let output = serde_json::from_value(json!([{
            "type": "compaction",
            "id": "cmp-runtime-account",
            "encrypted_content": "opaque-runtime-account-history"
        }]))
        .unwrap();
        xai_grok_sampling_types::codex_compact_output_to_conversation(
            output,
            compatibility_for_sampling_identity(identity),
        )
        .unwrap()
    }

    fn runtime_account_client(env_var: &str, static_account: Option<&str>) -> SamplingClient {
        let mut config = SamplerConfig {
            base_url: CODEX_BACKEND_BASE_URL.into(),
            model: "gpt-runtime-account".into(),
            api_backend: xai_grok_sampling_types::ApiBackend::Responses,
            ..Default::default()
        };
        if let Some(account) = static_account {
            config.extra_headers.insert(
                xai_grok_sampling_types::CHATGPT_ACCOUNT_ID_HEADER.into(),
                account.into(),
            );
        }
        config.env_http_headers.insert(
            xai_grok_sampling_types::CHATGPT_ACCOUNT_ID_HEADER.into(),
            env_var.into(),
        );
        SamplingClient::new(config).unwrap()
    }

    #[test]
    #[serial_test::serial]
    fn env_only_account_native_replacement_is_immediately_replay_compatible() {
        let _account = xai_grok_test_support::env::EnvGuard::set(
            "XAI_NATIVE_COMPACT_ENV_ONLY_ACCOUNT",
            "acct-env-only",
        );
        let client = runtime_account_client("XAI_NATIVE_COMPACT_ENV_ONLY_ACCOUNT", None);
        let identity = client.sampling_identity_for_model("gpt-runtime-account");
        let replacement = native_replacement_for_identity(&identity);

        xai_grok_sampling_types::validate_history_for_sampling_identity(&replacement, &identity)
            .unwrap();
        assert_eq!(
            identity.chatgpt_account_id.as_deref(),
            Some("acct-env-only")
        );
    }

    #[test]
    #[serial_test::serial]
    fn env_account_override_native_replacement_is_immediately_replay_compatible() {
        let _account = xai_grok_test_support::env::EnvGuard::set(
            "XAI_NATIVE_COMPACT_OVERRIDE_ACCOUNT",
            "acct-env-override",
        );
        let client =
            runtime_account_client("XAI_NATIVE_COMPACT_OVERRIDE_ACCOUNT", Some("acct-static"));
        let identity = client.sampling_identity_for_model("gpt-runtime-account");
        let replacement = native_replacement_for_identity(&identity);

        xai_grok_sampling_types::validate_history_for_sampling_identity(&replacement, &identity)
            .unwrap();
        assert_eq!(
            identity.chatgpt_account_id.as_deref(),
            Some("acct-env-override")
        );
    }

    #[tokio::test]
    async fn adapter_structural_minimum_falls_back_without_an_http_attempt() {
        let client = adapter_client("http://127.0.0.1:1/v1".into());
        let mut input = adapter_input(
            vec![ConversationItem::tool_result("call-a", &"x".repeat(8_000))],
            &client,
        );
        input.context_window = 1;
        input.tool_tokens = 100;

        assert!(matches!(
            run_native_compaction(input, &client).await,
            NativeCompactionOutcome::LocalFallback {
                reason: NativeCompactionFallbackReason::StructuralMinimumTooLarge,
                counters,
            } if counters.attempts == 0 && counters.transient_rejections == 0
        ));
    }

    #[tokio::test]
    async fn adapter_restores_canonical_system_and_validates_encrypted_output() {
        let app = Router::new().route(
            "/v1/responses/compact",
            post(|| async {
                Json(json!({
                    "output": [{
                        "type": "compaction",
                        "id": "cmp-adapter",
                        "encrypted_content": "opaque-adapter-history"
                    }]
                }))
            }),
        );
        let client = adapter_client(spawn_compact_server(app).await);
        let input = adapter_input(
            vec![
                ConversationItem::system("canonical instructions"),
                ConversationItem::user("objective"),
            ],
            &client,
        );

        let NativeCompactionOutcome::Success {
            replacement_history,
            output,
            counters,
        } = run_native_compaction(input, &client).await
        else {
            panic!("valid encrypted native output must succeed")
        };
        assert!(matches!(
            replacement_history.first(),
            Some(ConversationItem::System(system)) if &*system.content == "canonical instructions"
        ));
        assert!(replacement_history.iter().any(|item| {
            matches!(item, ConversationItem::Provider(provider) if provider.is_encrypted_compaction())
        }));
        let compatibility =
            xai_grok_sampling_types::native_compaction_compatibility(&replacement_history)
                .unwrap()
                .unwrap();
        assert_eq!(compatibility.model, "gpt-test");
        assert_eq!(
            compatibility.chatgpt_account_id.as_deref(),
            Some("acct-adapter")
        );
        assert_eq!(output.stop_reason, "native_replacement");
        assert_eq!(counters.attempts, 1);
    }

    #[tokio::test]
    async fn adapter_keeps_auth_failure_hard() {
        let app = Router::new().route(
            "/v1/responses/compact",
            post(|| async { StatusCode::UNAUTHORIZED }),
        );
        let client = adapter_client(spawn_compact_server(app).await);
        let input = adapter_input(
            vec![
                ConversationItem::system("canonical instructions"),
                ConversationItem::user("objective"),
            ],
            &client,
        );

        assert!(matches!(
            run_native_compaction(input, &client).await,
            NativeCompactionOutcome::HardFailure {
                source: NativeCompactionFailureSource::Sampling(SamplingError::Auth(_)),
                counters,
                estimated_input_tokens,
            } if counters.attempts == 1 && estimated_input_tokens > 0
        ));
    }
}
