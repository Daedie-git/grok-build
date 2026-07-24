//! Native Codex compaction strategy selection and preflight helpers.
//!
//! Included from [`super::compaction`] via `#[path = "compaction_native.rs"]`.

use crate::session::helpers::session_compact::is_context_length_error;
use xai_grok_sampling_types::{ConversationItem, SamplingError};

/// Compaction implementation selection. This is deliberately separate from
/// [`super::CompactionInputStage`]: native replacement history and local summary input
/// fitting are different implementations, not stages of one algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CompactionStrategy {
    /// Unary Codex `/responses/compact`; installs provider replacement items.
    NativeCodex,
    /// Existing ordinary `/responses` summarization and local history rebuild.
    LocalSummary,
}

impl CompactionStrategy {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::NativeCodex => "native_codex",
            Self::LocalSummary => "local_summary",
        }
    }

    pub(super) fn uses_local_summary_pipeline(self) -> bool {
        self == Self::LocalSummary
    }
}

/// Resolve the strategy from backend capability plus the explicit developer
/// override `GROK_CODEX_COMPACTION_STRATEGY=native|native_codex`.
///
/// Native compaction is deliberately opt-in. Missing, empty, local, and invalid
/// values use the hardened local-summary pipeline. Non-Codex backends always
/// stay local because the native endpoint and encrypted item contract are
/// Codex-specific.
pub(super) fn resolve_compaction_strategy(
    base_url: &str,
    override_value: Option<&str>,
) -> CompactionStrategy {
    if !xai_grok_sampling_types::capabilities_for_base_url(base_url).native_compact {
        return CompactionStrategy::LocalSummary;
    }
    match override_value
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
    }
}

/// Apply the command-level contract after backend/override selection.
/// `/compact <context>` remains local because the native endpoint has no field
/// for user-authored summary guidance; silently discarding it is unsafe.
pub(super) fn select_compaction_strategy(
    base_url: &str,
    override_value: Option<&str>,
    user_context_supplied: bool,
) -> CompactionStrategy {
    let strategy = resolve_compaction_strategy(base_url, override_value);
    if user_context_supplied && strategy == CompactionStrategy::NativeCodex {
        CompactionStrategy::LocalSummary
    } else {
        strategy
    }
}

/// Only endpoint capability failures may cross from native to local. Auth,
/// malformed bodies, schema errors, context overflow, rate limits, and server
/// failures remain native failures so fallback never hides correctness issues.
pub(super) fn native_endpoint_unavailable(error: &SamplingError) -> bool {
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

pub(super) fn native_context_overflow(error: &SamplingError) -> bool {
    match error {
        SamplingError::Api { message, .. } | SamplingError::StreamError { message, .. } => {
            is_context_length_error(message)
        }
        _ => false,
    }
}

pub(super) const NATIVE_TRUNCATED_OUTPUT: &str =
    "Output exceeded the available model context and was truncated";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NativePreflightResult {
    pub(super) rewritten_outputs: usize,
    pub(super) initial_tokens: i128,
    pub(super) final_tokens: i128,
}

/// Preserve call/result topology while reducing only a contiguous suffix of
/// tool outputs. This mirrors Codex's remote-compaction preflight: walk from
/// newest to oldest, stop at the first ineligible item, and never let a
/// saturating public estimate conceal overflow.
pub(super) fn trim_native_tool_outputs_to_fit_context_window(
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
            content: NATIVE_TRUNCATED_OUTPUT.into(),
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

#[cfg(test)]
mod compaction_strategy_tests {
    use super::{
        CompactionStrategy, NATIVE_TRUNCATED_OUTPUT, native_context_overflow,
        native_endpoint_unavailable, select_compaction_strategy,
        trim_native_tool_outputs_to_fit_context_window,
    };
    use reqwest::StatusCode;
    use xai_grok_sampling_types::ConversationItem;
    use xai_grok_sampling_types::{CODEX_BACKEND_BASE_URL, SamplingError};

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
                select_compaction_strategy(CODEX_BACKEND_BASE_URL, value, false),
                CompactionStrategy::LocalSummary
            );
        }
        for value in [Some("native"), Some(" native_codex ")] {
            assert_eq!(
                select_compaction_strategy(CODEX_BACKEND_BASE_URL, value, false),
                CompactionStrategy::NativeCodex
            );
        }
    }

    #[test]
    fn non_codex_and_manual_guidance_remain_local() {
        assert_eq!(
            select_compaction_strategy("https://api.openai.com/v1", Some("native"), false),
            CompactionStrategy::LocalSummary
        );
        assert_eq!(
            select_compaction_strategy(CODEX_BACKEND_BASE_URL, None, true),
            CompactionStrategy::LocalSummary
        );
    }

    #[test]
    fn native_is_not_a_prefit_or_two_pass_local_summary_variant() {
        assert!(!CompactionStrategy::NativeCodex.uses_local_summary_pipeline());
        assert!(CompactionStrategy::LocalSummary.uses_local_summary_pipeline());
    }

    #[test]
    fn fallback_is_limited_to_endpoint_capability_statuses() {
        for status in [
            StatusCode::NOT_FOUND,
            StatusCode::METHOD_NOT_ALLOWED,
            StatusCode::NOT_IMPLEMENTED,
        ] {
            assert!(native_endpoint_unavailable(&api_error(status)), "{status}");
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
            assert!(!native_endpoint_unavailable(&api_error(status)), "{status}");
        }
        let context_overflow = SamplingError::Api {
            status: StatusCode::BAD_REQUEST,
            message: "input exceeds the context window".into(),
            model_metadata: None,
            retry_after_secs: None,
            should_retry: Some(false),
        };
        assert!(
            !native_endpoint_unavailable(&context_overflow),
            "native context overflow must not switch to local summarization"
        );
        assert!(!native_endpoint_unavailable(&SamplingError::Auth(
            "expired".into()
        )));
        assert!(!native_endpoint_unavailable(
            &SamplingError::serialization_message("malformed compact response")
        ));
        assert!(native_context_overflow(&context_overflow));
        assert!(!native_context_overflow(&api_error(
            StatusCode::BAD_REQUEST
        )));
    }

    #[test]
    fn native_preflight_rewrites_multiple_trailing_outputs_and_preserves_ids() {
        let mut items = vec![
            ConversationItem::user("prompt"),
            ConversationItem::tool_result("call-a", &"a".repeat(4_000)),
            ConversationItem::tool_result("call-b", &"b".repeat(4_000)),
        ];
        let result = trim_native_tool_outputs_to_fit_context_window(&mut items, 200, 0);
        assert_eq!(result.rewritten_outputs, 2);
        assert!(result.final_tokens <= 200);
        for (item, expected_id) in items[1..].iter().zip(["call-a", "call-b"]) {
            let ConversationItem::ToolResult(output) = item else {
                panic!("tool-result topology must be retained")
            };
            assert_eq!(output.tool_call_id, expected_id);
            assert_eq!(&*output.content, NATIVE_TRUNCATED_OUTPUT);
        }
    }

    #[test]
    fn native_preflight_does_not_rewrite_when_fitting_or_cross_ineligible_suffix() {
        let fitting = vec![ConversationItem::tool_result("call-a", "small")];
        let mut untouched = fitting.clone();
        let result = trim_native_tool_outputs_to_fit_context_window(&mut untouched, 100, 0);
        assert_eq!(result.rewritten_outputs, 0);
        assert_eq!(
            xai_chat_state::estimate_conversation_tokens(&untouched),
            xai_chat_state::estimate_conversation_tokens(&fitting)
        );

        let mut blocked = vec![
            ConversationItem::tool_result("old-large", &"x".repeat(8_000)),
            ConversationItem::assistant("suffix blocks backward rewriting"),
        ];
        let result = trim_native_tool_outputs_to_fit_context_window(&mut blocked, 10, 0);
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
        let result = trim_native_tool_outputs_to_fit_context_window(&mut items, 1, 100);
        assert_eq!(result.rewritten_outputs, 1);
        assert!(result.final_tokens > 1);
    }
}
