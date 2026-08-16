//! Compaction methods for `SessionActor`.
//!
//! This module contains all compaction-related methods: manual `/compact`,
//! auto-compact threshold checks, inline auto-compact with auto-continue,
//! error-recovery compaction, preflight overflow detection, and checkpoint
//! persistence. These methods form a second `impl SessionActor` block that
//! lives alongside the primary one in `acp_session.rs`.
#[path = "compaction/native/mod.rs"]
mod native;
use native::*;

use super::SessionActor;
use super::is_project_instructions;
use crate::remote::DEFAULT_CONTEXT_WINDOW;
use crate::session::compaction_config::{
    AUTO_COMPACT_RETRY_AFTER_RESET, AUTO_COMPACT_RETRY_READY, AsyncCompactionCache,
    BOUNDED_COMPACT_FAILED, BOUNDED_COMPACT_FINALIZING, BOUNDED_COMPACT_IN_FLIGHT,
    BOUNDED_COMPACT_NONE, SUPPRESS_AUTH, SUPPRESS_NONE, SUPPRESS_STICKY, SUPPRESS_TURN,
    SUPPRESS_UNTIL_SUCCESS,
};
use crate::session::helpers::CompactionStateContext;
use crate::session::helpers::compaction_context::CompactionInputs;
use crate::session::helpers::compaction_context::to_system_reminder;
use crate::session::helpers::replay::CompactionCommitKind;
use crate::session::helpers::session_compact::{
    CompactOutput, CompactionOutcome, build_compaction_prompt, build_two_pass_compaction_prompt,
    generate_session_compact, is_context_length_error,
};
use crate::session::persistence::PersistenceMsg;
use crate::session::two_pass::{
    TWO_PASS_DEFAULT_SPLIT_FRACTION, build_two_pass_pass1_history, build_two_pass_pass2_history,
    note_for_two_pass_pass2, split_conversation_for_two_pass,
};
use agent_client_protocol as acp;
use std::sync::Arc;
use xai_chat_state::compaction_utils::{
    CompactedHistoryInput, CompactionAttempt, build_compacted_history, is_degenerate_summary,
    prepare_conversation_for_verbatim_summarization, sanitize_compacted_history,
    validate_compacted_history,
};
use xai_grok_sampling_types::{ApiBackend, ConversationItem};

/// The validated replacement produced by exactly one compaction implementation.
/// Keeping native and local products discriminated prevents native history from
/// accidentally entering local summary reconstruction or sanitization.
enum CompactionProduct {
    NativeCodex {
        replacement_history: Vec<ConversationItem>,
        output: CompactOutput,
    },
    LocalSummary {
        summary: String,
        output: CompactOutput,
    },
}

enum CompactionReplacement {
    NativeCodex(Vec<ConversationItem>),
    LocalSummary(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutoCompactFailureDisposition {
    Cooldown,
    RetryAfterReset,
}

impl CompactionProduct {
    fn local_summary(&self) -> Option<&str> {
        match self {
            Self::NativeCodex { .. } => None,
            Self::LocalSummary { summary, .. } => Some(summary),
        }
    }

    fn into_parts(self) -> (CompactionReplacement, CompactOutput) {
        match self {
            Self::NativeCodex {
                replacement_history,
                output,
            } => (
                CompactionReplacement::NativeCodex(replacement_history),
                output,
            ),
            Self::LocalSummary { summary, output } => {
                (CompactionReplacement::LocalSummary(summary), output)
            }
        }
    }
}

/// Default percentage points below the auto-compact threshold at which prefire
/// (background pass-1) starts, giving pass-1 runway to finish before the limit.
/// Override with `GROK_PREFIRE_LEAD_PERCENT`.
const DEFAULT_PREFIRE_LEAD_PERCENT: u64 = 10;
fn prefire_lead_percent() -> u64 {
    std::env::var("GROK_PREFIRE_LEAD_PERCENT")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_PREFIRE_LEAD_PERCENT)
}

/// Owned snapshot of the process-level Codex compaction strategy override.
#[derive(Debug, PartialEq, Eq)]
enum ConfiguredCodexCompactionStrategy {
    Unset,
    Value(String),
    InvalidEncoding,
}

impl ConfiguredCodexCompactionStrategy {
    fn from_env_result(result: Result<String, std::env::VarError>) -> Self {
        match result {
            Ok(value) => Self::Value(value),
            Err(std::env::VarError::NotPresent) => Self::Unset,
            Err(std::env::VarError::NotUnicode(_)) => Self::InvalidEncoding,
        }
    }

    fn as_override(&self) -> CompactionStrategyOverride<'_> {
        match self {
            Self::Unset => CompactionStrategyOverride::Unset,
            Self::Value(value) => CompactionStrategyOverride::Value(value),
            Self::InvalidEncoding => CompactionStrategyOverride::InvalidEncoding,
        }
    }
}

/// Snapshot the developer strategy override once at compaction invocation.
///
/// Keeping the global read outside `run_compact_inner` makes its policy input
/// explicit and lets tests cover default behavior without mutating process state.
fn configured_codex_compaction_strategy() -> ConfiguredCodexCompactionStrategy {
    ConfiguredCodexCompactionStrategy::from_env_result(std::env::var(
        "GROK_CODEX_COMPACTION_STRATEGY",
    ))
}
/// Cheap fingerprint of a conversation prefix for prefire NOTE₁ validity. A
/// mismatch means the prefix changed (edit / rewind / branch) since pass-1, so
/// the cached NOTE₁ no longer summarizes the current prefix and must be dropped.
fn fingerprint_prefix(items: &[ConversationItem]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    items.len().hash(&mut h);
    for it in items {
        let tag: u8 = match it {
            ConversationItem::System(_) => 0,
            ConversationItem::User(_) => 1,
            ConversationItem::Assistant(_) => 2,
            ConversationItem::ToolResult(_) => 3,
            ConversationItem::BackendToolCall(_) => 4,
            ConversationItem::Reasoning(_) => 5,
            ConversationItem::Provider(provider) => {
                if provider.is_native_compaction_metadata() {
                    6
                } else if provider.is_encrypted_compaction() {
                    7
                } else {
                    8
                }
            }
        };
        tag.hash(&mut h);
        if let ConversationItem::Provider(provider) = it {
            if let Some(metadata) = provider.as_native_compaction_metadata() {
                metadata.schema_version.hash(&mut h);
                metadata.backend_family.hash(&mut h);
                metadata.model.hash(&mut h);
                metadata.chatgpt_account_id.hash(&mut h);
            }
            if let Some(metadata) = provider.as_response_output_metadata() {
                serde_json::to_string(metadata)
                    .unwrap_or_default()
                    .hash(&mut h);
            }
        }
        it.text_content().hash(&mut h);
    }
    h.finish()
}
/// Outcome of a background prefire pass-1 run, recorded on the
/// `session.prefire_pass1` span as `compaction_prefire_outcome`.
/// [`PrefireOutcome::as_str`] values are stable telemetry keys
/// (telemetry/dashboards key off them) — don't rename the strings.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PrefireOutcome {
    Cached,
    Disabled,
    DebugFailPass1,
    TooSmall,
    EmptySplit,
    SampleFailed,
    EmptyNote1,
}
impl PrefireOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cached => "cached",
            Self::Disabled => "disabled",
            Self::DebugFailPass1 => "debug_fail_pass1",
            Self::TooSmall => "too_small",
            Self::EmptySplit => "empty_split",
            Self::SampleFailed => "sample_failed",
            Self::EmptyNote1 => "empty_note1",
        }
    }
}
/// Telemetry from one prefire pass-1 run; recorded onto the
/// `session.prefire_pass1` span by [`SessionActor::run_prefire_pass1`].
/// `None` fields = the run exited before that stage.
struct PrefirePass1Run {
    outcome: PrefireOutcome,
    prefix_len: Option<usize>,
    prefix_est_tokens: Option<u64>,
    pass1_latency_ms: Option<u64>,
    note1_chars: Option<usize>,
}
impl From<PrefireOutcome> for PrefirePass1Run {
    /// A run that exited before splitting/sampling — outcome only.
    fn from(outcome: PrefireOutcome) -> Self {
        Self {
            outcome,
            prefix_len: None,
            prefix_est_tokens: None,
            pass1_latency_ms: None,
            note1_chars: None,
        }
    }
}
#[cfg(test)]
#[path = "compaction_two_pass_prefire_helper_tests.rs"]
mod two_pass_prefire_helper_tests;
impl SessionActor {
    /// Two-pass active for this session: flag resolved on at build AND not an
    /// agent that keeps its single short self-summary.
    pub(crate) fn two_pass_active(&self) -> bool {
        let agent = self.agent.borrow();
        agent.compaction_policy().two_pass_enabled
    }
    /// Run one summarization sample over a fully-built two-pass history (the
    /// prompt is already embedded, so this bypasses the single-pass sampler and
    /// calls `generate_session_compact` directly). Returns `None` on any error
    /// so callers fall back to single-pass.
    ///
    /// Agent `RefCell` borrows are only taken for synchronous snapshots (never
    /// held across `.await`). Prefire is `spawn_local` on the same LocalSet as
    /// the turn loop; a long-lived borrow would race with turn/compact/cancel
    /// and panic on double-borrow.
    async fn two_pass_sample(&self, history: Vec<ConversationItem>) -> Option<CompactOutput> {
        let sampling_config = self.reconstruct_full_config().await;
        let client = match self.prepare_chat_completion(false).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "two_pass: failed to prepare sampling client");
                return None;
            }
        };
        let tool_defs = self.prepare_tool_definitions().await;
        let tools = self.turn_base_tool_specs(&tool_defs);
        let compaction_tool_tokens = xai_chat_state::estimate_tool_specs_tokens(&tools);
        let wall_clock_budget_secs = self
            .agent
            .borrow()
            .compaction_policy()
            .wall_clock_budget_secs;
        let hosted_tools = self.hosted_tools_for_turn();
        let (cancel, _cancel_scope) = self.compaction.cancel.enter();
        match generate_session_compact(
            history,
            compaction_tool_tokens,
            tools,
            hosted_tools,
            client,
            self.session_info.id.clone(),
            &sampling_config,
            self.inference_idle_timeout,
            wall_clock_budget_secs,
            self.compaction.tool_choice,
            &cancel,
        )
        .await
        {
            Ok(out) => Some(out),
            Err(e) => {
                tracing::warn!(error = ?e, "two_pass: summarization sample failed");
                None
            }
        }
    }
    /// Per-turn prefire decision: usage has reached `threshold - lead` (so there
    /// is still runway before the hard auto-compact line at `threshold`).
    pub(crate) async fn should_prefire_two_pass(&self) -> bool {
        let sampling_cfg = self.chat_state_handle.get_sampling_config().await;
        let Some(cw) = sampling_cfg.as_ref().map(|c| c.context_window.get()) else {
            return false;
        };
        let estimated_total = self.chat_state_handle.get_estimated_total_tokens().await;
        let threshold = self.compaction.threshold_percent.get() as u64;
        let start_pct = threshold.saturating_sub(prefire_lead_percent());
        xai_token_estimation::exceeds_threshold(estimated_total, cw, start_pct as u8)
    }
    /// Background pass-1: summarize the ~95% prefix → NOTE₁ and cache it for a
    /// later pass-2 apply. Always releases the in-flight guard. Spawned via
    /// `spawn_local` from the turn loop; reads a conversation snapshot and does
    /// not mutate session state. The span makes speculative pass-1 spend
    /// measurable (hit rate, wasted input tokens) ahead of the fleet-wide ramp.
    #[tracing::instrument(
        name = "session.prefire_pass1",
        skip_all,
        fields(
            session_id = %self.session_info.id.0,
            compaction_prefire_outcome = tracing::field::Empty,
            compaction_pass1_latency_ms = tracing::field::Empty,
            compaction_prefire_prefix_len = tracing::field::Empty,
            compaction_prefire_prefix_est_tokens = tracing::field::Empty,
            compaction_prefire_note1_chars = tracing::field::Empty,
        )
    )]
    pub(crate) async fn run_prefire_pass1(self: &Arc<Self>) {
        struct InFlightGuard<'a>(&'a crate::session::compaction_config::PrefireState);
        impl Drop for InFlightGuard<'_> {
            fn drop(&mut self) {
                self.0.finish();
            }
        }
        let _guard = InFlightGuard(&self.compaction.prefire);
        let run = self.run_prefire_pass1_inner().await;
        let span = tracing::Span::current();
        span.record("compaction_prefire_outcome", run.outcome.as_str());
        if let Some(v) = run.prefix_len {
            span.record("compaction_prefire_prefix_len", v as i64);
        }
        if let Some(v) = run.prefix_est_tokens {
            span.record("compaction_prefire_prefix_est_tokens", v as i64);
        }
        if let Some(v) = run.pass1_latency_ms {
            span.record("compaction_pass1_latency_ms", v as i64);
        }
        if let Some(v) = run.note1_chars {
            span.record("compaction_prefire_note1_chars", v as i64);
        }
    }
    async fn run_prefire_pass1_inner(self: &Arc<Self>) -> PrefirePass1Run {
        if !self.two_pass_active() {
            return PrefireOutcome::Disabled.into();
        }
        if std::env::var("GROK_DEBUG_TWO_PASS_FAIL_PASS1")
            .is_ok_and(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
        {
            tracing::info!(
                target: "two_pass",
                "two_pass: DEBUG GROK_DEBUG_TWO_PASS_FAIL_PASS1 — prefire pass1 produces no cache"
            );
            return PrefireOutcome::DebugFailPass1.into();
        }
        let conversation = self.chat_state_handle.get_conversation().await;
        if conversation.len() < 4 {
            return PrefireOutcome::TooSmall.into();
        }
        let split = split_conversation_for_two_pass(&conversation, TWO_PASS_DEFAULT_SPLIT_FRACTION);
        if split.prefix.is_empty() || split.tail.is_empty() {
            return PrefireOutcome::EmptySplit.into();
        }
        let sampling_cfg = self.chat_state_handle.get_sampling_config().await;
        let strips = sampling_cfg
            .as_ref()
            .map(|c| c.api_backend == ApiBackend::Messages)
            .unwrap_or(false);
        let model_slug = sampling_cfg
            .as_ref()
            .map(|c| c.model.to_string())
            .unwrap_or_default();
        let prefix_prepared =
            prepare_conversation_for_verbatim_summarization(split.prefix.to_vec(), strips);
        let prefix_est_tokens = prefix_prepared
            .iter()
            .map(xai_chat_state::estimate_item_tokens)
            .sum::<u64>();
        let prompt = build_two_pass_compaction_prompt(None);
        let pass1_history = build_two_pass_pass1_history(&prefix_prepared, &prompt);
        let started = std::time::Instant::now();
        let out = self.two_pass_sample(pass1_history).await;
        let pass1_latency_ms = started.elapsed().as_millis() as u64;
        let attempted = |outcome: PrefireOutcome, note1_chars: Option<usize>| PrefirePass1Run {
            outcome,
            prefix_len: Some(split.split_idx),
            prefix_est_tokens: Some(prefix_est_tokens),
            pass1_latency_ms: Some(pass1_latency_ms),
            note1_chars,
        };
        let Some(out) = out else {
            return attempted(PrefireOutcome::SampleFailed, None);
        };
        let note1 = note_for_two_pass_pass2(&out.content);
        if note1.trim().is_empty() {
            return attempted(PrefireOutcome::EmptyNote1, None);
        }
        let note1_chars = note1.chars().count();
        let cache = AsyncCompactionCache {
            note1,
            prefix_len: split.split_idx,
            fingerprint: fingerprint_prefix(&conversation[..split.split_idx]),
            model_slug,
            pass1_latency_ms,
        };
        tracing::info!(
            target: "two_pass",
            prefix_len = cache.prefix_len,
            pass1_latency_ms = cache.pass1_latency_ms,
            "two_pass: prefire pass1 cached NOTE1"
        );
        self.compaction.prefire.store(cache);
        attempted(PrefireOutcome::Cached, Some(note1_chars))
    }
    /// Pass-2 apply: if a valid cached NOTE₁ exists for the current conversation,
    /// summarize (NOTE₁ + recent tail + special prompt) → final summary and
    /// return its `CompactOutput`. `None` → caller runs the single-pass path.
    ///
    /// **telemetry / `session.compact_inner` latency:** the returned `CompactOutput`
    /// stream timings are what land on `compaction_ttft_ms` /
    /// `compaction_stream_ms`. Those reflect **user-visible sync wait only**:
    /// - background pass-1 that already finished before compact is *not*
    ///   included (prefire hid that cost);
    /// - if pass-1 is still in flight we **do** add that await into
    ///   `ttft_ms` (time until first token of the final summary), because the
    ///   user is blocked on it;
    /// - `stream_ms` / `delta_count` / `itl_max_ms` are always pass-2 only
    ///   (the only sample that streams the successor-visible summary).
    async fn try_two_pass_pass2_apply(
        &self,
        user_context: Option<&str>,
        strips_reasoning: bool,
    ) -> Option<CompactOutput> {
        if !self.two_pass_active() {
            return None;
        }
        let mut prefire_waited_ms = 0u64;
        if let Some(handle) = self.compaction.prefire.take_handle() {
            let was_in_flight = self.compaction.prefire.is_in_flight();
            let waited = std::time::Instant::now();
            let _ = handle.await;
            if was_in_flight {
                prefire_waited_ms = waited.elapsed().as_millis() as u64;
                tracing::Span::current()
                    .record("compaction_prefire_waited_ms", prefire_waited_ms as i64);
                tracing::info!(
                    target: "two_pass",
                    wait_ms = prefire_waited_ms,
                    "two_pass: waited for in-flight prefire pass1 before pass2"
                );
            }
        }
        let cache = self.compaction.prefire.take()?;
        let live = self.chat_state_handle.get_conversation().await;
        let model_slug = self
            .chat_state_handle
            .get_sampling_config()
            .await
            .map(|c| c.model.to_string())
            .unwrap_or_default();
        if cache.prefix_len == 0
            || cache.prefix_len > live.len()
            || cache.model_slug != model_slug
            || fingerprint_prefix(&live[..cache.prefix_len]) != cache.fingerprint
        {
            tracing::Span::current().record("compaction_prefire_stale", true);
            tracing::info!(
                target: "two_pass",
                "two_pass: cached NOTE1 stale or model changed; falling back to single-pass"
            );
            return None;
        }
        let prefix = &live[..cache.prefix_len];
        let tail = &live[cache.prefix_len..];
        let prepared_tail =
            prepare_conversation_for_verbatim_summarization(tail.to_vec(), strips_reasoning);
        let prompt = build_two_pass_compaction_prompt(user_context);
        let pass2_history =
            build_two_pass_pass2_history(prefix, &prepared_tail, &cache.note1, &prompt);
        let started = std::time::Instant::now();
        let mut out = self.two_pass_sample(pass2_history).await?;
        if is_degenerate_summary(&out.content) {
            tracing::Span::current().record("compaction_prefire_stale", true);
            tracing::info!(
                target: "two_pass",
                "two_pass: pass2 summary empty/degenerate; falling back to single-pass"
            );
            return None;
        }
        let pass2_latency_ms = started.elapsed().as_millis() as u64;
        if prefire_waited_ms > 0 {
            out.ttft_ms = Some(out.ttft_ms.unwrap_or(0).saturating_add(prefire_waited_ms));
        }
        let span = tracing::Span::current();
        span.record("compaction_two_pass_used", true);
        span.record("compaction_prefire_hit", true);
        span.record("compaction_pass2_latency_ms", pass2_latency_ms as i64);
        tracing::info!(
            target: "two_pass",
            prefix_len = cache.prefix_len,
            tail_len = tail.len(),
            prefire_waited_ms,
            pass2_latency_ms,
            pass1_bg_latency_ms = cache.pass1_latency_ms,
            "two_pass: pass2 applied cached NOTE1 (prefire hit)"
        );
        Some(out)
    }
}
/// Trigger info for auto-compact decisions.
/// Why automatic compaction fired. Post-install headroom validation is
/// trigger-aware: a Codex safety compact must only prove Codex headroom, while
/// a soft-threshold compact must also clear Codex's provider limit when present.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AutoCompactTriggerKind {
    SoftThreshold,
    CodexSafety,
    PreflightOverflow,
    Forced,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AutoCompactTriggerInfo {
    pub tokens_used: u64,
    pub context_window: u64,
    pub percentage: u8,
    pub kind: AutoCompactTriggerKind,
}

const BOUNDED_SUBAGENT_FINALIZE_REMINDER: &str = "This bounded subagent task has used its one allowed automatic compaction. Stop exploring now and return the best complete final result using only evidence already collected. Do not call or request exploratory tools. If a required final-output tool is available, call only that tool once to submit the result. Preserve concrete file/line references and verified evidence; omit or explicitly qualify anything you could not verify.";

/// Token reserve for the finalize reminder (including `<system-reminder>` wrap)
/// so post-install headroom accounts for the message we are about to push.
fn bounded_finalize_reminder_token_reserve() -> u64 {
    let wrapped =
        format!("<system-reminder>\n{BOUNDED_SUBAGENT_FINALIZE_REMINDER}\n</system-reminder>");
    xai_token_estimation::estimate_tokens(&wrapped)
}

/// Output runway reserved when fitting a near-limit *local-summary* request.
/// Native Codex compaction bypasses this policy: `/responses/compact` owns its
/// replacement-history budget and receives the full prepared transcript.
const SUMMARY_BUDGET_RESERVE_TOKENS: u64 = 32_768;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompactionInputStage {
    Verbatim,
    VerbatimFitted,
    Lossy,
}

impl CompactionInputStage {
    fn as_str(self) -> &'static str {
        match self {
            Self::Verbatim => "verbatim",
            Self::VerbatimFitted => "verbatim_fitted",
            Self::Lossy => "lossy",
        }
    }

    fn after_context_overflow(self) -> Option<Self> {
        match self {
            Self::Verbatim => Some(Self::VerbatimFitted),
            Self::VerbatimFitted => Some(Self::Lossy),
            Self::Lossy => None,
        }
    }
}

/// Maximum prepared-history tokens that leave room for tool definitions, the
/// summary instruction, and the summary output reserve.
fn fitted_compaction_history_budget(
    context_window: u64,
    tool_tokens: u64,
    summary_prompt_tokens: u64,
) -> u64 {
    context_window
        .saturating_sub(SUMMARY_BUDGET_RESERVE_TOKENS)
        .saturating_sub(tool_tokens)
        .saturating_sub(summary_prompt_tokens)
}

fn codex_auto_compaction_needs_prefit(
    prepared_history_tokens: u64,
    context_window: u64,
    tool_tokens: u64,
    summary_prompt_tokens: u64,
) -> bool {
    prepared_history_tokens
        > fitted_compaction_history_budget(context_window, tool_tokens, summary_prompt_tokens)
}

fn initial_compaction_input_stage(
    verbatim_input_enabled: bool,
    auto_trigger: bool,
    capabilities: xai_grok_sampling_types::ProviderCapabilities,
    prepared_history_tokens: u64,
    context_window: u64,
    tool_tokens: u64,
    summary_prompt_tokens: u64,
) -> CompactionInputStage {
    if !verbatim_input_enabled {
        return CompactionInputStage::Lossy;
    }
    if auto_trigger
        && capabilities.enforces_auto_compact_safety()
        && codex_auto_compaction_needs_prefit(
            prepared_history_tokens,
            context_window,
            tool_tokens,
            summary_prompt_tokens,
        )
    {
        CompactionInputStage::VerbatimFitted
    } else {
        CompactionInputStage::Verbatim
    }
}

/// Cached two-pass pass 2 is built from its own unfitted prefix/tail history.
/// Until it can share the same fitting budget, bypass it when the safety
/// policy selected a fitted first request.
fn two_pass_allowed_for_initial_stage(two_pass_enabled: bool, stage: CompactionInputStage) -> bool {
    two_pass_enabled && stage != CompactionInputStage::VerbatimFitted
}

#[cfg(test)]
mod codex_prefit_tests {
    use super::*;

    #[test]
    fn prefit_threshold_includes_tools_prompt_and_output_reserve() {
        let context_window = 100_000;
        let tool_tokens = 4_000;
        let prompt_tokens = 2_000;
        let threshold =
            fitted_compaction_history_budget(context_window, tool_tokens, prompt_tokens);
        assert_eq!(threshold, 61_232);
        assert!(!codex_auto_compaction_needs_prefit(
            threshold,
            context_window,
            tool_tokens,
            prompt_tokens,
        ));
        assert!(codex_auto_compaction_needs_prefit(
            threshold + 1,
            context_window,
            tool_tokens,
            prompt_tokens,
        ));
    }

    #[test]
    fn initial_stage_selection_covers_trigger_backend_and_verbatim_policy() {
        let context_window = 100_000;
        let tool_tokens = 4_000;
        let prompt_tokens = 2_000;
        let over_budget =
            fitted_compaction_history_budget(context_window, tool_tokens, prompt_tokens) + 1;
        let stage = |verbatim, auto, provider_id| {
            let capabilities = xai_grok_sampling_types::resolve_provider(
                Some(provider_id),
                xai_grok_sampling_types::ApiBackend::Responses,
                "http://127.0.0.1:3210/v1",
            )
            .capabilities();
            initial_compaction_input_stage(
                verbatim,
                auto,
                capabilities,
                over_budget,
                context_window,
                tool_tokens,
                prompt_tokens,
            )
        };

        assert_eq!(
            stage(true, true, xai_grok_sampling_types::ProviderId::Codex),
            CompactionInputStage::VerbatimFitted,
            "automatic over-budget Codex must prefit"
        );
        assert_eq!(
            stage(true, false, xai_grok_sampling_types::ProviderId::Codex),
            CompactionInputStage::Verbatim,
            "manual Codex keeps the normal ladder entry"
        );
        assert_eq!(
            stage(
                true,
                true,
                xai_grok_sampling_types::ProviderId::OpenAiCompatible,
            ),
            CompactionInputStage::Verbatim,
            "automatic non-Codex keeps the normal ladder entry"
        );
        assert_eq!(
            stage(false, true, xai_grok_sampling_types::ProviderId::Codex),
            CompactionInputStage::Lossy,
            "disabled verbatim input starts lossy"
        );
    }

    #[test]
    fn two_pass_enabled_cannot_bypass_over_budget_automatic_codex_prefit() {
        let history = vec![
            ConversationItem::system("system"),
            ConversationItem::user("old ".repeat(20_000)),
            ConversationItem::assistant("old answer ".repeat(20_000)),
            ConversationItem::user("most recent useful request"),
            ConversationItem::assistant("most recent useful work"),
        ];
        let history_tokens = xai_chat_state::estimate_conversation_tokens(&history);
        let context_window = 50_000;
        let tool_tokens = 2_000;
        let prompt_tokens = 1_000;
        let budget = fitted_compaction_history_budget(context_window, tool_tokens, prompt_tokens);
        let capabilities = xai_grok_sampling_types::resolve_provider(
            Some(xai_grok_sampling_types::ProviderId::Codex),
            xai_grok_sampling_types::ApiBackend::Responses,
            "http://127.0.0.1:3210/v1",
        )
        .capabilities();
        let stage = initial_compaction_input_stage(
            true,
            true,
            capabilities,
            history_tokens,
            context_window,
            tool_tokens,
            prompt_tokens,
        );
        assert_eq!(stage, CompactionInputStage::VerbatimFitted);
        assert!(
            !two_pass_allowed_for_initial_stage(true, stage),
            "an enabled two-pass cache must not send its unfitted pass-2 request"
        );

        let outbound =
            xai_chat_state::compaction_utils::fit_conversation_to_budget(history.clone(), budget);
        assert!(
            outbound.len() < history.len(),
            "the selected single-pass first request must already be fitted"
        );
        let fitted_tokens = xai_chat_state::estimate_conversation_tokens(&outbound);
        assert!(
            fitted_tokens <= budget,
            "fit is achievable for this fixture: {fitted_tokens} > {budget}"
        );
        assert!(
            outbound
                .iter()
                .any(|item| item.text_content() == "most recent useful work")
        );
    }

    #[test]
    fn impossible_tiny_fit_reports_structural_minimum_above_budget() {
        let history = vec![
            ConversationItem::system("required system instructions"),
            ConversationItem::assistant("required newest turn"),
        ];
        let budget = 0;
        let outbound =
            xai_chat_state::compaction_utils::fit_conversation_to_budget(history, budget);
        let retained_tokens = xai_chat_state::estimate_conversation_tokens(&outbound);
        assert!(
            retained_tokens > budget,
            "a zero budget cannot honestly be reported as fitting retained structure"
        );
    }

    #[test]
    fn fitted_stage_keeps_overflow_fallback_to_lossy() {
        assert_eq!(
            CompactionInputStage::Verbatim.after_context_overflow(),
            Some(CompactionInputStage::VerbatimFitted)
        );
        assert_eq!(
            CompactionInputStage::VerbatimFitted.after_context_overflow(),
            Some(CompactionInputStage::Lossy)
        );
        assert_eq!(CompactionInputStage::Lossy.after_context_overflow(), None);
    }
}

/// Why auto-compaction was suppressed after a deterministic failure.
/// [`SuppressReason::as_str`] is a stable telemetry value (BQ/OTLP/dashboards key
/// off it) — don't rename the strings.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SuppressReason {
    CreditBlock,
    Size,
    Auth,
    Schema,
    Other,
}

/// Delay after a transient automatic-compaction lifecycle exhausts its own
/// retry budget. New turns may still be processed, but they cannot immediately
/// launch the same expensive network lifecycle again.
const AUTO_COMPACT_FAILURE_COOLDOWN_MS: u64 = 60_000;

impl SuppressReason {
    fn as_str(self) -> &'static str {
        match self {
            SuppressReason::CreditBlock => "credit_block",
            SuppressReason::Size => "size",
            SuppressReason::Auth => "auth",
            SuppressReason::Schema => "schema",
            SuppressReason::Other => "other",
        }
    }
    /// Suppression scope for this reason:
    /// - `size | schema` → [`SUPPRESS_STICKY`]: cleared only on a context-budget change.
    /// - `credit_block` → [`SUPPRESS_UNTIL_SUCCESS`]: wait for a model `200`.
    /// - `auth` → [`SUPPRESS_AUTH`]: clear on login/token refresh (not 200 — over-window deadlock).
    /// - `other` → [`SUPPRESS_TURN`]: optimistic per-turn retry.
    fn suppress_state(self) -> u8 {
        match self {
            SuppressReason::Size | SuppressReason::Schema => SUPPRESS_STICKY,
            SuppressReason::CreditBlock => SUPPRESS_UNTIL_SUCCESS,
            SuppressReason::Auth => SUPPRESS_AUTH,
            SuppressReason::Other => SUPPRESS_TURN,
        }
    }
}
/// Splice the preserved prefix (`conversation[0..prefix_len]`) onto the compacted
/// suffix, dropping the suffix's leading System and — if the prefix already has an
/// AGENTS.md item — its re-injected AGENTS.md too (else the model sees it twice).
/// Returns `Err(compacted_history)` unchanged when `prefix_len` is 0 or out of range.
fn preserve_inherited_prefix(
    conversation: &[ConversationItem],
    compacted_history: Vec<ConversationItem>,
    prefix_len: usize,
) -> Result<Vec<ConversationItem>, Vec<ConversationItem>> {
    if prefix_len == 0 || prefix_len > conversation.len() {
        return Err(compacted_history);
    }
    let inherited = &conversation[..prefix_len];
    let drop_reinjected_agents_md = inherited.iter().any(is_project_instructions);
    let mut preserved = inherited.to_vec();
    let child_items = compacted_history
        .into_iter()
        .skip_while(|i| matches!(i, ConversationItem::System(_)))
        .filter(|i| !(drop_reinjected_agents_md && is_project_instructions(i)));
    preserved.extend(child_items);
    Ok(preserved)
}
/// Project the token count a re-pinned (preserved) history would reseed to, so the
/// release decision compares against the same threshold the auto-compact trigger
/// applies next turn. This only APPROXIMATES the compaction reseed
/// (`xai-chat-state` `replace_conversation`, the authority): it matches the reseed's
/// round-and-cap but divides by the current conversation estimate, not the reseed's
/// frozen `estimate_at_last_response`. The conversation only grows, so the current
/// estimate is >= that frozen value; this therefore under-estimates the reseed (a
/// lower bound) and can lean toward preserve. That never re-loops: the post-replace
/// `exceeds_threshold` check on the real reseeded total still sets sticky Size
/// suppression if a preserve leaves the fork over budget.
fn project_preserved_reseed_tokens(
    preserved_estimate: u64,
    tokens_before: u64,
    full_conv_estimate: u64,
) -> u64 {
    let ratio = tokens_before as f64 / full_conv_estimate.max(1) as f64;
    ((preserved_estimate as f64 * ratio).round() as u64).min(tokens_before)
}
impl SessionActor {
    /// Only explicitly bounded child tasks must converge within one automatic
    /// compaction cycle. Nesting depth identifies a child but does not define
    /// its context-recovery lifecycle.
    pub(crate) fn is_bounded_subagent_task(&self) -> bool {
        self.tool_context.subagent_depth > 0
            && self.tool_context.subagent_compaction_policy
                == xai_grok_tools::implementations::grok_build::task::types::SubagentCompactionPolicy::FinalizeAfterOneCompaction
    }

    /// Reset the bounded-task circuit breaker at the start of a genuinely new prompt turn.
    pub(crate) fn reset_bounded_auto_compaction_for_turn(&self) {
        self.compaction
            .bounded_auto_compact_state
            .store(BOUNDED_COMPACT_NONE, std::sync::atomic::Ordering::Relaxed);
    }

    /// After the first successful automatic compaction, the next model call is
    /// synthesis-only: no further exploration tools are exposed.
    pub(crate) fn bounded_subagent_must_finalize(&self) -> bool {
        self.is_bounded_subagent_task()
            && self
                .compaction
                .bounded_auto_compact_state
                .load(std::sync::atomic::Ordering::Relaxed)
                == BOUNDED_COMPACT_FINALIZING
    }

    fn claim_bounded_auto_compaction(&self) -> Result<bool, acp::Error> {
        if !self.is_bounded_subagent_task() {
            return Ok(false);
        }
        self.compaction
            .bounded_auto_compact_state
            .compare_exchange(
                BOUNDED_COMPACT_NONE,
                BOUNDED_COMPACT_IN_FLIGHT,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            )
            .map(|_| true)
            .map_err(|state| {
                let prior = match state {
                    BOUNDED_COMPACT_IN_FLIGHT => "an automatic compaction is already in flight",
                    BOUNDED_COMPACT_FINALIZING => {
                        "one automatic compaction already succeeded and finalization did not converge"
                    }
                    BOUNDED_COMPACT_FAILED => "the first automatic compaction failed",
                    _ => "the automatic compaction state is invalid",
                };
                acp::Error::internal_error().data(format!(
                    "bounded subagent compaction cycle limit reached: {prior}; stopping instead of starting another compaction"
                ))
            })
    }

    fn finish_bounded_auto_compaction(&self, success: bool) {
        if self.is_bounded_subagent_task() {
            self.compaction.bounded_auto_compact_state.store(
                if success {
                    BOUNDED_COMPACT_FINALIZING
                } else {
                    BOUNDED_COMPACT_FAILED
                },
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    }

    /// Path to the raw `updates.jsonl` transcript if it exists, else `None`.
    /// `pub(crate)` so the `Transcript`-mode dispatch in `compaction_segments`
    /// and transcript-location pointers can both reuse it.
    ///
    /// The `path.exists()` guard keeps the pointer safe when a session (e.g. a
    /// nested sub-agent) never wrote one -- the hint is simply omitted rather
    /// than dangling.
    pub(crate) fn get_transcript_path(&self) -> Option<String> {
        let path =
            crate::session::persistence::session_dir(&self.session_info).join("updates.jsonl");
        if path.exists() {
            Some(path.to_string_lossy().into_owned())
        } else {
            None
        }
    }
    /// Increment the compaction counter and launch a pre-compaction memory flush.
    ///
    /// The counter is incremented before the flush check so the once-per-cycle
    /// guard does not suppress the first eligible flush.
    async fn maybe_pre_compaction_flush(
        self: &Arc<Self>,
        total_tokens: u64,
        context_window: u64,
        trigger: &'static str,
    ) {
        let compaction_count = self
            .compaction
            .count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        if !self.agent.borrow().compaction_policy().memory_flush_enabled {
            return;
        }
        let last_flush = self
            .memory
            .last_flush_compaction
            .load(std::sync::atomic::Ordering::Relaxed);
        if crate::session::helpers::memory_flush::should_flush(
            total_tokens,
            context_window,
            self.compaction.threshold_percent.get(),
            &self.memory.flush_config,
            last_flush,
            compaction_count,
        ) {
            let snapshot = self.snapshot_memory_flush_state().await;
            tokio::task::spawn_local({
                let session = self.clone();
                async move {
                    if session.run_memory_flush(trigger, Some(snapshot)).await {
                        session
                            .memory
                            .last_flush_compaction
                            .store(compaction_count, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            });
        }
    }
    /// Tag the current `session.compact` span with `mode` (and `detail`, for
    /// `segments`) — the A/B variant key for grouping outcomes in telemetry.
    fn record_compaction_variant(&self) {
        let mode = self.compaction.compaction_mode;
        let span = tracing::Span::current();
        span.record("mode", tracing::field::display(mode));
        if let Some(detail) = mode.segment_detail() {
            span.record("detail", tracing::field::display(detail));
        }
    }
    /// Runs the compact operation over here which compresses the current conversation
    /// and helps with saving the context for the model
    #[tracing::instrument(
        name = "session.compact",
        skip_all,
        fields(
            session_id = %self.session_info.id.0,
            trigger = "manual",
            mode = tracing::field::Empty,
            detail = tracing::field::Empty,
            pre_tokens = tracing::field::Empty,
            post_tokens = tracing::field::Empty,
            success = tracing::field::Empty,
            error = tracing::field::Empty,
        )
    )]
    pub(crate) async fn run_compact(
        self: &Arc<Self>,
        user_context: Option<String>,
    ) -> Result<(), acp::Error> {
        let (_cancel, _cancel_scope) = self.compaction.cancel.enter();
        let strategy_override = configured_codex_compaction_strategy();
        self.record_compaction_variant();
        // Manual compaction runs between prompt turns and therefore receives
        // its own routing scope instead of replaying the previous turn's state.
        self.chat_state_handle.begin_turn_routing_scope();
        let total_tokens = self.chat_state_handle.get_total_tokens().await;
        tracing::Span::current().record("pre_tokens", total_tokens as i64);
        let sampling_config = self.chat_state_handle.get_sampling_config().await;
        let context_window = sampling_config
            .as_ref()
            .map(|c| c.context_window.get())
            .unwrap_or(DEFAULT_CONTEXT_WINDOW);
        self.maybe_pre_compaction_flush(total_tokens, context_window, "pre_compaction")
            .await;
        if let Err(e) = self
            .run_compact_inner(
                user_context,
                None,
                xai_grok_telemetry::events::CompactionTrigger::Manual,
                strategy_override.as_override(),
            )
            .await
        {
            let span = tracing::Span::current();
            span.record("success", false);
            span.record("error", e.to_string().as_str());
            return Err(e);
        }
        use crate::extensions::notification::SessionUpdate as XaiSessionUpdate;
        let tokens_after = self.chat_state_handle.get_total_tokens().await;
        let span = tracing::Span::current();
        span.record("post_tokens", tokens_after as i64);
        span.record("success", true);
        self.send_xai_notification(XaiSessionUpdate::AutoCompactCompleted {
            tokens_before: Some(total_tokens),
            tokens_after,
            elapsed_ms: None,
            summary_preview: None,
        })
        .await;
        Ok(())
    }
    async fn emit_compact_cancelled(&self, auto_trigger: bool) -> Result<(), acp::Error> {
        if auto_trigger {
            use crate::extensions::notification::SessionUpdate as XaiSessionUpdate;
            self.send_xai_notification(XaiSessionUpdate::AutoCompactCancelled {
                reason: crate::extensions::notification::AutoCompactCancelReason::UserCancelled,
            })
            .await;
        }
        Err(crate::session::helpers::session_compact::CompactFailure::cancelled_error())
    }
    /// Suppress AUTO compaction after a deterministic failure. Scope depends on
    /// the reason (see [`SuppressReason::suppress_state`]): size/schema sticky,
    /// credit until 200, auth until credentials recover, other clears next turn.
    /// Manual `/compact` is exempt.
    async fn suppress_auto_compaction(
        &self,
        reason: SuppressReason,
        estimated_tokens: u64,
        context_window: u64,
    ) {
        self.suppress_auto_compaction_with_error(reason, estimated_tokens, context_window, None)
            .await;
    }

    /// Production suppression path. Adds the original provider/client failure
    /// to the reason-specific guidance so `AutoCompactFailed.error` is useful
    /// without emitting a duplicate notification from the outer lifecycle.
    async fn suppress_auto_compaction_with_error(
        &self,
        reason: SuppressReason,
        estimated_tokens: u64,
        context_window: u64,
        error_detail: Option<&str>,
    ) {
        let new_state = reason.suppress_state();
        if self
            .compaction
            .auto_compact_suppressed
            .compare_exchange(
                SUPPRESS_NONE,
                new_state,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_ok()
        {
            tracing::warn!(
                suppress_reason = reason.as_str(),
                estimated_tokens,
                context_window,
                "auto-compaction suppressed after deterministic compaction failure"
            );
            xai_grok_telemetry::session_ctx::log_event(
                xai_grok_telemetry::events::AutoCompactSuppressed {
                    reason: reason.as_str(),
                    estimated_tokens,
                    context_window,
                },
            );
            let guidance = match reason {
                SuppressReason::CreditBlock => {
                    "out of credits or over your spending limit. Add credits and retry."
                }
                SuppressReason::Auth => {
                    "authentication problem — re-authenticate using /login and retry."
                }
                SuppressReason::Size => "this conversation is too large to compact.",
                SuppressReason::Schema => "this conversation can't be summarized.",
                SuppressReason::Other => {
                    "automatic retry is paused; use /compact manually or start a new session using /new."
                }
            };
            let error = error_detail.map_or_else(
                || guidance.to_string(),
                |detail| {
                    let detail = detail.chars().take(500).collect::<String>();
                    format!("{guidance} Provider error: {detail}")
                },
            );
            self.send_xai_notification(
                crate::extensions::notification::SessionUpdate::AutoCompactFailed { error },
            )
            .await;
        }
    }
    /// Map a deterministic failure's error text to a fixed, content-free
    /// [`SuppressReason`] (drives telemetry + sticky-vs-per-turn scope).
    fn classify_suppress_reason(error_msg: &str) -> SuppressReason {
        let m = error_msg.to_ascii_lowercase();
        if m.contains("spending-limit")
            || m.contains("spending limit")
            || m.contains("out of credits")
            || m.contains("usage balance exhausted")
            || m.contains("usage limit reached")
        {
            SuppressReason::CreditBlock
        } else if is_context_length_error(&m) {
            SuppressReason::Size
        } else if m.contains("status 401") || m.contains("unauthorized") {
            SuppressReason::Auth
        } else if m.contains("invalid_request_error")
            || m.contains("native compact response invalid")
            || m.contains("cannot be replayed exactly")
            || m.contains("serialization error")
        {
            SuppressReason::Schema
        } else {
            SuppressReason::Other
        }
    }

    fn current_epoch_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .min(u128::from(u64::MAX - 1)) as u64
    }

    fn clear_auto_compact_retry_gate(&self) {
        self.compaction.auto_compact_retry_not_before_ms.store(
            AUTO_COMPACT_RETRY_READY,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    fn auto_compact_failure_disposition(error: &acp::Error) -> AutoCompactFailureDisposition {
        let reason = Self::classify_suppress_reason(&Self::acp_error_message(error));
        if reason.suppress_state() == SUPPRESS_STICKY {
            AutoCompactFailureDisposition::RetryAfterReset
        } else {
            AutoCompactFailureDisposition::Cooldown
        }
    }

    /// Record a terminal automatic-compaction failure after its internal retry
    /// budget is exhausted. The disposition belongs to the current lifecycle;
    /// persistent suppression may describe an earlier successful soft overshoot.
    fn record_auto_compact_retry_gate(&self, disposition: AutoCompactFailureDisposition) {
        let retry_not_before_ms = match disposition {
            AutoCompactFailureDisposition::RetryAfterReset => AUTO_COMPACT_RETRY_AFTER_RESET,
            AutoCompactFailureDisposition::Cooldown => {
                Self::current_epoch_ms().saturating_add(AUTO_COMPACT_FAILURE_COOLDOWN_MS)
            }
        };
        self.compaction
            .auto_compact_retry_not_before_ms
            .store(retry_not_before_ms, std::sync::atomic::Ordering::Relaxed);
    }

    /// Return an error when a previous automatic-compaction lifecycle has
    /// already established that another immediate network request is unsafe or
    /// wasteful. The Codex hard gate still calls this method and surfaces the
    /// error, so sampling never proceeds past its provider safety limit.
    fn auto_compact_retry_gate_error(&self) -> Option<acp::Error> {
        let retry_not_before_ms = self
            .compaction
            .auto_compact_retry_not_before_ms
            .load(std::sync::atomic::Ordering::Relaxed);
        if retry_not_before_ms == AUTO_COMPACT_RETRY_READY {
            return None;
        }
        if retry_not_before_ms == AUTO_COMPACT_RETRY_AFTER_RESET {
            return Some(acp::Error::internal_error().data(
                "automatic compaction previously failed deterministically; change the model or context, rewind, or run /compact manually before retrying",
            ));
        }

        let now_ms = Self::current_epoch_ms();
        if now_ms >= retry_not_before_ms {
            let _ = self
                .compaction
                .auto_compact_retry_not_before_ms
                .compare_exchange(
                    retry_not_before_ms,
                    AUTO_COMPACT_RETRY_READY,
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                );
            return None;
        }

        let remaining_secs = retry_not_before_ms
            .saturating_sub(now_ms)
            .saturating_add(999)
            / 1_000;
        Some(acp::Error::internal_error().data(format!(
            "automatic compaction is cooling down after a failed lifecycle; retry in {remaining_secs}s"
        )))
    }
    /// ACP error payload string (plain string or `{message, ...}`).
    fn acp_error_message(err: &acp::Error) -> String {
        match err.data.as_ref() {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(obj) => obj
                .get("message")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| obj.to_string()),
            None => err.message.clone(),
        }
    }
    /// Auth/401 compact failure — abort for reauth resubmit; don't sample oversized.
    pub(crate) fn is_auth_compact_error(err: &acp::Error) -> bool {
        matches!(
            Self::classify_suppress_reason(&Self::acp_error_message(err)),
            SuppressReason::Auth
        )
    }
    /// Terminal auth compact failure: emit RetryState auth (reauth stash) + auth_required.
    /// Separate from `AutoCompactFailed` (user-facing); this aborts the turn.
    pub(crate) async fn surface_compact_auth_failure(&self, err: acp::Error) -> acp::Error {
        use crate::extensions::notification::SessionUpdate as XaiSessionUpdate;
        let detailed = Self::acp_error_message(&err);
        let message = if detailed.to_ascii_lowercase().contains("unauthorized") {
            detailed
        } else {
            format!(
                "Unauthorized (401): compaction failed — re-authenticate with /login \
                 and retry. ({detailed})"
            )
        };
        tracing::warn!(
            session_id = %self.session_info.id.0,
            error = %message,
            "auto-compact auth failure: aborting turn for re-auth"
        );
        xai_grok_telemetry::unified_log::warn(
            "auto-compact auth failure: aborting turn for re-auth",
            Some(self.session_info.id.0.as_ref()),
            Some(serde_json::json!({
                "message": crate::util::truncate(&message, 300),
            })),
        );
        self.send_xai_notification(XaiSessionUpdate::RetryState(
            crate::extensions::notification::RetryState::Failed {
                error_type: "auth".to_string(),
                message: message.clone(),
            },
        ))
        .await;
        acp::Error::auth_required().data(crate::sampling::error::terminal_error_data(
            message,
            Some(401),
            xai_grok_sampler::SamplingErrorKind::Auth,
        ))
    }
    /// Clear [`SUPPRESS_AUTH`] on login/token refresh (credit suppress waits for a 200).
    pub(crate) fn clear_auth_compact_suppression(&self) {
        if self
            .compaction
            .auto_compact_suppressed
            .compare_exchange(
                SUPPRESS_AUTH,
                SUPPRESS_NONE,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_ok()
        {
            self.clear_auto_compact_retry_gate();
        }
    }
    /// Credit or auth suppress — a model switch cannot clear these.
    fn is_account_state_suppressed(&self) -> bool {
        matches!(
            self.compaction
                .auto_compact_suppressed
                .load(std::sync::atomic::Ordering::Relaxed),
            SUPPRESS_UNTIL_SUCCESS | SUPPRESS_AUTH
        )
    }
    /// Choose the post-compaction history for a forked session: re-pin the inherited
    /// prefix, or release it (fall back to the self-contained summary the summarizer
    /// already built from the whole conversation) when re-pinning would leave the fork
    /// at/over the auto-compact threshold. On release, sets the sticky flag and records
    /// the release span field (this runs within the `run_compact_inner` span).
    ///
    /// This runtime release compensates for a verbatim mirror-fork that pinned its whole
    /// parent transcript; bounding the inherited prefix at fork admission is the
    /// structural alternative that would remove this path.
    async fn resolve_forked_compacted_history(
        &self,
        compacted_history: Vec<ConversationItem>,
        prefix_len: usize,
        tokens_before: u64,
        context_window: u64,
    ) -> Vec<ConversationItem> {
        let full_conv = self.chat_state_handle.get_conversation().await;
        let compacted_len = compacted_history.len();
        let release_candidate = compacted_history.clone();
        match preserve_inherited_prefix(&full_conv, compacted_history, prefix_len) {
            Ok(preserved) => {
                let projected_preserved = project_preserved_reseed_tokens(
                    xai_chat_state::estimate_conversation_tokens(&preserved),
                    tokens_before,
                    xai_chat_state::estimate_conversation_tokens(&full_conv),
                );
                if xai_token_estimation::exceeds_threshold(
                    projected_preserved,
                    context_window,
                    self.compaction.threshold_percent.get(),
                ) {
                    self.compaction
                        .prefix_released
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    tracing::Span::current().record("compaction_prefix_released", true);
                    tracing::info!(
                        session_id = %self.session_info.id.0,
                        prefix_len,
                        projected_preserved,
                        "compaction: releasing inherited prefix under pressure"
                    );
                    release_candidate
                } else {
                    tracing::info!(
                        session_id = %self.session_info.id.0,
                        prefix_len,
                        compacted_len,
                        "Preserving inherited prefix across compaction"
                    );
                    preserved
                }
            }
            Err(original) => {
                tracing::warn!(
                    session_id = %self.session_info.id.0,
                    prefix_len,
                    conversation_len = full_conv.len(),
                    "Inherited prefix invalid, using compacted history as-is"
                );
                original
            }
        }
    }
    /// Inner implementation of compaction that supports an optional `auto_continue`
    /// payload for the checkpoint.
    #[tracing::instrument(
        name = "session.compact_inner",
        skip_all,
        fields(
            session_id = %self.session_info.id.0,
            compaction_tokens_before = tracing::field::Empty,
            compaction_tokens_after = tracing::field::Empty,
            compaction_summary_chars = tracing::field::Empty,
            compaction_degenerate_rejections = tracing::field::Empty,
            compaction_input_overflow_rejections = tracing::field::Empty,
            compaction_deterministic_rejections = tracing::field::Empty,
            compaction_transient_rejections = tracing::field::Empty,
            compaction_attempts = tracing::field::Empty,
            compaction_trigger = tracing::field::Empty,
            compaction_trigger_pct = tracing::field::Empty,
            compaction_threshold_pct = tracing::field::Empty,
            compaction_outcome = tracing::field::Empty,
            compaction_strategy = tracing::field::Empty,
            compaction_stop_reason = tracing::field::Empty,
            compaction_ttft_ms = tracing::field::Empty,
            compaction_stream_ms = tracing::field::Empty,
            compaction_delta_count = tracing::field::Empty,
            compaction_itl_max_ms = tracing::field::Empty,
            compaction_two_pass_used = tracing::field::Empty,
            compaction_prefire_hit = tracing::field::Empty,
            compaction_pass2_latency_ms = tracing::field::Empty,
            compaction_prefire_waited_ms = tracing::field::Empty,
            compaction_prefire_stale = tracing::field::Empty,
            compaction_prefix_released = tracing::field::Empty,
        )
    )]
    async fn run_compact_inner(
        &self,
        user_context: Option<String>,
        auto_continue: Option<crate::extensions::notification::AutoContinueInfo>,
        trigger: xai_grok_telemetry::events::CompactionTrigger,
        strategy_override: CompactionStrategyOverride<'_>,
    ) -> Result<(), acp::Error> {
        let (cancel, _cancel_scope) = self.compaction.cancel.enter();
        let tokens_before = self.chat_state_handle.get_total_tokens().await;
        tracing::Span::current().record("compaction_tokens_before", tokens_before as i64);
        self.signals_handle().record_compaction(tokens_before);
        let trigger_str = match trigger {
            xai_grok_telemetry::events::CompactionTrigger::Manual => "manual",
            xai_grok_telemetry::events::CompactionTrigger::Auto => "auto",
        };
        let sampling_config = self.chat_state_handle.get_sampling_config().await;
        let context_window = sampling_config
            .as_ref()
            .map(|c| c.context_window.get())
            .unwrap_or(DEFAULT_CONTEXT_WINDOW);
        {
            let span = tracing::Span::current();
            let trigger_pct = if context_window == 0 {
                0
            } else {
                ((tokens_before as f64 / context_window as f64) * 100.0).round() as i64
            };
            span.record("compaction_trigger_pct", trigger_pct);
            span.record(
                "compaction_threshold_pct",
                self.compaction.threshold_percent.get() as i64,
            );
            span.record("compaction_trigger", trigger_str);
        }
        let summary_strips_reasoning = sampling_config
            .as_ref()
            .map(|c| c.api_backend == ApiBackend::Messages)
            .unwrap_or(false);
        let model_id = sampling_config.map(|c| c.model).unwrap_or_default();
        let compaction = xai_grok_telemetry::events::CompactionScope::begin(
            trigger,
            tokens_before,
            context_window,
            model_id.clone(),
            user_context.is_some(),
        );
        let compact_source = trigger_str;
        self.dispatch_hook(
            xai_grok_hooks::event::HookEventName::PreCompact,
            xai_grok_hooks::event::HookPayload::PreCompact {
                source: compact_source.into(),
            },
            None,
            None,
        )
        .await;
        let max_retries = 3u32;
        let retry_delay_secs = 3u64;
        let (conv_len, system_message, full_conversation) = tokio::join!(
            self.chat_state_handle.get_conversation_len(),
            self.chat_state_handle.get_system_message(),
            self.chat_state_handle.get_conversation(),
        );
        let segment_messages = if self.compaction.compaction_mode.writes_segments() {
            xai_chat_state::compaction_utils::prepare_conversation_for_segment(
                full_conversation.clone(),
            )
        } else {
            Vec::new()
        };
        let verbatim_input_enabled = self.compaction.verbatim_input;
        // Native receives this exact source history (system is normalized into
        // `instructions` at the sampling boundary). Local summary variants use
        // their existing verbatim/lossy preparation below.
        let native_source_conversation = full_conversation.clone();
        let simplified_messages = if verbatim_input_enabled {
            xai_chat_state::compaction_utils::prepare_conversation_for_verbatim_summarization(
                full_conversation,
                summary_strips_reasoning,
            )
        } else {
            xai_chat_state::compaction_utils::prepare_conversation_for_summarization(
                full_conversation,
            )
        };
        if conv_len == 0 {
            tracing::error!(
                session_id = %self.session_info.id.0,
                "Compaction failed: conversation is empty (ChatStateActor may have died)"
            );
            return Err(
                acp::Error::internal_error().data("Compaction failed: conversation is empty")
            );
        }
        let system_message = match system_message {
            Some(msg) => msg,
            None => {
                tracing::error!(
                    session_id = %self.session_info.id.0,
                    conversation_len = conv_len,
                    "Compaction failed: no system message in conversation history"
                );
                return Err(acp::Error::internal_error()
                    .data("Compaction failed: no system message in conversation history"));
            }
        };
        if simplified_messages.is_empty() {
            tracing::error!(
                session_id = %self.session_info.id.0,
                conversation_len = conv_len,
                "Compaction failed: simplified conversation is empty"
            );
            return Err(acp::Error::internal_error()
                .data("Compaction failed: simplified conversation is empty"));
        }
        if !simplified_messages
            .iter()
            .any(|msg| matches!(msg, ConversationItem::System(_)))
        {
            tracing::error!(
                session_id = %self.session_info.id.0,
                conversation_len = conv_len,
                simplified_len = simplified_messages.len(),
                "Compaction failed: no system message in simplified conversation"
            );
            return Err(acp::Error::internal_error()
                .data("Compaction failed: no system message in simplified conversation"));
        }
        let sampling_config = self.reconstruct_full_config().await;
        let provider = xai_grok_sampling_types::resolve_provider(
            sampling_config.provider_id,
            sampling_config.api_backend.clone(),
            &sampling_config.base_url,
        );
        let sampling_client = self.prepare_chat_completion(false).await?;
        let backend_search_active = self.backend_search_active();
        let effective_tool_defs: Vec<xai_grok_sampling_types::ToolDefinition> = self
            .prepare_tool_definitions()
            .await
            .into_iter()
            .filter(|td| !backend_search_active || td.function.name != "web_search")
            .collect();
        let compaction_tool_tokens =
            xai_chat_state::estimate_tool_definitions_tokens(&effective_tool_defs);
        let compaction_tools: Vec<xai_grok_sampling_types::ToolSpec> = effective_tool_defs
            .into_iter()
            .map(xai_grok_sampling_types::ToolSpec::from)
            .collect();
        let compaction_hosted_tools: Vec<xai_grok_sampling_types::HostedTool> =
            self.hosted_tools_for_turn();
        tracing::info!(
            num_tools = compaction_tools.len(),
            tool_tokens = compaction_tool_tokens,
            "Running compact with model '{}' (user model: '{}')",
            &sampling_config.model,
            &sampling_config.model
        );
        let auto_trigger = matches!(trigger, xai_grok_telemetry::events::CompactionTrigger::Auto);
        let mut strategy = select_compaction_strategy(
            provider.capabilities(),
            strategy_override,
            user_context.is_some(),
        );
        if user_context.is_some() && strategy == CompactionStrategy::LocalSummary {
            tracing::info!("using local summary because manual compaction supplied user context");
        }
        let mut product: Option<CompactionProduct> = None;
        let mut native_counters = NativeCompactionCounters::default();
        let mut native_remote_fallback = false;
        if strategy == CompactionStrategy::NativeCodex {
            let turn_routing_state = self.chat_state_handle.get_turn_routing_state().await;
            let sampling_identity =
                sampling_client.sampling_identity_for_model(&sampling_config.model);
            let native_outcome = run_native_compaction(
                CodexCompactionInput {
                    source_conversation: native_source_conversation,
                    canonical_system_message: system_message.clone(),
                    tools: compaction_tools.clone(),
                    hosted_tools: compaction_hosted_tools.clone(),
                    sampling_identity,
                    reasoning_effort: sampling_config.reasoning_effort,
                    session_id: self.session_info.id.to_string(),
                    turn_routing_state,
                    context_window,
                    tool_tokens: compaction_tool_tokens,
                },
                &sampling_client,
            )
            .await;
            match native_outcome {
                NativeCompactionOutcome::Success {
                    replacement_history,
                    output,
                    counters,
                } => {
                    native_counters = counters;
                    product = Some(CompactionProduct::NativeCodex {
                        replacement_history,
                        output: CompactOutput {
                            content: String::new(),
                            stop_reason: Some(output.stop_reason),
                            truncated: output.truncated,
                            ttft_ms: output.ttft_ms,
                            stream_ms: output.stream_ms,
                            delta_count: output.delta_count,
                            itl_max_ms: output.itl_max_ms,
                        },
                    });
                }
                NativeCompactionOutcome::LocalFallback { reason, counters } => {
                    native_counters = counters;
                    strategy = CompactionStrategy::LocalSummary;
                    native_remote_fallback =
                        reason != NativeCompactionFallbackReason::StructuralMinimumTooLarge;
                }
                NativeCompactionOutcome::HardFailure {
                    source,
                    counters,
                    estimated_input_tokens,
                } => {
                    let suppress_auto = source.suppresses_auto_compaction();
                    let message = source.message();
                    let span = tracing::Span::current();
                    span.record(
                        "compaction_strategy",
                        CompactionStrategy::NativeCodex.as_str(),
                    );
                    span.record("compaction_attempts", counters.attempts as i64);
                    span.record(
                        "compaction_transient_rejections",
                        counters.transient_rejections as i64,
                    );
                    span.record("compaction_outcome", CompactionOutcome::Failed.as_str());
                    if auto_trigger && suppress_auto {
                        self.suppress_auto_compaction_with_error(
                            Self::classify_suppress_reason(&message),
                            estimated_input_tokens,
                            context_window,
                            Some(&message),
                        )
                        .await;
                    }
                    return Err(acp::Error::internal_error().data(message));
                }
            }
        }
        tracing::Span::current().record(
            "compaction_strategy",
            if native_remote_fallback {
                "local_summary_fallback"
            } else {
                strategy.as_str()
            },
        );
        let mut last_error: Option<acp::Error> = None;
        let mut last_failure_outcome = CompactionOutcome::Failed;
        let use_short_prompt = false;
        let started_at = chrono::Utc::now().to_rfc3339();
        let estimated_input_tokens =
            xai_chat_state::estimate_conversation_tokens(&simplified_messages);
        let summary_prompt_tokens =
            xai_chat_state::estimate_conversation_tokens(&[ConversationItem::user(
                build_compaction_prompt(user_context.as_deref(), use_short_prompt),
            )]);
        let fitted_history_budget = fitted_compaction_history_budget(
            context_window,
            compaction_tool_tokens,
            summary_prompt_tokens,
        );
        let mut input_stage = if strategy.uses_local_summary_pipeline() {
            initial_compaction_input_stage(
                verbatim_input_enabled,
                auto_trigger,
                provider.capabilities(),
                estimated_input_tokens,
                context_window,
                compaction_tool_tokens,
                summary_prompt_tokens,
            )
        } else {
            // Native compaction has already completed above. Its request used
            // the exact source transcript and never enters the local
            // verbatim-fit-lossy ladder.
            CompactionInputStage::Verbatim
        };
        let prefit_first_request = strategy.uses_local_summary_pipeline()
            && input_stage == CompactionInputStage::VerbatimFitted;
        let wall_clock_budget_secs = self
            .agent
            .borrow()
            .compaction_policy()
            .wall_clock_budget_secs;
        let sampler = crate::session::helpers::full_replace_compaction::ShellCompactionSampler::new(
            use_short_prompt,
            user_context.clone(),
            compaction_tools.clone(),
            compaction_hosted_tools.clone(),
            compaction_tool_tokens,
            sampling_client,
            self.session_info.id.clone(),
            sampling_config.clone(),
            self.inference_idle_timeout,
            wall_clock_budget_secs,
            self.compaction.tool_choice,
            cancel.clone(),
        );
        let observer =
            crate::session::helpers::full_replace_compaction::ShellFullReplaceObserver::new(
                trigger,
                context_window,
                compaction.compaction_id.clone(),
                self.session_info.id.0.to_string(),
                estimated_input_tokens,
                retry_delay_secs,
            );
        let fr_config = xai_grok_compaction::FullReplaceConfig {
            max_attempts: max_retries,
            retry_delay_secs,
            sampling_timeout_secs: 0,
        };
        let mut request_turns = if prefit_first_request {
            let fitted = xai_chat_state::compaction_utils::fit_conversation_to_budget(
                simplified_messages.clone(),
                fitted_history_budget,
            );
            let fitted_tokens = xai_chat_state::estimate_conversation_tokens(&fitted);
            let fits_budget = fitted_tokens <= fitted_history_budget;
            if !fits_budget {
                tracing::warn!(
                    session_id = %self.session_info.id.0,
                    fitted_tokens,
                    fitted_history_budget,
                    "Automatic Codex compaction retained a structural minimum larger than the calculated history budget"
                );
            }
            tracing::info!(
                session_id = %self.session_info.id.0,
                original_items = simplified_messages.len(),
                fitted_items = fitted.len(),
                estimated_input_tokens,
                fitted_tokens,
                fitted_history_budget,
                fits_budget,
                context_window,
                summary_reserve_tokens = SUMMARY_BUDGET_RESERVE_TOKENS,
                summary_prompt_tokens,
                compaction_tool_tokens,
                "Pre-fitting first automatic Codex compaction request"
            );
            fitted
        } else {
            simplified_messages.clone()
        };
        let mut input_overflow_rejections: u32 = 0;
        let two_pass_enabled = self.two_pass_active();
        let two_pass_output = if strategy.uses_local_summary_pipeline()
            && two_pass_allowed_for_initial_stage(two_pass_enabled, input_stage)
        {
            self.try_two_pass_pass2_apply(user_context.as_deref(), summary_strips_reasoning)
                .await
        } else {
            if two_pass_enabled {
                tracing::info!(
                    session_id = %self.session_info.id.0,
                    strategy = strategy.as_str(),
                    input_stage = input_stage.as_str(),
                    "Bypassing two-pass apply for native or pre-fitted compaction"
                );
            }
            None
        };
        let mut used_two_pass = false;
        if product.is_none() {
            if let Some(output) = two_pass_output {
                used_two_pass = true;
                product = Some(CompactionProduct::LocalSummary {
                    summary: output.content.clone(),
                    output,
                });
            } else {
                while product.is_none() {
                    match xai_grok_compaction::sample_full_replace_summary(
                        &sampler,
                        &request_turns,
                        user_context.as_deref(),
                        &fr_config,
                        &observer,
                    )
                    .await
                    {
                        Ok(summary) => {
                            let output = sampler.take_last_success().expect(
                                "a successful local full-replace sample stashes its CompactOutput",
                            );
                            product = Some(CompactionProduct::LocalSummary {
                                summary: summary.summary,
                                output,
                            });
                            break;
                        }
                        Err(xai_grok_compaction::FullReplaceError::NothingToCompact) => {
                            last_error = Some(
                                acp::Error::internal_error()
                                    .data("compact failed: nothing to compact"),
                            );
                            break;
                        }
                        Err(xai_grok_compaction::FullReplaceError::EmptyResponse) => {
                            last_failure_outcome = if observer.degenerate_seen() {
                                CompactionOutcome::Degenerate
                            } else {
                                CompactionOutcome::Transient
                            };
                            last_error = Some(acp::Error::internal_error().data(
                                observer.last_error_message().unwrap_or_else(|| {
                                    "compact failed: model returned empty response".to_string()
                                }),
                            ));
                            break;
                        }
                        Err(xai_grok_compaction::FullReplaceError::Sampler {
                            message,
                            deterministic,
                            context_overflow,
                        }) => {
                            if cancel.is_cancelled()
                                || message.contains(
                                    crate::session::helpers::session_compact::COMPACT_CANCELLED_MSG,
                                )
                            {
                                return self.emit_compact_cancelled(auto_trigger).await;
                            }
                            if context_overflow {
                                let next_stage = input_stage.after_context_overflow();
                                if let Some(stage) = next_stage {
                                    input_overflow_rejections += 1;
                                    xai_grok_telemetry::session_ctx::log_event(
                                        xai_grok_telemetry::events::CompactionRetryDegraded {
                                            trigger,
                                            reason: "input_overflow",
                                            from_stage: Some(input_stage.as_str()),
                                            to_stage: Some(stage.as_str()),
                                            summary_chars: None,
                                            attempt: observer.attempt_count(),
                                            context_window,
                                            compaction_id: compaction.compaction_id.clone(),
                                        },
                                    );
                                    tracing::warn!(
                                        session_id = %self.session_info.id.0,
                                        ?stage,
                                        error = %message,
                                        "Compaction input overflowed deterministically; stepping down the input ladder to avoid an incompactable state"
                                    );
                                    let conv = self.chat_state_handle.get_conversation().await;
                                    request_turns = match stage {
                                        CompactionInputStage::VerbatimFitted => {
                                            let verbatim = xai_chat_state::compaction_utils::prepare_conversation_for_verbatim_summarization(
                                        conv,
                                        summary_strips_reasoning,
                                    );
                                            xai_chat_state::compaction_utils::fit_conversation_to_budget(
                                        verbatim,
                                        fitted_history_budget,
                                    )
                                        }
                                        CompactionInputStage::Lossy => {
                                            let lossy_budget = (context_window.saturating_mul(7)
                                                / 10)
                                                .saturating_sub(compaction_tool_tokens);
                                            xai_chat_state::compaction_utils::fit_conversation_to_budget(
                                        xai_chat_state::compaction_utils::prepare_conversation_for_summarization(
                                            conv,
                                        ),
                                        lossy_budget,
                                    )
                                        }
                                        CompactionInputStage::Verbatim => {
                                            unreachable!("ladder only steps forward")
                                        }
                                    };
                                    input_stage = stage;
                                    continue;
                                }
                                last_failure_outcome = CompactionOutcome::Deterministic;
                                if auto_trigger {
                                    self.suppress_auto_compaction_with_error(
                                        SuppressReason::Size,
                                        estimated_input_tokens,
                                        context_window,
                                        Some(&message),
                                    )
                                    .await;
                                }
                                last_error = Some(acp::Error::internal_error().data(message));
                                break;
                            }
                            if deterministic {
                                last_failure_outcome = CompactionOutcome::Deterministic;
                                if auto_trigger {
                                    let reason = Self::classify_suppress_reason(&message);
                                    self.suppress_auto_compaction_with_error(
                                        reason,
                                        estimated_input_tokens,
                                        context_window,
                                        Some(&message),
                                    )
                                    .await;
                                }
                                last_error = Some(acp::Error::internal_error().data(message));
                                break;
                            }
                            last_failure_outcome = CompactionOutcome::Transient;
                            last_error = Some(acp::Error::internal_error().data(message));
                            break;
                        }
                    }
                }
            }
        }
        let telemetry = observer.into_telemetry();
        // Native retries and an endpoint-unavailable probe followed by local
        // fallback are part of the same compaction lifecycle.
        let total_attempts = native_counters.attempts.saturating_add(telemetry.attempts);
        let total_transient_rejections = native_counters
            .transient_rejections
            .saturating_add(telemetry.transient_rejections);
        if strategy.uses_local_summary_pipeline()
            && !used_two_pass
            && let Some(request_chat_history) = sampler.take_last_attempted_items()
        {
            self.persist_compaction_request_artifact(
                request_chat_history,
                compaction_tools,
                user_context.as_deref(),
                use_short_prompt,
                &sampling_config.model,
                trigger,
                product
                    .as_ref()
                    .and_then(CompactionProduct::local_summary)
                    .or(telemetry.last_rejected_summary.as_deref()),
                last_error.as_ref(),
                telemetry.attempts,
                telemetry.attempt_details,
                started_at,
            );
        }
        let product = match product {
            Some(product) => product,
            None => {
                let span = tracing::Span::current();
                span.record("compaction_attempts", total_attempts as i64);
                span.record(
                    "compaction_degenerate_rejections",
                    telemetry.degenerate_rejections as i64,
                );
                span.record(
                    "compaction_input_overflow_rejections",
                    input_overflow_rejections as i64,
                );
                span.record(
                    "compaction_deterministic_rejections",
                    telemetry.deterministic_rejections as i64,
                );
                span.record(
                    "compaction_transient_rejections",
                    total_transient_rejections as i64,
                );
                span.record("compaction_outcome", last_failure_outcome.as_str());
                return Err(last_error.unwrap_or_else(|| {
                    acp::Error::internal_error().data("compaction failed: unknown error")
                }));
            }
        };
        let (replacement, compact_output) = product.into_parts();
        let user_message_prefix = self.build_user_message_prefix().await;
        let conversation = self.chat_state_handle.get_conversation().await;
        let (discovered_agents_md, all_skills_for_compaction, _agent_edited_paths, state_context) =
            if use_short_prompt {
                let empty_edited: std::collections::BTreeSet<String> = Default::default();
                let ctx =
                    CompactionStateContext::build(&conversation, CompactionInputs::default()).await;
                (Vec::<std::path::PathBuf>::new(), vec![], empty_edited, ctx)
            } else {
                let agents_md: Vec<std::path::PathBuf> = self
                    .agent
                    .borrow()
                    .tool_bridge()
                    .agents_md_reminded_paths()
                    .await
                    .into_iter()
                    .collect();
                let skills = self.slash_skills_for_resolve().await;
                let edited_paths = self.chat_state_handle.get_agent_edited_paths().await;
                let ctx = {
                    let bridge_tasks = self
                        .agent
                        .borrow()
                        .tool_bridge()
                        .list_background_tasks()
                        .await;
                    let pending_tasks: Vec<_> =
                        bridge_tasks.into_iter().filter(|t| !t.completed).collect();
                    let (execute_tool_name, monitor_tool_name) = if pending_tasks.is_empty() {
                        (None, None)
                    } else {
                        let agent_ref = self.agent.borrow();
                        let bridge = agent_ref.tool_bridge();
                        let empty = serde_json::json!({});
                        let execute = bridge
                            .render_prompt("${{ tools.by_kind.execute }}", &empty)
                            .await
                            .filter(|s| !s.is_empty() && !s.contains("by_kind"));
                        let monitor = bridge
                            .render_prompt("${{ tools.by_kind.monitor }}", &empty)
                            .await
                            .filter(|s| !s.is_empty() && !s.contains("by_kind"));
                        (execute, monitor)
                    };
                    let running_tasks: Vec<_> = pending_tasks
                        .into_iter()
                        .map(|t| {
                            let tool_name = match t.kind {
                                xai_grok_tools::computer::types::TaskKind::Monitor => {
                                    monitor_tool_name.clone()
                                }
                                xai_grok_tools::computer::types::TaskKind::Bash => {
                                    execute_tool_name.clone()
                                }
                            };
                            CompactionStateContext::task_summary(
                                t.task_id, t.command, "running", tool_name,
                            )
                        })
                        .collect();
                    let running_subagents = if let Some(ref event_tx) =
                        self.tool_context.subagent_event_tx
                    {
                        let (tx, rx) = tokio::sync::oneshot::channel();
                        use xai_grok_tools::implementations::grok_build::task::types::{
                            SubagentEvent, SubagentListActiveRequest,
                        };
                        let _ =
                            event_tx.send(SubagentEvent::ListActive(SubagentListActiveRequest {
                                parent_session_id: self.session_id_string(),
                                respond_to: tx,
                            }));
                        rx.await
                        .unwrap_or_default()
                        .into_iter()
                        .map(|s| crate::session::helpers::compaction_context::RunningSubagentSummary {
                            subagent_id: s.subagent_id,
                            subagent_type: s.subagent_type,
                            description: s.description,
                            elapsed_ms: s.elapsed_ms,
                        })
                        .collect()
                    } else {
                        vec![]
                    };
                    let connected_mcp_servers = {
                        use crate::session::helpers::compaction_context::CompactionServerSummary;
                        use xai_grok_tools::implementations::search_tool::{
                            sanitize_description, truncate_description,
                        };
                        self.connected_server_summaries()
                            .into_iter()
                            .map(|s| {
                                let desc = s
                                    .description
                                    .map(|d| truncate_description(&sanitize_description(&d)))
                                    .filter(|d| !d.is_empty());
                                CompactionServerSummary {
                                    name: s.name,
                                    tool_count: s.tool_count,
                                    description: desc,
                                }
                            })
                            .collect()
                    };
                    let todos = {
                        use crate::session::helpers::compaction_context::{
                            TodoSummary, TodoSummaryStatus,
                        };
                        use crate::tools::todo::{TodoState, TodoStatus};
                        use xai_grok_tools::types::resources::State;
                        let bridge = self.agent.borrow().tool_bridge().clone();
                        bridge
                            .read_resource::<State<TodoState>>()
                            .await
                            .map(|s| {
                                s.0.todo_items_with_ids()
                                    .map(|(id, item)| TodoSummary {
                                        id: id.clone(),
                                        content: item.content.clone(),
                                        status: match item.status {
                                            TodoStatus::Pending => TodoSummaryStatus::Pending,
                                            TodoStatus::InProgress => TodoSummaryStatus::InProgress,
                                            TodoStatus::Completed => TodoSummaryStatus::Completed,
                                            TodoStatus::Cancelled => TodoSummaryStatus::Cancelled,
                                        },
                                    })
                                    .collect()
                            })
                            .unwrap_or_default()
                    };
                    CompactionStateContext::build(
                        &conversation,
                        CompactionInputs {
                            running_tasks,
                            running_subagents,
                            agent_edited_paths: edited_paths.clone(),
                            connected_mcp_servers,
                            todos,
                            ..Default::default()
                        },
                    )
                    .await
                };
                (agents_md, skills, edited_paths, ctx)
            };
        use crate::session::helpers::compaction_context::SubagentToolNames;
        let subagent_tool_names: Option<SubagentToolNames> =
            if use_short_prompt || state_context.running_subagents.is_empty() {
                None
            } else {
                let agent_ref = self.agent.borrow();
                let bridge = agent_ref.tool_bridge();
                let empty = serde_json::json!({});
                let poll_name = bridge
                    .render_prompt("${{ tools.by_kind.background_task_action }}", &empty)
                    .await
                    .filter(|s| !s.is_empty() && !s.contains("by_kind"));
                let cancel_name = bridge
                    .render_prompt("${{ tools.by_kind.kill_task_action }}", &empty)
                    .await
                    .filter(|s| !s.is_empty() && !s.contains("by_kind"));
                match (poll_name, cancel_name) {
                    (Some(poll), Some(cancel)) => Some(SubagentToolNames { poll, cancel }),
                    (poll, cancel) => {
                        tracing::warn!(
                            session_id = %self.session_info.id.0,
                            poll_resolved = poll.is_some(),
                            cancel_resolved = cancel.is_some(),
                            "could not resolve subagent tool names, \
                             omitting subagent reminder from compacted conversation"
                        );
                        None
                    }
                }
            };
        use crate::session::helpers::compaction_context::McpToolNames;
        let mcp_tool_names: Option<McpToolNames> =
            if use_short_prompt || state_context.connected_mcp_servers.is_empty() {
                None
            } else {
                let agent_ref = self.agent.borrow();
                let bridge = agent_ref.tool_bridge();
                let empty = serde_json::json!({});
                let search_name = bridge
                    .render_prompt("${{ tools.by_kind.search_tool }}", &empty)
                    .await
                    .filter(|s| !s.is_empty() && !s.contains("by_kind"));
                let call_name = bridge
                    .render_prompt("${{ tools.by_kind.use_tool }}", &empty)
                    .await
                    .filter(|s| !s.is_empty() && !s.contains("by_kind"));
                match (search_name, call_name) {
                    (Some(search), Some(call)) => Some(McpToolNames { search, call }),
                    _ => None,
                }
            };
        let memory_backend_impl = {
            let g = self.memory.storage.borrow();
            g.as_ref()
                .zip(self.memory.backend_params.as_ref())
                .map(|(storage, params)| {
                    crate::session::memory::MemoryBackendImpl::from_session_params(
                        storage.clone(),
                        &crate::session::memory::MemoryBackendParams {
                            search_source: "compaction_recovery",
                            ..params.clone()
                        },
                    )
                })
        };
        let memory_opt_out = false;
        let memory_ref: Option<&dyn xai_grok_tools::types::memory_backend::MemoryBackend> =
            if memory_opt_out {
                None
            } else {
                memory_backend_impl
                    .as_ref()
                    .map(|b| b as &dyn xai_grok_tools::types::memory_backend::MemoryBackend)
            };
        let suppress_state_reminder = false;
        let system_reminder = if suppress_state_reminder {
            None
        } else {
            to_system_reminder(
                &state_context,
                &discovered_agents_md,
                &all_skills_for_compaction,
                memory_ref,
                subagent_tool_names.as_ref(),
                mcp_tool_names.as_ref(),
            )
            .await
        };
        let system_reminder = {
            let plan_path = {
                let guard = self.plan_mode.lock();
                guard
                    .is_active()
                    .then(|| guard.plan_file_path().to_path_buf())
            };
            if let Some(plan_path) = plan_path {
                let plan_has_content =
                    crate::session::plan_mode::plan_file_has_content(&plan_path).await;
                let template = crate::session::plan_mode::plan_mode_reminder_full_template();
                let wrapper = self.reminder_wrapper_tag();
                let rendered = self
                    .render_plan_template(template, &plan_path, plan_has_content)
                    .await;
                match (system_reminder, rendered) {
                    (Some(mut existing), Some(plan_section)) => {
                        if let Some(pos) = existing.rfind("</system-reminder>") {
                            existing.insert_str(pos, &format!("\n\n{}\n", plan_section));
                        } else {
                            existing.push_str("\n\n");
                            existing.push_str(&plan_section);
                        }
                        Some(existing)
                    }
                    (None, Some(plan_section)) => Some(format!(
                        "<{tag}>\n{body}\n</{tag}>",
                        tag = wrapper,
                        body = plan_section,
                    )),
                    (existing, None) => {
                        tracing::warn!(
                            session_id = %self.session_info.id.0,
                            "compaction: plan mode active but template render failed"
                        );
                        existing
                    }
                }
            } else {
                system_reminder
            }
        };
        if let Some(ref recovery_backend) = memory_backend_impl {
            let n = recovery_backend
                .search_counter
                .load(std::sync::atomic::Ordering::Relaxed);
            if n > 0 {
                self.memory
                    .compaction_recovery_count
                    .fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                tracing::debug!(
                    target: xai_grok_telemetry::memory_log::TARGET,
                    count = n,
                    "MEMORY_COMPACTION_RECOVERY: {} search(es) performed",
                    n,
                );
            }
        }
        if cancel.is_cancelled() {
            return self.emit_compact_cancelled(auto_trigger).await;
        }
        let agents_md_reminder = self.agent.borrow().agents_md_user_reminder();
        let compaction_context = state_context.for_compaction();
        let compaction_state_context: &CompactionStateContext = &compaction_context;
        if let CompactionReplacement::LocalSummary(summary) = &replacement {
            self.persist_compaction_segment(&segment_messages, summary);
        }
        let transcript_hint = self.transcript_hint();
        let summary_count = self
            .compaction
            .count
            .load(std::sync::atomic::Ordering::Relaxed);
        let compacted_history = match replacement {
            CompactionReplacement::NativeCodex(replacement) => {
                // Install provider-authored structured history verbatim (apart
                // from the canonical System item restored by the adapter). Do
                // not route it through local reconstruction or sanitization.
                replacement
            }
            CompactionReplacement::LocalSummary(generate_session_compact) => {
                let raw_compacted = build_compacted_history(CompactedHistoryInput {
                    system_message: system_message.clone(),
                    user_message_prefix: user_message_prefix.clone(),
                    agents_md_reminder: agents_md_reminder.clone(),
                    state_context: compaction_state_context,
                    compaction_summary: generate_session_compact.clone(),
                    system_reminder: system_reminder.clone(),
                    summary_before_recent: use_short_prompt,
                    transcript_hint: transcript_hint.clone(),
                    summary_count,
                });
                let sanitize_result = sanitize_compacted_history(raw_compacted);
                let compacted_history = if sanitize_result.stripped_tool_call_ids.is_empty() {
                    sanitize_result.items
                } else {
                    tracing::warn!(
                        session_id = %self.session_info.id,
                        stripped_count = sanitize_result.stripped_tool_call_ids.len(),
                        stripped_ids = ?sanitize_result.stripped_tool_call_ids,
                        "compaction: stripped orphaned ToolResults from compacted history"
                    );
                    sanitize_result.items
                };
                let remaining_violations = validate_compacted_history(&compacted_history);
                if remaining_violations.is_empty() {
                    compacted_history
                } else {
                    tracing::error!(
                        session_id = %self.session_info.id,
                        violation_count = remaining_violations.len(),
                        violation_ids = ?remaining_violations,
                        "compaction: sanitized history still has invalid ToolResults -- \
                         falling back to minimal compacted history (no recent_messages)"
                    );
                    build_compacted_history(CompactedHistoryInput {
                        system_message,
                        user_message_prefix,
                        agents_md_reminder,
                        state_context: &state_context.for_compaction(),
                        compaction_summary: generate_session_compact,
                        system_reminder,
                        summary_before_recent: use_short_prompt,
                        transcript_hint,
                        summary_count,
                    })
                }
            }
        };
        let prompt_index_at_compaction = self.chat_state_handle.get_prompt_index().await;
        let original_user_info = self
            .chat_state_handle
            .get_conversation_item_at(1)
            .await
            .and_then(|item| match item {
                ConversationItem::User(parts) => {
                    parts.content.into_iter().next().and_then(|p| match p {
                        xai_grok_sampling_types::ContentPart::Text { text } => {
                            Some(text.as_ref().to_owned())
                        }
                        _ => None,
                    })
                }
                _ => None,
            });
        let commit_kind = match strategy {
            CompactionStrategy::NativeCodex => CompactionCommitKind::NativeCodex,
            CompactionStrategy::LocalSummary => CompactionCommitKind::LocalSummary,
        };
        // Resolve the exact final replacement before the durable transaction so
        // checkpoint, cache, lost-ack verify, and live install share one history.
        // Native Codex never preserves an inherited fork prefix.
        let prefix_len = if strategy == CompactionStrategy::NativeCodex
            || self
                .compaction
                .prefix_released
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            0
        } else {
            self.startup_hints.inherited_prefix_len.unwrap_or(0)
        };
        let compacted_history = if prefix_len == 0 {
            compacted_history
        } else {
            self.resolve_forked_compacted_history(
                compacted_history,
                prefix_len,
                tokens_before,
                context_window,
            )
            .await
        };
        if cancel.is_cancelled() {
            return self.emit_compact_cancelled(auto_trigger).await;
        }
        self.persist_compaction_install(
            &compacted_history,
            prompt_index_at_compaction,
            auto_continue.clone(),
            original_user_info.clone(),
            commit_kind,
        )
        .await?;
        let new_len = compacted_history.len();
        self.chat_state_handle
            .install_persisted_compaction(compacted_history)
            .await
            .ok_or_else(|| {
                self.compaction.reconciliation_required.store(
                    true,
                    std::sync::atomic::Ordering::Release,
                );
                acp::Error::internal_error().data(
                    "compaction was persisted but could not be installed in live chat state; reload required",
                )
            })?;
        self.chat_state_handle
            .record_compaction_at(prompt_index_at_compaction);
        let post_replace_tokens = self.chat_state_handle.get_total_tokens().await;
        if xai_token_estimation::exceeds_threshold(
            post_replace_tokens,
            context_window,
            self.compaction.threshold_percent.get(),
        ) {
            self.compaction
                .auto_compact_suppressed
                .store(SUPPRESS_STICKY, std::sync::atomic::Ordering::Relaxed);
            tracing::warn!(
                session_id = %self.session_info.id.0,
                post_replace_tokens,
                context_window,
                inherited_prefix = self.startup_hints.inherited_prefix_len.is_some(),
                "compaction: replacement history still over threshold; suppressing AUTO to avoid an immediate re-loop"
            );
        } else {
            self.compaction
                .auto_compact_suppressed
                .store(SUPPRESS_NONE, std::sync::atomic::Ordering::Relaxed);
            self.clear_auto_compact_retry_gate();
        }
        self.last_idle_flush_conversation_len
            .store(new_len, std::sync::atomic::Ordering::Relaxed);
        self.memory
            .context_injected
            .store(false, std::sync::atomic::Ordering::Relaxed);
        if self.memory.is_enabled() {
            tracing::info!(target: xai_grok_telemetry::memory_log::TARGET, "MEMORY_COMPACT: post-compaction reset, next turn re-checks injection (search only if no block persisted)");
        }
        let _ = self
            .notifications
            .persistence_tx
            .send(PersistenceMsg::PlanState(
                crate::tools::todo::TodoState::default(),
            ));
        self.agent
            .borrow()
            .tool_bridge()
            .on_agents_md_compaction()
            .await;
        self.agent
            .borrow()
            .tool_bridge()
            .on_skill_discovery_compaction()
            .await;
        self.persist_announcement_state().await;
        self.plan_mode.lock().reset_after_compaction();
        self.persist_plan_mode_state();
        self.dispatch_hook(
            xai_grok_hooks::event::HookEventName::PostCompact,
            xai_grok_hooks::event::HookPayload::PostCompact {
                source: compact_source.into(),
            },
            None,
            None,
        )
        .await;
        let tokens_after = self.chat_state_handle.get_total_tokens().await;
        {
            let span = tracing::Span::current();
            span.record("compaction_tokens_after", tokens_after as i64);
            span.record(
                "compaction_summary_chars",
                compact_output.content.chars().count() as i64,
            );
            span.record("compaction_attempts", total_attempts as i64);
            span.record(
                "compaction_degenerate_rejections",
                telemetry.degenerate_rejections as i64,
            );
            span.record(
                "compaction_input_overflow_rejections",
                input_overflow_rejections as i64,
            );
            span.record(
                "compaction_deterministic_rejections",
                telemetry.deterministic_rejections as i64,
            );
            span.record(
                "compaction_transient_rejections",
                total_transient_rejections as i64,
            );
            let stop_reason = compact_output.stop_reason.as_deref().unwrap_or("stop");
            span.record("compaction_stop_reason", stop_reason);
            let outcome = if compact_output.truncated {
                CompactionOutcome::Truncated
            } else {
                CompactionOutcome::Success
            };
            span.record("compaction_outcome", outcome.as_str());
            span.record("compaction_delta_count", compact_output.delta_count as i64);
            if let Some(ms) = compact_output.ttft_ms {
                span.record("compaction_ttft_ms", ms as i64);
            }
            if let Some(ms) = compact_output.stream_ms {
                span.record("compaction_stream_ms", ms as i64);
            }
            if let Some(ms) = compact_output.itl_max_ms {
                span.record("compaction_itl_max_ms", ms as i64);
            }
        }
        compaction.complete(tokens_after);
        Ok(())
    }
    /// Check if auto-compact should be triggered based on context window usage.
    /// Returns Some(AutoCompactTriggerInfo) if threshold is reached, None otherwise.
    pub(crate) fn should_auto_compact(
        &self,
        total_tokens: u64,
        context_window: std::num::NonZeroU64,
    ) -> Option<AutoCompactTriggerInfo> {
        let cw = context_window.get();
        if xai_token_estimation::exceeds_threshold(
            total_tokens,
            cw,
            self.compaction.threshold_percent.get(),
        ) {
            let percentage = xai_token_estimation::usage_percentage_u8(total_tokens, cw);
            Some(AutoCompactTriggerInfo {
                tokens_used: total_tokens,
                context_window: cw,
                percentage,
                kind: AutoCompactTriggerKind::SoftThreshold,
            })
        } else {
            None
        }
    }

    /// True when `total_tokens` is at/above Codex's non-suppressible safety limit.
    async fn codex_safety_limit_exceeded(&self, total_tokens: u64) -> bool {
        let Some(cfg) = self.chat_state_handle.get_sampling_config().await else {
            return false;
        };
        let configured_limit = self
            .compaction_at_tokens
            .get()
            .and_then(|limit| limit.resolve(cfg.context_window.get(), 90));
        let safety = xai_grok_sampling_types::resolve_provider(
            cfg.provider_id,
            cfg.api_backend.clone(),
            &cfg.base_url,
        )
        .capabilities()
        .auto_compact_safety();
        crate::session::compaction_config::resolve_codex_auto_compact_token_limit(
            safety,
            cfg.context_window,
            configured_limit,
        )
        .is_some_and(|limit| total_tokens >= limit.get())
    }

    /// Post-install headroom check keyed to the trigger that fired.
    ///
    /// - Codex safety compact: only the Codex provider budget must clear.
    ///   Soft threshold may still be over; sticky suppress already covers that.
    /// - Soft / forced / preflight: soft threshold must clear, and on Codex
    ///   backends the provider safety limit must clear too (otherwise the next
    ///   loop iteration would immediately re-enter via the Codex gate).
    async fn auto_compaction_would_immediately_retrigger(
        &self,
        total_tokens: u64,
        kind: AutoCompactTriggerKind,
    ) -> bool {
        let Some(cfg) = self.chat_state_handle.get_sampling_config().await else {
            return false;
        };
        match kind {
            AutoCompactTriggerKind::CodexSafety => {
                self.codex_safety_limit_exceeded(total_tokens).await
            }
            AutoCompactTriggerKind::SoftThreshold
            | AutoCompactTriggerKind::Forced
            | AutoCompactTriggerKind::PreflightOverflow => {
                if self
                    .should_auto_compact(total_tokens, cfg.context_window)
                    .is_some()
                {
                    return true;
                }
                self.codex_safety_limit_exceeded(total_tokens).await
            }
        }
    }

    /// Returns true if the error response indicates tokens exceed the
    /// model's context window. Inspects only the model-metadata
    /// portion of the [`SamplingErrorInfo`] (the `context_window`
    /// field) against the session's tracked token estimate.
    ///
    /// Called from `handle_sampling_failure` with the
    /// `SamplingErrorInfo` the sampler hands back.
    pub(crate) async fn should_compact_on_error(
        &self,
        err: &xai_grok_sampler::SamplingErrorInfo,
    ) -> bool {
        // Hard recovery remains active during bounded FINALIZING so a second
        // cycle hits claim_bounded and fail-closes instead of sampling oversize.
        if self
            .compaction
            .auto_compact_suppressed
            .load(std::sync::atomic::Ordering::Relaxed)
            != SUPPRESS_NONE
        {
            return false;
        }
        let Some(ref metadata) = err.model_metadata else {
            return false;
        };
        let Some(context_window) = metadata.context_window else {
            return false;
        };
        if context_window == 0 {
            return false;
        }
        let estimated_total = self.chat_state_handle.get_estimated_total_tokens().await;
        estimated_total > context_window
    }
    /// Non-suppressible Codex provider safety gate. Unlike the configurable
    /// soft threshold, this remains active for budgeted tasks and after a
    /// previous compaction failure, so a tool loop cannot keep sampling past
    /// Codex's effective input budget.
    pub(crate) async fn check_codex_auto_compact_needed(&self) -> Option<AutoCompactTriggerInfo> {
        // Reminder-token reserve already prevents a finalize-reminder-only re-arm.
        // Keep this hard provider gate live during FINALIZING so post-compact
        // refill hits claim_bounded and fail-closes before another model call.
        let sampling_cfg = self.chat_state_handle.get_sampling_config().await?;
        let provider = xai_grok_sampling_types::resolve_provider(
            sampling_cfg.provider_id,
            sampling_cfg.api_backend.clone(),
            &sampling_cfg.base_url,
        );
        if !provider.capabilities().enforces_auto_compact_safety() {
            return None;
        }
        let context_window = sampling_cfg.context_window;
        let configured_limit = self
            .compaction_at_tokens
            .get()
            .and_then(|limit| limit.resolve(context_window.get(), 90));
        let limit = crate::session::compaction_config::resolve_codex_auto_compact_token_limit(
            provider.capabilities().auto_compact_safety(),
            context_window,
            configured_limit,
        )?
        .get();
        let estimated_total = self.chat_state_handle.get_estimated_total_tokens().await;
        if estimated_total < limit {
            return None;
        }
        let context_window = context_window.get();
        let percentage = xai_token_estimation::usage_percentage_u8(estimated_total, context_window);
        tracing::warn!(
            model = %sampling_cfg.model,
            estimated_total,
            codex_auto_compact_token_limit = limit,
            context_window,
            "Codex auto-compaction safety limit reached"
        );
        Some(AutoCompactTriggerInfo {
            tokens_used: estimated_total,
            context_window,
            percentage,
            kind: AutoCompactTriggerKind::CodexSafety,
        })
    }

    /// Pre-sampling compaction check. Uses `get_estimated_total_tokens()`
    /// (exact prior count + byte-estimate of items since last response) so
    /// tool results are accounted for. Returns `None` when `is_flushing`.
    pub(crate) async fn check_auto_compact_needed(&self) -> Option<AutoCompactTriggerInfo> {
        // Soft threshold stays quiet during bounded FINALIZING; hard gates
        // (Codex safety / preflight / on-error) remain active so refill cannot
        // sample past a hard budget.
        if self.bounded_subagent_must_finalize() {
            return None;
        }
        if self
            .memory
            .is_flushing
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return None;
        }
        let sampling_cfg = self.chat_state_handle.get_sampling_config().await;
        let context_window = sampling_cfg.as_ref().map(|c| c.context_window)?;
        let cw = context_window.get();
        let model = sampling_cfg
            .as_ref()
            .map(|c| c.model.clone())
            .unwrap_or_default();
        let estimated_total = self.chat_state_handle.get_estimated_total_tokens().await;
        self.signals_handle()
            .update_context_usage(estimated_total, cw);
        if self
            .compaction
            .auto_compact_suppressed
            .load(std::sync::atomic::Ordering::Relaxed)
            != SUPPRESS_NONE
        {
            return None;
        }
        if self
            .compaction
            .force_compact
            .compare_exchange(
                true,
                false,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_ok()
        {
            let percentage = xai_token_estimation::usage_percentage_u8(estimated_total, cw);
            tracing::info!(
                "Forced auto-compact trigger (debug): model={model}, \
                 {percentage}% full ({estimated_total}/{cw} tokens)",
            );
            return Some(AutoCompactTriggerInfo {
                tokens_used: estimated_total,
                context_window: cw,
                percentage,
                kind: AutoCompactTriggerKind::Forced,
            });
        }
        if let Some(trigger_info) = self.should_auto_compact(estimated_total, context_window) {
            tracing::info!(
                "Pre-sampling auto-compact trigger: model={model}, \
                 {}% full ({}/{} tokens)",
                trigger_info.percentage,
                trigger_info.tokens_used,
                trigger_info.context_window,
            );
            return Some(trigger_info);
        }
        None
    }
    /// Returns `Some` when tool call outputs have pushed the estimated token
    /// count past the context window, indicating pre-emptive compaction is needed.
    pub(crate) async fn check_preflight_overflow(&self) -> Option<AutoCompactTriggerInfo> {
        // Hard context-window gate stays live during FINALIZING (unlike soft).
        if self
            .compaction
            .auto_compact_suppressed
            .load(std::sync::atomic::Ordering::Relaxed)
            != SUPPRESS_NONE
        {
            return None;
        }
        let estimated_total = self.chat_state_handle.get_estimated_total_tokens().await;
        let cfg = self.chat_state_handle.get_sampling_config().await?;
        let cw = cfg.context_window.get();
        if estimated_total <= cw {
            return None;
        }
        let overflow = estimated_total.saturating_sub(cw);
        let percentage = xai_token_estimation::usage_percentage_u8(estimated_total, cw);
        tracing::warn!(
            estimated_total,
            context_window = cw,
            overflow,
            model = %cfg.model,
            "CONTEXT_OVERFLOW_PREFLIGHT: estimated tokens exceed context window \
             after tool call outputs"
        );
        Some(AutoCompactTriggerInfo {
            tokens_used: estimated_total,
            context_window: cw,
            percentage,
            kind: AutoCompactTriggerKind::PreflightOverflow,
        })
    }
    /// On a model or context-window change: clear sticky/other suppress and
    /// compact if the window shrank. Model slug plus context window form the
    /// transition identity because remote metadata can resize the same model.
    /// Leaves credit/auth suppress (a model/context change can't fix those) and
    /// short-circuits.
    /// Auth compact failures abort the turn (same as pre-sampling/preflight).
    pub(crate) async fn maybe_compact_on_model_switch(self: &Arc<Self>) -> Result<(), acp::Error> {
        self.refresh_token_if_expired().await;
        let Some(prev) = self.compaction.previous_model.take() else {
            return Ok(());
        };
        let Some(cfg) = self.chat_state_handle.get_sampling_config().await else {
            return Ok(());
        };
        let context_window = cfg.context_window.get();
        if cfg.model == prev.model_slug && context_window == prev.context_window {
            return Ok(());
        }
        if self.is_account_state_suppressed() {
            return Ok(());
        }
        self.compaction
            .auto_compact_suppressed
            .store(SUPPRESS_NONE, std::sync::atomic::Ordering::Relaxed);
        self.clear_auto_compact_retry_gate();
        if prev.context_window <= context_window {
            return Ok(());
        }
        let total_tokens = self.chat_state_handle.get_estimated_total_tokens().await;
        let Some(trigger_info) = self.should_auto_compact(total_tokens, cfg.context_window) else {
            return Ok(());
        };
        tracing::info!(
            previous_model = %prev.model_slug,
            previous_context_window = prev.context_window,
            model = %cfg.model,
            context_window,
            percentage = trigger_info.percentage,
            "Proactive model/context-change compact after context-window shrink",
        );
        if let Err(e) = self.run_compact_only(trigger_info).await {
            tracing::error!(error = %e, "Model/context-change compaction failed");
            if Self::is_auth_compact_error(&e) {
                return Err(self.surface_compact_auth_failure(e).await);
            }
            if self.is_bounded_subagent_task() {
                return Err(e);
            }
        }
        Ok(())
    }
    /// Record the current model and context budget for transition detection on
    /// the next turn.
    pub(crate) async fn record_turn_model(&self) {
        if let Some(cfg) = self.chat_state_handle.get_sampling_config().await {
            self.compaction.previous_model.set(Some(
                crate::session::compaction_config::PreviousModelInfo {
                    model_slug: cfg.model.clone(),
                    context_window: cfg.context_window.get(),
                },
            ));
        }
    }
    /// Compact without auto-continue. The outer turn loop rebuilds and retries.
    /// Emits telemetry (`auto_compact_fired`) and UI notifications automatically.
    #[tracing::instrument(
        name = "session.compact",
        skip_all,
        fields(
            session_id = %self.session_info.id.0,
            trigger = "auto",
            mode = tracing::field::Empty,
            detail = tracing::field::Empty,
            pre_tokens = tracing::field::Empty,
            post_tokens = tracing::field::Empty,
            success = tracing::field::Empty,
            error = tracing::field::Empty,
        )
    )]
    pub(crate) async fn run_compact_only(
        self: &Arc<Self>,
        trigger_info: AutoCompactTriggerInfo,
    ) -> Result<(), acp::Error> {
        use crate::extensions::notification::SessionUpdate as XaiSessionUpdate;
        let (_cancel, _cancel_scope) = self.compaction.cancel.enter();
        let strategy_override = configured_codex_compaction_strategy();
        if let Some(error) = self.auto_compact_retry_gate_error() {
            tracing::warn!(
                trigger = ?trigger_info.kind,
                error = %error,
                "automatic compaction retry gate prevented an immediate repeated lifecycle"
            );
            return Err(error);
        }
        let bounded_claimed = self.claim_bounded_auto_compaction()?;
        self.record_compaction_variant();
        let tokens_before = self.chat_state_handle.get_total_tokens().await;
        tracing::Span::current().record("pre_tokens", tokens_before as i64);
        xai_grok_telemetry::session_ctx::log_event(xai_grok_telemetry::events::AutoCompactFired {
            tokens_before: trigger_info.tokens_used,
            percentage: trigger_info.percentage,
        });
        self.signals_handle()
            .record_compaction(trigger_info.tokens_used);
        self.send_xai_notification(XaiSessionUpdate::AutoCompactStarted {
            tokens_used: trigger_info.tokens_used,
            context_window: trigger_info.context_window,
            percentage: trigger_info.percentage,
            reason: format!("Context window {}% full", trigger_info.percentage),
        })
        .await;
        self.maybe_pre_compaction_flush(
            trigger_info.tokens_used,
            trigger_info.context_window,
            "pre_compact_on_error",
        )
        .await;
        let compact_start = std::time::Instant::now();
        let result = self
            .run_compact_inner(
                None,
                None,
                xai_grok_telemetry::events::CompactionTrigger::Auto,
                strategy_override.as_override(),
            )
            .await;
        let elapsed_ms = compact_start.elapsed().as_millis() as i64;
        match result {
            Ok(()) => {
                let tokens_after = self.chat_state_handle.get_total_tokens().await;
                let span = tracing::Span::current();
                span.record("post_tokens", tokens_after as i64);
                // Bounded tasks push a finalize reminder after success; reserve
                // those tokens so headroom reflects the history the next sample
                // will actually see.
                let headroom_tokens = if bounded_claimed {
                    tokens_after.saturating_add(bounded_finalize_reminder_token_reserve())
                } else {
                    tokens_after
                };
                if self
                    .auto_compaction_would_immediately_retrigger(headroom_tokens, trigger_info.kind)
                    .await
                {
                    let hard_context_exceeded = headroom_tokens >= trigger_info.context_window;
                    let codex_still_over = self.codex_safety_limit_exceeded(headroom_tokens).await;
                    // Soft-only overshoot (under the hard context window, and
                    // under any provider safety limit) is sticky-suppressed by
                    // `run_compact_inner`. Treat that as usable installed
                    // history so callers do not see failure after a durable
                    // install. History still at/above the hard window — or
                    // still over Codex safety — must fail terminally.
                    if !hard_context_exceeded && !codex_still_over {
                        tracing::warn!(
                            session_id = %self.session_info.id.0,
                            tokens_after,
                            headroom_tokens,
                            context_window = trigger_info.context_window,
                            trigger = ?trigger_info.kind,
                            "automatic compaction installed history still over soft threshold; continuing with sticky suppress"
                        );
                        // Reminder-token reserve can tip soft headroom even when
                        // `run_compact_inner` saw post-install tokens under soft.
                        self.compaction
                            .auto_compact_suppressed
                            .store(SUPPRESS_STICKY, std::sync::atomic::Ordering::Relaxed);
                        if bounded_claimed {
                            self.finish_bounded_auto_compaction(true);
                            self.push_system_reminder(BOUNDED_SUBAGENT_FINALIZE_REMINDER);
                        }
                        span.record("success", true);
                        self.send_xai_notification(XaiSessionUpdate::AutoCompactCompleted {
                            tokens_before: Some(trigger_info.tokens_used),
                            tokens_after,
                            elapsed_ms: Some(elapsed_ms),
                            summary_preview: None,
                        })
                        .await;
                        return Ok(());
                    }
                    let message = format!(
                        "automatic compaction installed replacement history with insufficient headroom ({tokens_after}/{} tokens); stopping to avoid an immediate compaction loop",
                        trigger_info.context_window
                    );
                    span.record("success", false);
                    span.record("error", message.as_str());
                    if bounded_claimed {
                        self.finish_bounded_auto_compaction(false);
                    }
                    self.record_auto_compact_retry_gate(
                        AutoCompactFailureDisposition::RetryAfterReset,
                    );
                    self.send_xai_notification(XaiSessionUpdate::AutoCompactFailed {
                        error: message.clone(),
                    })
                    .await;
                    return Err(acp::Error::internal_error().data(message));
                }
                if bounded_claimed {
                    self.finish_bounded_auto_compaction(true);
                    self.push_system_reminder(BOUNDED_SUBAGENT_FINALIZE_REMINDER);
                }
                self.clear_auto_compact_retry_gate();
                span.record("success", true);
                self.send_xai_notification(XaiSessionUpdate::AutoCompactCompleted {
                    tokens_before: Some(trigger_info.tokens_used),
                    tokens_after,
                    elapsed_ms: Some(elapsed_ms),
                    summary_preview: None,
                })
                .await;
                Ok(())
            }
            Err(e) => {
                if bounded_claimed {
                    self.finish_bounded_auto_compaction(false);
                }
                let span = tracing::Span::current();
                span.record("success", false);
                span.record("error", e.to_string().as_str());
                let cancelled = self.compaction.cancel.is_cancelled()
                    || e.data.as_ref().and_then(|d| d.as_str()).is_some_and(|s| {
                        s.contains(crate::session::helpers::session_compact::COMPACT_CANCELLED_MSG)
                    });
                if !cancelled {
                    self.record_auto_compact_retry_gate(Self::auto_compact_failure_disposition(&e));
                }
                if !cancelled
                    && self
                        .compaction
                        .auto_compact_suppressed
                        .load(std::sync::atomic::Ordering::Relaxed)
                        == SUPPRESS_NONE
                {
                    let error = Self::acp_error_message(&e);
                    self.send_xai_notification(XaiSessionUpdate::AutoCompactFailed { error })
                        .await;
                }
                Err(e)
            }
        }
    }
    /// Persist a compaction request artifact for offline prompt iteration.
    ///
    /// Writes `{session_dir}/compaction_requests/{request_id}.json` containing
    /// the exact ConversationItem list sent to the compaction model plus the
    /// summary (or final error) it produced. The file rides on
    /// the post-turn session archive to cloud storage via the existing per-turn upload
    /// pipeline — no separate upload path is needed.
    ///
    /// `created_at` is taken from the caller-supplied `started_at` (captured
    /// before the retry loop) rather than `Utc::now()` here, so transient
    /// retries don't skew the timestamp away from when the call actually
    /// started.
    ///
    /// Best-effort: send-failures are logged at `warn` and never surfaced to
    /// the user, because the artifact is purely for offline analysis.
    #[allow(clippy::too_many_arguments)]
    fn persist_compaction_request_artifact(
        &self,
        chat_history: Vec<ConversationItem>,
        tools: Vec<xai_grok_sampling_types::ToolSpec>,
        user_context: Option<&str>,
        use_short_prompt: bool,
        model: &str,
        trigger: xai_grok_telemetry::events::CompactionTrigger,
        summary: Option<&str>,
        error: Option<&acp::Error>,
        attempts: u32,
        attempt_details: Vec<CompactionAttempt>,
        started_at: String,
    ) {
        use crate::extensions::notification::CompactionRequestFile;
        let request_id = uuid::Uuid::new_v4().to_string();
        let trigger_str = match trigger {
            xai_grok_telemetry::events::CompactionTrigger::Manual => "manual",
            xai_grok_telemetry::events::CompactionTrigger::Auto => "auto",
        };
        let prompt_variant = if use_short_prompt {
            "short"
        } else {
            "detailed"
        };
        let error_str = error.map(|e| {
            e.data
                .as_ref()
                .and_then(|d| d.as_str())
                .unwrap_or("<no error data>")
                .to_owned()
        });
        let artifact = CompactionRequestFile {
            schema_version: 2,
            request_id,
            created_at: started_at,
            trigger: trigger_str.to_owned(),
            prompt_variant: prompt_variant.to_owned(),
            model: model.to_owned(),
            user_context: user_context.map(str::to_owned),
            chat_history,
            tools,
            summary: summary.map(str::to_owned),
            error: error_str,
            attempts,
            attempt_details,
        };
        if self
            .notifications
            .persistence_tx
            .send(PersistenceMsg::CompactionRequest(artifact))
            .is_err()
        {
            tracing::warn!(
                session_id = %self.session_info.id.0,
                "Failed to send compaction request artifact to persistence channel"
            );
        }
    }
    /// Persist compaction via the shared marker→ack→install protocol used by
    /// rewind and both local/native strategies. Live chat is not mutated here;
    /// the caller installs only after this returns successfully.
    async fn persist_compaction_install(
        &self,
        compacted_history: &[ConversationItem],
        prompt_index_at_compaction: usize,
        auto_continue: Option<crate::extensions::notification::AutoContinueInfo>,
        original_user_info: Option<String>,
        commit_kind: CompactionCommitKind,
    ) -> Result<(), acp::Error> {
        use crate::extensions::notification::{
            CompactionCheckpointFile, CompactionCheckpointInfo,
            FINALIZED_COMPACTION_CHECKPOINT_SCHEMA_VERSION, SessionNotification,
            SessionUpdate as XaiSessionUpdate,
        };
        use crate::session::helpers::timeline_transaction::{
            await_timeline_transaction, ensure_timeline_committed,
        };
        let op_name = match commit_kind {
            CompactionCommitKind::NativeCodex => "native compaction",
            CompactionCommitKind::LocalSummary => "local compaction",
        };
        let checkpoint_id = uuid::Uuid::new_v4().to_string();
        let checkpoint_file = format!("compaction_checkpoints/{checkpoint_id}.json");
        let created_at = chrono::Utc::now().to_rfc3339();
        let checkpoint = CompactionCheckpointFile {
            checkpoint_id: checkpoint_id.clone(),
            prompt_index_at_compaction,
            compacted_history: compacted_history.to_vec(),
            schema_version: FINALIZED_COMPACTION_CHECKPOINT_SCHEMA_VERSION,
            created_at: created_at.clone(),
            original_user_info,
            reread_file_paths: vec![],
        };
        let marker = crate::session::storage::SessionUpdate::Xai(Box::new(SessionNotification {
            session_id: self.session_info.id.clone(),
            update: XaiSessionUpdate::CompactionCheckpoint(Box::new(CompactionCheckpointInfo {
                checkpoint_id,
                prompt_index_at_compaction,
                checkpoint_file,
                auto_continue,
                schema_version: FINALIZED_COMPACTION_CHECKPOINT_SCHEMA_VERSION,
                created_at,
            })),
            meta: Some(self.build_notification_meta()),
        }));
        let verification_checkpoint = checkpoint.clone();
        let (respond_to, response) = tokio::sync::oneshot::channel();
        self.notifications
            .persistence_tx
            .send(PersistenceMsg::InstallCompactionAndAck {
                checkpoint,
                marker,
                replacement: compacted_history.to_vec(),
                respond_to,
            })
            .map_err(|_| {
                acp::Error::internal_error().data(format!(
                    "{op_name} persistence channel closed before install; original history remains live"
                ))
            })?;
        let session_dir = crate::session::persistence::session_dir(&self.session_info);
        let outcome = await_timeline_transaction(response, op_name, move || {
            crate::session::helpers::replay::verify_compaction_commit(
                &session_dir,
                &verification_checkpoint,
                commit_kind,
            )
        })
        .await
        .map_err(|error| {
            if error.requires_reconciliation() {
                self.compaction
                    .reconciliation_required
                    .store(true, std::sync::atomic::Ordering::Release);
            }
            acp::Error::internal_error().data(error.message().to_owned())
        })?;
        let cache_status =
            ensure_timeline_committed(outcome, op_name, self.session_info.id.0.as_ref())
                .map_err(|message| acp::Error::internal_error().data(message))?;
        if matches!(
            cache_status,
            crate::session::persistence::TimelineCacheStatus::RepairRequired(_)
        ) {
            self.compaction
                .reconciliation_required
                .store(true, std::sync::atomic::Ordering::Release);
            return Err(acp::Error::internal_error().data(format!(
                "{op_name} committed but the chat cache was not replaced; reload the session before continuing"
            )));
        }
        tracing::info!(
            prompt_index_at_compaction,
            ?commit_kind,
            "Persisted compaction checkpoint via durable transaction"
        );
        Ok(())
    }
}
#[cfg(test)]
#[path = "compaction_inline_auto_compact_flow_tests.rs"]
mod inline_auto_compact_flow_tests;
