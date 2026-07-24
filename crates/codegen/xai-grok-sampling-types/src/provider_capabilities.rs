//! Provider capability seam for Responses/compaction/auth behavior.
//!
//! Call sites should prefer [`capabilities_for_base_url`] over sniffing URLs
//! directly. Codex remains one adapter behind [`crate::is_codex_backend_url`];
//! future providers can return a different capability set without teaching
//! every harness module about their URLs.

use crate::is_codex_backend_url;

/// How a backend treats hosted / extra tools on CreateResponse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostedToolPolicy {
    /// Grok-style: extra hosted tools may be attached by the client.
    AllowExtra,
    /// Codex-style: reject unknown hosted tools; omit client extras.
    RejectUnknown,
}

/// Behavioral flags for a sampling backend URL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderCapabilities {
    /// May call a native `/responses/compact` replacement endpoint.
    pub native_compact: bool,
    /// Capture/replay sticky turn routing state (`x-codex-turn-state`).
    pub sticky_turn_state: bool,
    /// Preserve exact Responses output order via item ids + manifests.
    pub preserve_output_order: bool,
    /// Preserve ordinary response metadata passthrough on SSE/unary decode.
    pub preserve_response_metadata: bool,
    /// Hosted-tool attachment policy for CreateResponse.
    pub hosted_tool_policy: HostedToolPolicy,
    /// Apply the provider's non-suppressible auto-compact safety limit.
    pub provider_auto_compact_safety: bool,
    /// Skip Grok-only compaction remaining / at-tokens response headers.
    pub skip_grok_compaction_headers: bool,
    /// Use ChatGPT auth.json credentials and account headers.
    pub chatgpt_auth: bool,
    /// Value for ACP model meta `providerKind`, when set.
    pub provider_kind_meta: Option<&'static str>,
    /// Clear xAI-only `api_base_url` overrides in the model catalog.
    pub clear_xai_api_base_url: bool,
    /// Normalize CreateResponse bodies for this backend before send.
    pub normalize_create_response: bool,
}

const DEFAULT_CAPS: ProviderCapabilities = ProviderCapabilities {
    native_compact: false,
    sticky_turn_state: false,
    preserve_output_order: false,
    preserve_response_metadata: false,
    hosted_tool_policy: HostedToolPolicy::AllowExtra,
    provider_auto_compact_safety: false,
    skip_grok_compaction_headers: false,
    chatgpt_auth: false,
    provider_kind_meta: None,
    clear_xai_api_base_url: false,
    normalize_create_response: false,
};

const CODEX_CAPS: ProviderCapabilities = ProviderCapabilities {
    native_compact: true,
    sticky_turn_state: true,
    preserve_output_order: true,
    preserve_response_metadata: true,
    hosted_tool_policy: HostedToolPolicy::RejectUnknown,
    provider_auto_compact_safety: true,
    skip_grok_compaction_headers: true,
    chatgpt_auth: true,
    provider_kind_meta: Some("codex"),
    clear_xai_api_base_url: true,
    normalize_create_response: true,
};

/// Resolve capability flags for a sampling `base_url`.
pub fn capabilities_for_base_url(base_url: &str) -> ProviderCapabilities {
    if is_codex_backend_url(base_url) {
        CODEX_CAPS
    } else {
        DEFAULT_CAPS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CODEX_BACKEND_BASE_URL;

    #[test]
    fn codex_url_gets_codex_capabilities() {
        let caps = capabilities_for_base_url(CODEX_BACKEND_BASE_URL);
        assert!(caps.native_compact);
        assert!(caps.preserve_output_order);
        assert_eq!(caps.provider_kind_meta, Some("codex"));
        assert_eq!(caps.hosted_tool_policy, HostedToolPolicy::RejectUnknown);
    }

    #[test]
    fn grok_url_gets_default_capabilities() {
        let caps = capabilities_for_base_url("https://api.x.ai/v1");
        assert!(!caps.native_compact);
        assert!(!caps.chatgpt_auth);
        assert_eq!(caps.hosted_tool_policy, HostedToolPolicy::AllowExtra);
        assert!(caps.provider_kind_meta.is_none());
    }
}
