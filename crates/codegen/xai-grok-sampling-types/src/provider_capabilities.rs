//! Resolved sampling-provider identity and cross-layer provider policy.
//!
//! Provider identity is explicit when supplied by the model catalog and falls
//! back to URL recognition for legacy configurations. Capabilities are derived
//! from that identity; they are never used to infer it.

use crate::{ApiBackend, is_codex_backend_url};

/// Logical provider selected by model/catalog configuration.
///
/// `base_url` is a transport destination and may be a proxy. It is therefore
/// deliberately not part of this enum. `None` at configuration boundaries
/// means "legacy auto-detection", not an additional provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderId {
    Xai,
    Codex,
    OpenAiCompatible,
    Custom,
}

/// Provider plus wire protocol, resolved independently of transport location.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolIdentity {
    provider_id: Option<ProviderId>,
    api_backend: ApiBackend,
}

impl ProtocolIdentity {
    /// Resolve explicit provider identity first, retaining URL inference only
    /// for configurations and snapshots written before `provider_id` existed.
    pub fn resolve(
        provider_id: Option<ProviderId>,
        api_backend: ApiBackend,
        base_url: &str,
    ) -> Self {
        let provider_id =
            provider_id.or_else(|| is_codex_backend_url(base_url).then_some(ProviderId::Codex));
        Self {
            provider_id,
            api_backend,
        }
    }

    pub fn provider_id(&self) -> Option<ProviderId> {
        self.provider_id
    }

    pub fn api_backend(&self) -> &ApiBackend {
        &self.api_backend
    }

    pub fn is_codex_responses(&self) -> bool {
        self.provider_id == Some(ProviderId::Codex) && self.api_backend == ApiBackend::Responses
    }
}

/// A provider resolved once from model identity and transport configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedProvider {
    protocol: ProtocolIdentity,
    capabilities: ProviderCapabilities,
}

impl ResolvedProvider {
    pub fn resolve(
        provider_id: Option<ProviderId>,
        api_backend: ApiBackend,
        base_url: &str,
    ) -> Self {
        let protocol = ProtocolIdentity::resolve(provider_id, api_backend, base_url);
        let capabilities = capabilities_for_protocol(&protocol);
        Self {
            protocol,
            capabilities,
        }
    }

    pub fn protocol(&self) -> &ProtocolIdentity {
        &self.protocol
    }

    pub fn provider_id(&self) -> Option<ProviderId> {
        self.protocol.provider_id()
    }

    pub fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities
    }
}

/// How a backend treats hosted / extra tools on CreateResponse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostedToolPolicy {
    /// Grok-style: extra hosted tools may be attached by the client.
    AllowExtra,
    /// Codex-style: reject unknown hosted tools; omit client extras.
    RejectUnknown,
}

/// Process-local routing behavior for one prompt turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnRoutingPolicy {
    None,
    FirstValueWinsHeader(&'static str),
}

/// Provider-specific automatic-compaction safety policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoCompactSafety {
    None,
    MaxContextFraction { numerator: u32, denominator: u32 },
}

/// Native replacement-history implementation selected for this protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeCompactionKind {
    None,
    Codex,
}

/// Responses request/metadata dialect selected from explicit protocol identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponsesWireProtocol {
    Standard,
    Codex,
}

/// Cross-layer policy derived from one resolved protocol identity.
///
/// Fields are private so callers ask semantic questions and cannot construct
/// unsupported combinations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderCapabilities {
    native_compaction: NativeCompactionKind,
    turn_routing: TurnRoutingPolicy,
    responses_wire: ResponsesWireProtocol,
    preserve_response_metadata: bool,
    hosted_tool_policy: HostedToolPolicy,
    auto_compact_safety: AutoCompactSafety,
    skip_grok_compaction_headers: bool,
    chatgpt_auth: bool,
    provider_kind_meta: Option<&'static str>,
    clear_xai_api_base_url: bool,
}

const DEFAULT_CAPS: ProviderCapabilities = ProviderCapabilities {
    native_compaction: NativeCompactionKind::None,
    turn_routing: TurnRoutingPolicy::None,
    responses_wire: ResponsesWireProtocol::Standard,
    preserve_response_metadata: false,
    hosted_tool_policy: HostedToolPolicy::AllowExtra,
    auto_compact_safety: AutoCompactSafety::None,
    skip_grok_compaction_headers: false,
    chatgpt_auth: false,
    provider_kind_meta: None,
    clear_xai_api_base_url: false,
};

const CODEX_CAPS: ProviderCapabilities = ProviderCapabilities {
    native_compaction: NativeCompactionKind::Codex,
    turn_routing: TurnRoutingPolicy::FirstValueWinsHeader("x-codex-turn-state"),
    responses_wire: ResponsesWireProtocol::Codex,
    preserve_response_metadata: true,
    hosted_tool_policy: HostedToolPolicy::RejectUnknown,
    auto_compact_safety: AutoCompactSafety::MaxContextFraction {
        numerator: 9,
        denominator: 10,
    },
    skip_grok_compaction_headers: true,
    chatgpt_auth: true,
    provider_kind_meta: Some("codex"),
    clear_xai_api_base_url: true,
};

impl ProviderCapabilities {
    pub fn supports_native_compact(self) -> bool {
        self.native_compaction != NativeCompactionKind::None
    }

    pub fn native_compaction_kind(self) -> NativeCompactionKind {
        self.native_compaction
    }

    pub fn turn_routing_policy(self) -> TurnRoutingPolicy {
        self.turn_routing
    }

    pub fn uses_turn_routing(self) -> bool {
        self.turn_routing != TurnRoutingPolicy::None
    }

    pub fn sticky_turn_header(self) -> Option<&'static str> {
        match self.turn_routing {
            TurnRoutingPolicy::None => None,
            TurnRoutingPolicy::FirstValueWinsHeader(header) => Some(header),
        }
    }

    pub fn responses_wire_protocol(self) -> ResponsesWireProtocol {
        self.responses_wire
    }

    pub fn preserves_output_order(self) -> bool {
        self.responses_wire == ResponsesWireProtocol::Codex
    }

    pub fn preserves_response_metadata(self) -> bool {
        self.preserve_response_metadata
    }

    pub fn allows_extra_hosted_tools(self) -> bool {
        self.hosted_tool_policy == HostedToolPolicy::AllowExtra
    }

    pub fn hosted_tool_policy(self) -> HostedToolPolicy {
        self.hosted_tool_policy
    }

    pub fn auto_compact_safety(self) -> AutoCompactSafety {
        self.auto_compact_safety
    }

    pub fn enforces_auto_compact_safety(self) -> bool {
        self.auto_compact_safety != AutoCompactSafety::None
    }

    pub fn skips_grok_compaction_headers(self) -> bool {
        self.skip_grok_compaction_headers
    }

    pub fn uses_chatgpt_auth(self) -> bool {
        self.chatgpt_auth
    }

    pub fn provider_kind_meta(self) -> Option<&'static str> {
        self.provider_kind_meta
    }

    pub fn clears_xai_api_base_url(self) -> bool {
        self.clear_xai_api_base_url
    }

    pub fn normalizes_create_response(self) -> bool {
        self.responses_wire == ResponsesWireProtocol::Codex
    }

    pub fn replays_native_compaction(self) -> bool {
        self.native_compaction == NativeCompactionKind::Codex
    }
}

pub fn capabilities_for_protocol(protocol: &ProtocolIdentity) -> ProviderCapabilities {
    if protocol.is_codex_responses() {
        CODEX_CAPS
    } else {
        DEFAULT_CAPS
    }
}

/// Resolve identity and policy together. Explicit provider identity wins over
/// URL inference, allowing proxies and deterministic local test servers.
pub fn resolve_provider(
    provider_id: Option<ProviderId>,
    api_backend: ApiBackend,
    base_url: &str,
) -> ResolvedProvider {
    ResolvedProvider::resolve(provider_id, api_backend, base_url)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CODEX_BACKEND_BASE_URL;

    #[test]
    fn explicit_codex_identity_works_through_custom_transport() {
        let provider = resolve_provider(
            Some(ProviderId::Codex),
            ApiBackend::Responses,
            "http://127.0.0.1:3210/v1",
        );
        assert_eq!(provider.provider_id(), Some(ProviderId::Codex));
        assert!(provider.capabilities().supports_native_compact());
        assert_eq!(
            provider.capabilities().sticky_turn_header(),
            Some("x-codex-turn-state")
        );
    }

    #[test]
    fn legacy_codex_url_still_infers_codex() {
        let provider = resolve_provider(None, ApiBackend::Responses, CODEX_BACKEND_BASE_URL);
        assert_eq!(provider.provider_id(), Some(ProviderId::Codex));
        assert!(provider.capabilities().preserves_output_order());
    }

    #[test]
    fn explicit_non_codex_identity_overrides_codex_looking_url() {
        for provider_id in [
            ProviderId::Xai,
            ProviderId::OpenAiCompatible,
            ProviderId::Custom,
        ] {
            let provider = resolve_provider(
                Some(provider_id),
                ApiBackend::Responses,
                CODEX_BACKEND_BASE_URL,
            );
            assert_eq!(provider.provider_id(), Some(provider_id));
            assert!(!provider.capabilities().normalizes_create_response());
            assert!(!provider.capabilities().replays_native_compaction());
        }
    }

    #[test]
    fn codex_identity_requires_responses_protocol() {
        let provider = resolve_provider(
            Some(ProviderId::Codex),
            ApiBackend::ChatCompletions,
            CODEX_BACKEND_BASE_URL,
        );
        assert!(!provider.capabilities().normalizes_create_response());
        assert!(!provider.capabilities().uses_chatgpt_auth());
    }

    #[test]
    fn default_policy_preserves_extra_tools() {
        let caps = resolve_provider(
            Some(ProviderId::Xai),
            ApiBackend::Responses,
            "https://api.x.ai/v1",
        )
        .capabilities();
        assert!(caps.allows_extra_hosted_tools());
        assert!(caps.provider_kind_meta().is_none());
        assert_eq!(caps.auto_compact_safety(), AutoCompactSafety::None);
    }
}
