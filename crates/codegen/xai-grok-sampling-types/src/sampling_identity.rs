//! Runtime sampling identity and checked config/identity bindings.
//!
//! Identity is resolved from logical provider/API configuration independently
//! of the transport URL. Credentials and bearer tokens are never retained.

use serde::{Deserialize, Serialize};

use crate::{ApiBackend, ProtocolIdentity, SamplingConfig};

/// Complete sampling identity used to decide whether provider-owned history can
/// be replayed. Codex's supported public host spellings and explicit proxies
/// are canonicalized to one logical backend URL; other backends retain their
/// configured transport URL.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SamplingIdentity {
    pub api_backend: ApiBackend,
    pub backend_family: String,
    pub base_url: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chatgpt_account_id: Option<String>,
}

impl SamplingIdentity {
    pub const CODEX_RESPONSES_FAMILY: &'static str = "codex_responses";
    pub const OTHER_BACKEND_FAMILY: &'static str = "other";

    /// Compatibility constructor for persisted configurations without an
    /// explicit provider ID. New runtime code should use `new_for_protocol`.
    pub fn new(
        api_backend: ApiBackend,
        base_url: impl Into<String>,
        model: impl Into<String>,
        chatgpt_account_id: Option<String>,
    ) -> Self {
        let base_url = base_url.into();
        let protocol = ProtocolIdentity::resolve(None, api_backend, &base_url);
        Self::new_for_protocol(&protocol, base_url, model, chatgpt_account_id)
    }

    pub fn new_for_protocol(
        protocol: &ProtocolIdentity,
        transport_base_url: impl Into<String>,
        model: impl Into<String>,
        chatgpt_account_id: Option<String>,
    ) -> Self {
        let transport_base_url = transport_base_url.into();
        let (backend_family, base_url) = if protocol.is_codex_responses() {
            (
                Self::CODEX_RESPONSES_FAMILY.to_string(),
                crate::CODEX_BACKEND_BASE_URL.to_string(),
            )
        } else {
            (Self::OTHER_BACKEND_FAMILY.to_string(), transport_base_url)
        };
        Self {
            api_backend: protocol.api_backend().clone(),
            backend_family,
            base_url,
            model: model.into(),
            chatgpt_account_id,
        }
    }

    pub fn from_sampling_config(config: &SamplingConfig) -> Self {
        let protocol = ProtocolIdentity::resolve(
            config.provider_id,
            config.api_backend.clone(),
            &config.base_url,
        );
        Self::new_for_protocol(
            &protocol,
            config.base_url.clone(),
            config.model.clone(),
            chatgpt_account_id_from_headers(&config.extra_headers).map(str::to_owned),
        )
    }

    pub(crate) fn is_codex_responses(&self) -> bool {
        self.api_backend == ApiBackend::Responses
            && self.backend_family == Self::CODEX_RESPONSES_FAMILY
            && self.base_url == crate::CODEX_BACKEND_BASE_URL
    }
}

/// A checked, process-local binding between persisted routing configuration and
/// the effective runtime identity used to validate provider-owned history.
///
/// This value is deliberately not serializable: callers must resolve it again
/// from the final config and effective headers at each transition seam.
#[derive(Debug, Clone)]
pub struct ResolvedSamplingTarget {
    config: SamplingConfig,
    identity: SamplingIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingTargetMismatch;

impl std::fmt::Display for SamplingTargetMismatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(
            "runtime sampling identity does not match the proposed provider, API backend, logical backend route, or model",
        )
    }
}

impl std::error::Error for SamplingTargetMismatch {}

impl ResolvedSamplingTarget {
    pub fn new(
        config: SamplingConfig,
        identity: SamplingIdentity,
    ) -> Result<Self, SamplingTargetMismatch> {
        let protocol = ProtocolIdentity::resolve(
            config.provider_id,
            config.api_backend.clone(),
            &config.base_url,
        );
        let expected = SamplingIdentity::new_for_protocol(
            &protocol,
            config.base_url.clone(),
            config.model.clone(),
            identity.chatgpt_account_id.clone(),
        );
        if identity != expected {
            return Err(SamplingTargetMismatch);
        }
        Ok(Self { config, identity })
    }

    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    pub fn identity(&self) -> &SamplingIdentity {
        &self.identity
    }

    pub fn into_parts(self) -> (SamplingConfig, SamplingIdentity) {
        (self.config, self.identity)
    }
}

/// Extract the effective ChatGPT account from ordered config headers.
/// Header names are ASCII-case-insensitive and later entries win, matching
/// their eventual HTTP-header installation semantics.
pub fn chatgpt_account_id_from_headers(
    headers: &indexmap::IndexMap<String, String>,
) -> Option<&str> {
    headers
        .iter()
        .rev()
        .find(|(name, _)| name.eq_ignore_ascii_case(crate::CHATGPT_ACCOUNT_ID_HEADER))
        .map(|(_, value)| value.as_str())
}
