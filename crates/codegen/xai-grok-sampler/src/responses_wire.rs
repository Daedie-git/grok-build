//! Provider-specific Responses dialects behind one narrow seam.
//!
//! The sampling client owns HTTP/SSE transport and generic defaults. This
//! module owns the request shape, replay metadata, final normalization, and
//! sideband frames that vary across Responses wire protocols.

use reqwest::header::{HeaderMap, HeaderValue};

use xai_grok_sampling_types::{
    ConversationRequest, CreateResponseWrapper, ResponseMetadataOrigin, ResponsesWireProtocol,
    Result, SamplingError, TurnRoutingPolicy, TurnRoutingState,
};

/// A provider-neutral effect carried by a consumed Responses sideband frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ResponsesSideband {
    pub turn_routing_value: Option<String>,
}

/// Resolved Responses wire behavior for one sampling client.
///
/// Standard Responses has no sideband events. Provider adapters remain private
/// implementation details selected from explicit protocol identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResponsesWireAdapter {
    protocol: ResponsesWireProtocol,
    turn_routing: TurnRoutingPolicy,
}

impl ResponsesWireAdapter {
    pub(crate) fn new(protocol: ResponsesWireProtocol, turn_routing: TurnRoutingPolicy) -> Self {
        Self {
            protocol,
            turn_routing,
        }
    }

    /// Apply this protocol's turn-scoped routing value to an outgoing request.
    /// Unsupported providers ignore the state instead of leaking another
    /// provider's routing metadata.
    pub(crate) fn apply_turn_routing(
        self,
        builder: reqwest::RequestBuilder,
        state: Option<&TurnRoutingState>,
    ) -> reqwest::RequestBuilder {
        let TurnRoutingPolicy::FirstValueWinsHeader(header) = self.turn_routing else {
            return builder;
        };
        match state.and_then(TurnRoutingState::value) {
            Some(value) if HeaderValue::from_str(value).is_ok() => builder.header(header, value),
            _ => builder,
        }
    }

    /// Capture this protocol's routing value from response headers. The shared
    /// state enforces first-value-wins across headers and sideband frames.
    pub(crate) fn capture_turn_routing(
        self,
        headers: &HeaderMap,
        state: Option<&TurnRoutingState>,
    ) {
        let (TurnRoutingPolicy::FirstValueWinsHeader(header), Some(state)) =
            (self.turn_routing, state)
        else {
            return;
        };
        let Some(value) = headers
            .get(header)
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.is_empty())
        else {
            return;
        };
        state.capture_first(value.to_owned());
    }

    pub(crate) fn prepare_create_response(
        self,
        request: &ConversationRequest,
        metadata_origin: Option<ResponseMetadataOrigin>,
    ) -> Result<CreateResponseWrapper> {
        match self.protocol {
            ResponsesWireProtocol::Standard => {
                let mut wrapper = CreateResponseWrapper::new(request.into());
                wrapper.extra_tool_entries =
                    xai_grok_sampling_types::extra_tool_entries(&request.hosted_tools);
                Ok(wrapper)
            }
            ResponsesWireProtocol::Codex => {
                let response =
                    xai_grok_sampling_types::conversation_request_to_codex_create_response(request);
                let mut wrapper = CreateResponseWrapper::new(response);
                wrapper.response_message_item_ids =
                    xai_grok_sampling_types::response_message_item_ids(request);
                wrapper.response_item_metadata_passthrough =
                    xai_grok_sampling_types::response_item_metadata_passthrough_for_origin(
                        request,
                        metadata_origin.as_ref(),
                    )
                    .map_err(SamplingError::serialization_message)?;
                wrapper.response_metadata_origin = metadata_origin;
                Ok(wrapper)
            }
        }
    }

    /// Apply provider constraints after generic defaults have been populated.
    pub(crate) fn normalize_create_response(self, wrapper: &mut CreateResponseWrapper) {
        if self.protocol == ResponsesWireProtocol::Codex {
            let model = wrapper.inner.model.clone().unwrap_or_default();
            xai_grok_sampling_types::normalize_create_response_for_codex(
                &mut wrapper.inner,
                &model,
            );
        }
    }

    pub(crate) fn preserves_response_metadata(self) -> bool {
        self.protocol == ResponsesWireProtocol::Codex
    }

    /// Decode a recognized provider sideband. `None` means the caller must pass
    /// the frame to the standard typed Responses decoder unchanged.
    pub(crate) fn decode_sideband(self, data: &str) -> Result<Option<ResponsesSideband>> {
        match self.protocol {
            ResponsesWireProtocol::Standard => Ok(None),
            ResponsesWireProtocol::Codex => codex::decode_sideband(data, self.turn_routing),
        }
    }
}

mod codex {
    use super::*;

    const RESPONSE_METADATA_EVENT: &str = "response.metadata";

    pub(super) fn decode_sideband(
        data: &str,
        turn_routing: TurnRoutingPolicy,
    ) -> Result<Option<ResponsesSideband>> {
        // Avoid parsing every ordinary delta twice. The JSON discriminator
        // below remains authoritative when the cheap prefilter matches.
        if !data.contains(RESPONSE_METADATA_EVENT) {
            return Ok(None);
        }

        let value = serde_json::from_str::<serde_json::Value>(data)
            .map_err(SamplingError::Serialization)?;
        if value.get("type").and_then(serde_json::Value::as_str) != Some(RESPONSE_METADATA_EVENT) {
            return Ok(None);
        }

        let Some(headers) = value.get("headers") else {
            return Ok(Some(ResponsesSideband::default()));
        };
        let headers = headers.as_object().ok_or_else(|| {
            SamplingError::serialization_message(
                "Codex response.metadata `headers` must be a JSON object",
            )
        })?;
        let TurnRoutingPolicy::FirstValueWinsHeader(turn_routing_header) = turn_routing else {
            return Ok(Some(ResponsesSideband::default()));
        };
        let Some((_, raw_value)) = headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(turn_routing_header))
        else {
            return Ok(Some(ResponsesSideband::default()));
        };
        let routing_value = raw_value.as_str().or_else(|| {
            raw_value
                .as_array()
                .and_then(|values| values.first())
                .and_then(serde_json::Value::as_str)
        });
        let Some(routing_value) = routing_value.filter(|value| !value.is_empty()) else {
            return Err(SamplingError::serialization_message(
                "Codex response.metadata x-codex-turn-state must be a non-empty string or string array",
            ));
        };
        HeaderValue::from_str(routing_value).map_err(|_| {
            SamplingError::serialization_message(
                "Codex response.metadata x-codex-turn-state is not a valid HTTP header value",
            )
        })?;

        Ok(Some(ResponsesSideband {
            turn_routing_value: Some(routing_value.to_owned()),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const METADATA_FRAME: &str = r#"{
        "type": "response.metadata",
        "headers": {
            "X-Codex-Turn-State": ["turn-state-1"],
            "openai-model": "gpt-5-codex"
        },
        "metadata": {
            "openai_verification_recommendation": ["trusted_access_for_cyber"],
            "openai_chatgpt_moderation_metadata": {"opaque": true}
        },
        "safety_buffering": {"retry_model": "gpt-5-codex"}
    }"#;

    #[test]
    fn codex_consumes_metadata_and_returns_only_allowlisted_effects() {
        let sideband = ResponsesWireAdapter::new(
            ResponsesWireProtocol::Codex,
            TurnRoutingPolicy::FirstValueWinsHeader("x-codex-turn-state"),
        )
        .decode_sideband(METADATA_FRAME)
        .unwrap()
        .expect("recognized sideband");
        assert_eq!(
            sideband,
            ResponsesSideband {
                turn_routing_value: Some("turn-state-1".into())
            }
        );
    }

    #[test]
    fn standard_protocol_leaves_codex_metadata_for_fail_closed_decoder() {
        assert!(
            ResponsesWireAdapter::new(ResponsesWireProtocol::Standard, TurnRoutingPolicy::None)
                .decode_sideband(METADATA_FRAME)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn codex_near_matches_are_not_swallowed() {
        for frame in [
            r#"{"type":"response.metadata.delta"}"#,
            r#"{"type":"response.failed","message":"response.metadata"}"#,
        ] {
            assert!(
                ResponsesWireAdapter::new(
                    ResponsesWireProtocol::Codex,
                    TurnRoutingPolicy::FirstValueWinsHeader("x-codex-turn-state"),
                )
                .decode_sideband(frame)
                .unwrap()
                .is_none(),
                "{frame}"
            );
        }
    }

    #[test]
    fn codex_rejects_malformed_routing_header() {
        let error = ResponsesWireAdapter::new(
            ResponsesWireProtocol::Codex,
            TurnRoutingPolicy::FirstValueWinsHeader("x-codex-turn-state"),
        )
        .decode_sideband(r#"{"type":"response.metadata","headers":{"x-codex-turn-state":42}}"#)
        .unwrap_err();
        assert!(error.to_string().contains("non-empty string"), "{error}");
    }
}
