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

const CODEX_RESPONSES_LITE_HEADER: &str = "x-openai-internal-codex-responses-lite";

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

    /// Apply headers selected by the exact Responses dialect and model.
    pub(crate) fn apply_request_headers(
        self,
        builder: reqwest::RequestBuilder,
        model: &str,
    ) -> reqwest::RequestBuilder {
        if self.uses_responses_lite(model) {
            builder.header(CODEX_RESPONSES_LITE_HEADER, "true")
        } else {
            builder
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
                wrapper.response_message_metadata =
                    xai_grok_sampling_types::response_message_metadata(request)
                        .map_err(SamplingError::serialization_message)?;
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
            xai_grok_sampling_types::normalize_create_response_for_codex(&mut wrapper.inner);
        }
    }

    /// Finish the serialized request after typed metadata has been patched.
    /// Responses Lite's prefix is request-only and must never enter persisted
    /// conversation history.
    pub(crate) fn finalize_serialized_request(
        self,
        body: &mut serde_json::Value,
        wrapper: &CreateResponseWrapper,
    ) -> Result<()> {
        append_extra_tools(body, &wrapper.extra_tool_entries)?;
        self.append_client_metadata(
            body,
            wrapper
                .x_grok_session_id
                .as_deref()
                .or(wrapper.x_grok_conv_id.as_deref()),
            wrapper.x_grok_req_id.as_deref(),
        )?;
        let model = wrapper.inner.model.as_deref().unwrap_or_default();
        if self.uses_responses_lite(model) {
            codex::apply_responses_lite(body)?;
        }
        Ok(())
    }

    pub(crate) fn append_client_metadata(
        self,
        body: &mut serde_json::Value,
        session_id: Option<&str>,
        turn_id: Option<&str>,
    ) -> Result<()> {
        if self.protocol != ResponsesWireProtocol::Codex {
            return Ok(());
        }
        codex::append_client_metadata(body, session_id, turn_id)
    }

    fn uses_responses_lite(self, model: &str) -> bool {
        self.protocol == ResponsesWireProtocol::Codex
            && matches!(model, "gpt-5.6-sol" | "gpt-5.6-luna" | "gpt-5.6-terra")
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

fn append_extra_tools(body: &mut serde_json::Value, extra: &[serde_json::Value]) -> Result<()> {
    if extra.is_empty() {
        return Ok(());
    }
    let object = body.as_object_mut().ok_or_else(|| {
        SamplingError::serialization_message("Responses request body must be an object")
    })?;
    match object
        .entry("tools")
        .or_insert_with(|| serde_json::json!([]))
    {
        serde_json::Value::Array(tools) => tools.extend(extra.iter().cloned()),
        value @ serde_json::Value::Null => {
            *value = serde_json::Value::Array(extra.to_vec());
        }
        _ => {
            return Err(SamplingError::serialization_message(
                "Responses request tools must be an array",
            ));
        }
    }
    Ok(())
}

mod codex {
    use super::*;
    use std::sync::OnceLock;

    const KEEPALIVE_EVENT: &str = "keepalive";
    const RESPONSE_METADATA_EVENT: &str = "response.metadata";

    pub(super) fn append_client_metadata(
        body: &mut serde_json::Value,
        session_id: Option<&str>,
        turn_id: Option<&str>,
    ) -> Result<()> {
        let Some(session_id) = session_id.filter(|value| !value.is_empty()) else {
            return Ok(());
        };
        let object = body.as_object_mut().ok_or_else(|| {
            SamplingError::serialization_message("Responses request body must be an object")
        })?;
        let mut metadata = serde_json::Map::from_iter([
            (
                "x-codex-installation-id".to_owned(),
                serde_json::json!(process_identity(&INSTALLATION_ID)),
            ),
            ("session_id".to_owned(), serde_json::json!(session_id)),
            ("thread_id".to_owned(), serde_json::json!(session_id)),
            (
                "x-codex-window-id".to_owned(),
                serde_json::json!(process_identity(&WINDOW_ID)),
            ),
        ]);
        if let Some(turn_id) = turn_id.filter(|value| !value.is_empty()) {
            metadata.insert("turn_id".to_owned(), serde_json::json!(turn_id));
        }
        object.insert(
            "client_metadata".to_owned(),
            serde_json::Value::Object(metadata),
        );
        Ok(())
    }

    static INSTALLATION_ID: OnceLock<String> = OnceLock::new();
    static WINDOW_ID: OnceLock<String> = OnceLock::new();

    fn process_identity(slot: &'static OnceLock<String>) -> &'static str {
        slot.get_or_init(|| uuid::Uuid::new_v4().to_string())
    }

    pub(super) fn apply_responses_lite(body: &mut serde_json::Value) -> Result<()> {
        let object = body.as_object_mut().ok_or_else(|| {
            SamplingError::serialization_message("Responses request body must be an object")
        })?;

        let instructions = object
            .remove("instructions")
            .and_then(|value| value.as_str().map(ToOwned::to_owned))
            .unwrap_or_default();
        let tools = match object.remove("tools") {
            Some(serde_json::Value::Array(tools)) => responses_lite_tools(tools),
            Some(serde_json::Value::Null) | None => Vec::new(),
            Some(_) => {
                return Err(SamplingError::serialization_message(
                    "Responses Lite tools must be an array",
                ));
            }
        };

        let mut input = match object.remove("input") {
            Some(serde_json::Value::Array(items)) => items,
            Some(serde_json::Value::String(text)) => vec![serde_json::json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            })],
            Some(serde_json::Value::Null) | None => Vec::new(),
            Some(_) => {
                return Err(SamplingError::serialization_message(
                    "Responses Lite input must be text or an item array",
                ));
            }
        };
        let mut prefix = vec![serde_json::json!({
            "type": "additional_tools",
            "role": "developer",
            "tools": tools,
        })];
        if !instructions.is_empty() {
            prefix.push(serde_json::json!({
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": instructions}],
            }));
        }
        prefix.append(&mut input);
        object.insert("input".to_owned(), serde_json::Value::Array(prefix));
        object.insert("parallel_tool_calls".to_owned(), serde_json::json!(false));
        object
            .entry("tool_choice")
            .or_insert_with(|| serde_json::json!("auto"));

        let reasoning = object
            .entry("reasoning")
            .or_insert_with(|| serde_json::json!({}));
        if reasoning.is_null() {
            *reasoning = serde_json::json!({});
        }
        let reasoning = reasoning.as_object_mut().ok_or_else(|| {
            SamplingError::serialization_message("Responses Lite reasoning must be an object")
        })?;
        reasoning.remove("summary");
        reasoning.insert("context".to_owned(), serde_json::json!("all_turns"));

        let text = object
            .entry("text")
            .or_insert_with(|| serde_json::json!({}));
        if text.is_null() {
            *text = serde_json::json!({});
        }
        text.as_object_mut()
            .ok_or_else(|| {
                SamplingError::serialization_message("Responses Lite text must be an object")
            })?
            .insert("verbosity".to_owned(), serde_json::json!("low"));
        Ok(())
    }

    fn responses_lite_tools(tools: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
        let mut output = Vec::new();
        let mut functions = Vec::new();
        let mut functions_index = None;
        let mut functions_description = String::new();

        for tool in tools {
            let kind = tool.get("type").and_then(serde_json::Value::as_str);
            let is_functions_namespace = kind == Some("namespace")
                && tool.get("name").and_then(serde_json::Value::as_str) == Some("functions");
            if matches!(kind, Some("function" | "custom")) {
                functions_index.get_or_insert(output.len());
                functions.push(tool);
            } else if is_functions_namespace {
                functions_index.get_or_insert(output.len());
                if let Some(description) = tool
                    .get("description")
                    .and_then(serde_json::Value::as_str)
                    .filter(|description| !description.trim().is_empty())
                {
                    functions_description = description.to_owned();
                }
                if let Some(namespace_tools) =
                    tool.get("tools").and_then(serde_json::Value::as_array)
                {
                    functions.extend(namespace_tools.iter().cloned());
                }
            } else {
                output.push(tool);
            }
        }

        if let Some(index) = functions_index.filter(|_| !functions.is_empty()) {
            output.insert(
                index,
                serde_json::json!({
                    "type": "namespace",
                    "name": "functions",
                    "description": functions_description,
                    "tools": functions,
                }),
            );
        }
        output
    }

    pub(super) fn decode_sideband(
        data: &str,
        turn_routing: TurnRoutingPolicy,
    ) -> Result<Option<ResponsesSideband>> {
        // Avoid parsing every ordinary delta twice. The JSON discriminator
        // below remains authoritative when either cheap prefilter matches.
        if !data.contains(RESPONSE_METADATA_EVENT) && !data.contains(KEEPALIVE_EVENT) {
            return Ok(None);
        }

        let value = serde_json::from_str::<serde_json::Value>(data)
            .map_err(SamplingError::Serialization)?;
        match value.get("type").and_then(serde_json::Value::as_str) {
            Some(KEEPALIVE_EVENT) => return Ok(Some(ResponsesSideband::default())),
            Some(RESPONSE_METADATA_EVENT) => {}
            _ => return Ok(None),
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
    fn codex_consumes_keepalive_as_noop_sideband() {
        let sideband = ResponsesWireAdapter::new(
            ResponsesWireProtocol::Codex,
            TurnRoutingPolicy::FirstValueWinsHeader("x-codex-turn-state"),
        )
        .decode_sideband(r#"{"type":"keepalive"}"#)
        .unwrap()
        .expect("recognized heartbeat");
        assert_eq!(sideband, ResponsesSideband::default());
    }

    #[test]
    fn standard_protocol_leaves_codex_sidebands_for_fail_closed_decoder() {
        let adapter =
            ResponsesWireAdapter::new(ResponsesWireProtocol::Standard, TurnRoutingPolicy::None);
        for frame in [METADATA_FRAME, r#"{"type":"keepalive"}"#] {
            assert!(adapter.decode_sideband(frame).unwrap().is_none(), "{frame}");
        }
    }

    #[test]
    fn codex_near_matches_are_not_swallowed() {
        for frame in [
            r#"{"type":"response.metadata.delta"}"#,
            r#"{"type":"response.failed","message":"response.metadata"}"#,
            r#"{"type":"keepalive.delta"}"#,
            r#"{"type":"response.failed","message":"keepalive"}"#,
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

    #[test]
    fn codex_responses_lite_moves_instructions_and_tools_into_input_prefix() {
        let adapter = ResponsesWireAdapter::new(
            ResponsesWireProtocol::Codex,
            TurnRoutingPolicy::FirstValueWinsHeader("x-codex-turn-state"),
        );
        let mut wrapper = CreateResponseWrapper::new(xai_grok_sampling_types::rs::CreateResponse {
            model: Some("gpt-5.6-sol".into()),
            ..Default::default()
        });
        wrapper.x_grok_session_id = Some("session-1".into());
        wrapper.x_grok_req_id = Some("turn-1".into());
        wrapper.extra_tool_entries = vec![serde_json::json!({
            "type": "custom",
            "name": "shell",
            "description": "run a command"
        })];
        let mut body = serde_json::json!({
            "model": "gpt-5.6-sol",
            "instructions": "be precise",
            "input": [{"type":"message", "role":"user", "content":[]}],
            "tools": [{"type":"function", "name":"read_file", "strict":false}],
            "reasoning": {"effort":"high", "summary":"concise"},
            "text": {"format":{"type":"text"}},
        });

        adapter
            .finalize_serialized_request(&mut body, &wrapper)
            .unwrap();

        assert!(body.get("instructions").is_none());
        assert!(body.get("tools").is_none());
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["reasoning"]["context"], "all_turns");
        assert!(body["reasoning"].get("summary").is_none());
        assert_eq!(body["text"]["verbosity"], "low");
        assert_eq!(body["text"]["format"]["type"], "text");
        assert_eq!(body["input"][0]["type"], "additional_tools");
        assert_eq!(body["input"][0]["role"], "developer");
        let namespace = &body["input"][0]["tools"][0];
        assert_eq!(namespace["type"], "namespace");
        assert_eq!(namespace["name"], "functions");
        assert_eq!(namespace["tools"][0]["name"], "read_file");
        assert_eq!(namespace["tools"][1]["name"], "shell");
        assert_eq!(body["input"][1]["role"], "developer");
        assert_eq!(body["input"][1]["content"][0]["text"], "be precise");
        assert_eq!(body["input"][2]["role"], "user");
        assert_eq!(body["client_metadata"]["session_id"], "session-1");
        assert_eq!(body["client_metadata"]["thread_id"], "session-1");
        assert_eq!(body["client_metadata"]["turn_id"], "turn-1");
        for key in ["x-codex-installation-id", "x-codex-window-id"] {
            let value = body["client_metadata"][key].as_str().unwrap();
            uuid::Uuid::parse_str(value).expect("stable process identity is a UUID");
        }
    }

    #[test]
    fn standard_responses_does_not_apply_codex_wire_extensions() {
        let adapter =
            ResponsesWireAdapter::new(ResponsesWireProtocol::Standard, TurnRoutingPolicy::None);
        let mut wrapper = CreateResponseWrapper::new(xai_grok_sampling_types::rs::CreateResponse {
            model: Some("gpt-5.6-sol".into()),
            ..Default::default()
        });
        wrapper.x_grok_session_id = Some("session-1".into());
        let mut body = serde_json::json!({
            "instructions": "standard",
            "input": [],
            "tools": [],
        });

        adapter
            .finalize_serialized_request(&mut body, &wrapper)
            .unwrap();

        assert_eq!(body["instructions"], "standard");
        assert!(body.get("client_metadata").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn responses_lite_header_is_scoped_to_codex_lite_models() {
        let codex = ResponsesWireAdapter::new(
            ResponsesWireProtocol::Codex,
            TurnRoutingPolicy::FirstValueWinsHeader("x-codex-turn-state"),
        );
        let standard =
            ResponsesWireAdapter::new(ResponsesWireProtocol::Standard, TurnRoutingPolicy::None);
        let client = reqwest::Client::new();

        let lite = codex
            .apply_request_headers(client.post("http://example.test"), "gpt-5.6-luna")
            .build()
            .unwrap();
        assert_eq!(
            lite.headers()
                .get(CODEX_RESPONSES_LITE_HEADER)
                .and_then(|value| value.to_str().ok()),
            Some("true")
        );

        for request in [
            codex.apply_request_headers(client.post("http://example.test"), "unlisted-codex-model"),
            standard.apply_request_headers(client.post("http://example.test"), "gpt-5.6-sol"),
        ] {
            assert!(
                request
                    .build()
                    .unwrap()
                    .headers()
                    .get(CODEX_RESPONSES_LITE_HEADER)
                    .is_none()
            );
        }
    }
}
