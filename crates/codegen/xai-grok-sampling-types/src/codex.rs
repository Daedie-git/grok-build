//! Pure helpers for the ChatGPT/Codex Responses backend, plus shared
//! Responses stream reconstruction used by every Responses backend.
//!
//! No I/O. Used by the sampler to shape outbound CreateResponse bodies and to
//! rebuild empty `response.completed.output` from stream state (Grok and Codex).

use crate::rs;
use crate::{
    AssistantItem, BackendToolCallItem, BackendToolKind, CodexResponseMessageMetadata, ContentPart,
    ConversationItem, ConversationRequest, InternalChatMessageMetadataPassthrough,
    NativeCompactionCompatibility, NativeCompactionItemKind, NativeCompactionItemMetadata,
    ProviderReplayField, ReasoningEffort, ToolCall, ToolResultItem, UserMessageProviderMetadata,
};
use serde::{Deserialize, Serialize};

/// Codex Responses base URL.
pub const CODEX_BACKEND_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";

/// ChatGPT account usage endpoint used by the official Codex CLI.
pub const CODEX_ACCOUNT_USAGE_URL: &str = "https://chatgpt.com/backend-api/wham/usage";

/// HTTP header carrying the ChatGPT account id.
pub const CHATGPT_ACCOUNT_ID_HEADER: &str = "ChatGPT-Account-ID";
/// Client-family header required by the first-party Codex backend.
pub const CODEX_ORIGINATOR_HEADER: &str = "originator";
pub const CODEX_ORIGINATOR_VALUE: &str = "codex_cli_rs";

/// OpenAI OAuth client id used by the official Codex CLI.
pub const CODEX_OAUTH_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// Token refresh endpoint for ChatGPT OAuth.
pub const CODEX_OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";

/// True when `base_url` targets the ChatGPT/Codex backend.
pub fn is_codex_backend_url(base_url: &str) -> bool {
    let normalized = base_url.trim().trim_end_matches('/').to_ascii_lowercase();
    [
        "https://chatgpt.com/backend-api/codex",
        "https://chat.openai.com/backend-api/codex",
    ]
    .iter()
    .any(|canonical| {
        normalized == *canonical
            || normalized
                .strip_prefix(canonical)
                .is_some_and(|suffix| suffix.starts_with('/'))
    })
}

/// Message content accepted and returned by native Codex compaction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CodexCompactMessageContent {
    InputText {
        text: String,
    },
    InputImage {
        image_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<rs::ImageDetail>,
    },
    InputAudio {
        audio_url: String,
    },
    OutputText {
        text: String,
    },
}

/// Provider message item used by compact replacement history.
///
/// Unlike async-openai's request-side `InputMessage`, this retains the
/// provider-assigned ID returned by the compact endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CodexCompactMessage {
    #[serde(default)]
    pub r#type: rs::MessageType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub role: rs::Role,
    pub content: Vec<CodexCompactMessageContent>,
    #[serde(default, skip_serializing_if = "ProviderReplayField::is_missing")]
    pub status: ProviderReplayField<rs::OutputStatus>,
    #[serde(default, skip_serializing_if = "ProviderReplayField::is_missing")]
    pub phase: ProviderReplayField<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
}

/// Pinned async-openai input item plus official Codex transport fields that
/// upstream does not model on every request-side item class.
#[derive(Debug, Clone, PartialEq)]
pub struct CodexCompactTypedInputItem {
    pub item: rs::InputItem,
    pub item_id: Option<String>,
    pub internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
}

impl Serialize for CodexCompactTypedInputItem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut value = serde_json::to_value(&self.item).map_err(serde::ser::Error::custom)?;
        let object = value
            .as_object_mut()
            .ok_or_else(|| serde::ser::Error::custom("Responses compact input is not an object"))?;
        if let Some(id) = &self.item_id {
            object.insert("id".into(), serde_json::Value::String(id.clone()));
        }
        if let Some(metadata) = &self.internal_chat_message_metadata_passthrough {
            object.insert(
                "internal_chat_message_metadata_passthrough".into(),
                serde_json::to_value(metadata).map_err(serde::ser::Error::custom)?,
            );
        }
        value.serialize(serializer)
    }
}

/// Typed input union for the compact endpoint.
///
/// Native retained messages use the ID-capable message type; all ordinary
/// Responses items continue to use the pinned async-openai model.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
pub enum CodexCompactInputItem {
    Message(CodexCompactMessage),
    Item(CodexCompactTypedInputItem),
}

/// Typed replacement item returned by `POST /responses/compact`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CodexCompactOutputItem {
    Message {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        role: rs::Role,
        content: Vec<CodexCompactMessageContent>,
        #[serde(default, skip_serializing_if = "ProviderReplayField::is_missing")]
        status: ProviderReplayField<rs::OutputStatus>,
        #[serde(default, skip_serializing_if = "ProviderReplayField::is_missing")]
        phase: ProviderReplayField<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    Reasoning {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        summary: Vec<rs::SummaryPart>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<Vec<rs::ReasoningTextContent>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<rs::OutputStatus>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    #[serde(alias = "compaction_summary")]
    Compaction {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        encrypted_content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
}

/// Exact unary body accepted by Codex `POST /responses/compact`.
///
/// This intentionally contains only fields supported by the compact endpoint.
/// In particular, sampling/output controls such as `temperature`, `top_p`,
/// `max_output_tokens`, `stream`, and `store` cannot appear on this type.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CodexCompactRequest {
    pub model: String,
    pub input: Vec<CodexCompactInputItem>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub instructions: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<rs::Tool>>,
    pub parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<rs::Reasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<rs::ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<rs::ResponseTextParam>,
}

/// Unary response from Codex `POST /responses/compact`.
///
/// The endpoint returns provider-authored replacement history, not a normal
/// generated response. Its message records are response items (including IDs
/// and output-only status), so they intentionally do not use async-openai's
/// narrower request-side `InputItem` deserializer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CodexCompactResponse {
    pub output: Vec<CodexCompactOutputItem>,
}

/// Convert a normal conversation request into the compact endpoint's strict
/// body while sharing Codex's existing Responses normalization.
pub fn conversation_request_to_codex_compact_request(
    req: &ConversationRequest,
) -> std::result::Result<CodexCompactRequest, String> {
    conversation_request_to_codex_compact_request_for_origin(req, None)
}

pub fn conversation_request_to_codex_compact_request_for_origin(
    req: &ConversationRequest,
    origin: Option<&crate::ResponseMetadataOrigin>,
) -> std::result::Result<CodexCompactRequest, String> {
    let created = conversation_request_to_codex_create_response(req);
    let input = match created.input {
        rs::InputParam::Items(items) => {
            let mut message_metadata = response_message_metadata(req)?.into_iter();
            let metadata = response_item_metadata_passthrough_for_origin(req, origin)?;
            let mut input = items
                .into_iter()
                .enumerate()
                .map(|(input_index, item)| {
                    let metadata = metadata
                        .iter()
                        .find(|entry| entry.input_index == input_index);
                    let item_id = metadata.and_then(|entry| entry.item_id.clone());
                    let passthrough = metadata
                        .and_then(|entry| entry.internal_chat_message_metadata_passthrough.clone());
                    match item {
                        rs::InputItem::EasyMessage(message)
                            if matches!(message.role, rs::Role::User | rs::Role::Assistant) =>
                        {
                            let mut message_metadata = message_metadata
                                .next()
                                .expect("Responses conversion emitted an unexpected message");
                            message_metadata.item_id = item_id.or(message_metadata.item_id);
                            CodexCompactInputItem::Message(easy_message_with_response_metadata(
                                message,
                                message_metadata,
                                passthrough,
                            ))
                        }
                        other => CodexCompactInputItem::Item(CodexCompactTypedInputItem {
                            item: other,
                            item_id,
                            internal_chat_message_metadata_passthrough: passthrough,
                        }),
                    }
                })
                .collect();
            debug_assert!(message_metadata.next().is_none());
            reorder_response_input(&mut input, &metadata)?;
            input
        }
        rs::InputParam::Text(text) => {
            vec![CodexCompactInputItem::Item(CodexCompactTypedInputItem {
                item: rs::InputItem::EasyMessage(rs::EasyInputMessage {
                    r#type: rs::MessageType::Message,
                    role: rs::Role::User,
                    content: rs::EasyInputContent::Text(text),
                }),
                item_id: None,
                internal_chat_message_metadata_passthrough: None,
            })]
        }
    };
    Ok(CodexCompactRequest {
        model: created.model.unwrap_or_default(),
        input,
        instructions: created.instructions.unwrap_or_default(),
        tools: created.tools,
        parallel_tool_calls: created.parallel_tool_calls.unwrap_or(true),
        reasoning: created.reasoning,
        service_tier: created.service_tier,
        prompt_cache_key: created.prompt_cache_key,
        text: created.text,
    })
}

/// Provider fields corresponding one-for-one with user/assistant message
/// records emitted by Responses conversion. System messages are instructions
/// on Codex and tool-only assistants (no text and no item id) emit only
/// function calls, so neither consumes a slot here. Identified empty
/// assistant messages do occupy a slot so they can be replayed exactly.
pub fn response_message_metadata(
    req: &ConversationRequest,
) -> std::result::Result<Vec<CodexResponseMessageMetadata>, String> {
    // This also proves every provider-owned user envelope belongs to a valid
    // identity-bound native replacement before any wire patch can replay it.
    crate::native_compaction_compatibility(&req.items)?;
    Ok(req
        .items
        .iter()
        .filter_map(|item| match item {
            ConversationItem::User(user) => {
                let (status, phase) = user
                    .provider_metadata
                    .as_ref()
                    .map(UserMessageProviderMetadata::codex_fields)
                    .map(|(status, phase)| (status.clone(), phase.clone()))
                    .unwrap_or_default();
                Some(CodexResponseMessageMetadata {
                    item_id: user.response_item_id.clone(),
                    status,
                    phase,
                })
            }
            ConversationItem::Assistant(assistant) if assistant.projects_message() => {
                Some(CodexResponseMessageMetadata {
                    item_id: assistant.response_item_id.clone(),
                    status: ProviderReplayField::Missing,
                    phase: ProviderReplayField::Missing,
                })
            }
            _ => None,
        })
        .collect())
}

/// Derive all provider metadata/order bindings from the durable conversation.
/// Native compact entries retain their immutable manifest indices; each
/// ordinary group is matched completely against the following semantic output
/// owners and assigned the positions produced by Responses conversion.
pub fn response_item_metadata_passthrough(
    req: &ConversationRequest,
) -> std::result::Result<Vec<crate::ResponsesInputItemMetadata>, String> {
    response_item_metadata_passthrough_for_origin(req, None)
}

pub fn response_item_metadata_passthrough_for_origin(
    req: &ConversationRequest,
    origin: Option<&crate::ResponseMetadataOrigin>,
) -> std::result::Result<Vec<crate::ResponsesInputItemMetadata>, String> {
    use crate::ResponseOutputItemKind as Kind;

    let mut result = Vec::new();
    let mut input_index = 0usize;
    let mut active: Option<(&crate::ResponseOutputItemMetadata, Vec<bool>)> = None;
    let mut seen_response_ids = std::collections::BTreeSet::new();

    for item in &req.items {
        if let ConversationItem::Provider(provider) = item
            && let Some(metadata) = provider.as_response_output_metadata()
        {
            if active.is_some() {
                return Err("ordinary Responses output groups overlap".into());
            }
            // Missing and mismatched origins are intentionally ignored: the
            // semantic owner remains portable, but provider transport state is
            // valid only for the exact backend/model/account that produced it.
            if origin.is_none() || metadata.origin.as_ref() != origin {
                continue;
            }
            if metadata.response_id.is_empty() {
                return Err("ordinary Responses output group has an empty response id".into());
            }
            if !seen_response_ids.insert(metadata.response_id.as_str()) {
                return Err("ordinary Responses output group id is duplicated".into());
            }
            if metadata.output_items == 0
                || usize::try_from(metadata.output_items).ok() != Some(metadata.items.len())
            {
                return Err("ordinary Responses output manifest length is invalid".into());
            }
            let mut indices = std::collections::BTreeSet::new();
            for entry in &metadata.items {
                if entry.output_index >= metadata.output_items
                    || !indices.insert(entry.output_index)
                {
                    return Err(
                        "ordinary Responses output manifest has missing or duplicate indices"
                            .into(),
                    );
                }
            }
            if indices.iter().copied().ne(0..metadata.output_items) {
                return Err(
                    "ordinary Responses output manifest has missing or unordered indices".into(),
                );
            }
            active = Some((metadata, vec![false; metadata.items.len()]));
            continue;
        }

        let mut owners: Vec<(Kind, Option<&str>, Option<&str>)> = Vec::new();
        match item {
            ConversationItem::System(_) => {}
            ConversationItem::User(user) => {
                owners.push((Kind::Message, user.response_item_id.as_deref(), None));
            }
            ConversationItem::Reasoning(reasoning) => {
                owners.push((Kind::Reasoning, Some(reasoning.id.as_str()), None));
            }
            ConversationItem::Assistant(assistant) => {
                if assistant.projects_message() {
                    owners.push((Kind::Message, assistant.response_item_id.as_deref(), None));
                }
                owners.extend(
                    assistant
                        .tool_calls
                        .iter()
                        .map(|call| (Kind::FunctionCall, None, Some(call.id.as_ref()))),
                );
            }
            ConversationItem::ToolResult(output) => {
                owners.push((
                    Kind::FunctionCallOutput,
                    None,
                    Some(output.tool_call_id.as_str()),
                ));
            }
            ConversationItem::BackendToolCall(call) => match &call.kind {
                BackendToolKind::WebSearch(value) => {
                    owners.push((Kind::WebSearchCall, Some(value.id.as_str()), None));
                }
                BackendToolKind::XSearch(value) => owners.push((
                    Kind::CustomToolCall,
                    Some(value.id.as_str()),
                    Some(value.call_id.as_str()),
                )),
                BackendToolKind::CodeInterpreter(value) => {
                    owners.push((Kind::CodeInterpreterCall, Some(value.id.as_str()), None));
                }
            },
            ConversationItem::Provider(provider) => {
                if provider.is_encrypted_compaction() {
                    if active.is_some() {
                        return Err(
                            "ordinary Responses output group crosses native compaction".into()
                        );
                    }
                    input_index += 1;
                    continue;
                }
                debug_assert!(provider.is_native_compaction_metadata());
            }
        }

        if let Some((group, bound)) = &mut active {
            if matches!(
                item,
                ConversationItem::User(_) | ConversationItem::System(_)
            ) || matches!(item, ConversationItem::Provider(provider) if provider.is_native_compaction_item())
            {
                return Err("ordinary Responses output group crosses a response boundary".into());
            }
            for (owner_offset, (kind, item_id, call_id)) in owners.iter().enumerate() {
                let matches = group
                    .items
                    .iter()
                    .enumerate()
                    .filter(|(entry_index, entry)| {
                        !bound[*entry_index]
                            && entry.kind == *kind
                            && (item_id.is_none() || *item_id == entry.item_id.as_deref())
                            && *call_id == entry.call_id.as_deref()
                    })
                    .map(|(entry_index, _)| entry_index)
                    .collect::<Vec<_>>();
                let [entry_index] = matches.as_slice() else {
                    return Err(
                        "ordinary Responses output manifest does not uniquely match its owner"
                            .into(),
                    );
                };
                bound[*entry_index] = true;
                let entry = &group.items[*entry_index];
                result.push(crate::ResponsesInputItemMetadata {
                    input_index: input_index + owner_offset,
                    kind: entry.kind,
                    item_id: entry.item_id.clone(),
                    call_id: entry.call_id.clone(),
                    internal_chat_message_metadata_passthrough: entry
                        .internal_chat_message_metadata_passthrough
                        .clone(),
                    response_order: Some(crate::ResponsesInputItemOrder {
                        response_id: group.response_id.clone(),
                        output_index: entry.output_index,
                        output_items: group.output_items,
                    }),
                });
            }
        }
        input_index += owners.len();

        let complete = active
            .as_ref()
            .is_some_and(|(_, bound)| bound.iter().all(|value| *value));
        if complete {
            active = None;
        }

        if let ConversationItem::Provider(provider) = item
            && let Some(compatibility) = provider.as_native_compaction_metadata()
        {
            result.extend(compatibility.item_metadata.iter().filter_map(|entry| {
                let metadata = entry.internal_chat_message_metadata_passthrough.clone()?;
                let kind = match entry.kind {
                    NativeCompactionItemKind::Message => Kind::Message,
                    NativeCompactionItemKind::Reasoning => Kind::Reasoning,
                    NativeCompactionItemKind::Compaction => Kind::Compaction,
                };
                Some(crate::ResponsesInputItemMetadata {
                    input_index: entry.input_index,
                    kind,
                    item_id: entry.item_id.clone(),
                    call_id: None,
                    internal_chat_message_metadata_passthrough: Some(metadata),
                    response_order: None,
                })
            }));
        }
    }
    if active.is_some() {
        return Err("ordinary Responses output manifest is missing an owner".into());
    }
    result.sort_by_key(|entry| entry.input_index);
    Ok(result)
}

/// Restore provider transport metadata and output IDs after serializing
/// through the pinned async-openai request model.
pub fn patch_response_item_metadata_passthrough(
    body: &mut serde_json::Value,
    metadata: &[crate::ResponsesInputItemMetadata],
) -> std::result::Result<(), String> {
    // Standard Responses requests carry no Codex side table and must remain
    // untouched even though the shared transport calls this seam.
    if metadata.is_empty() {
        return Ok(());
    }
    let input = body
        .get_mut("input")
        .and_then(serde_json::Value::as_array_mut)
        .ok_or_else(|| "Responses request input is not an array".to_string())?;
    // Validate the complete ordering side table before mutating the request.
    let permutation = response_input_permutation(input.len(), metadata)?;
    for entry in metadata {
        let item = input
            .get_mut(entry.input_index)
            .ok_or_else(|| "Responses metadata input index is out of range".to_string())?;
        let item_type = item.get("type").and_then(serde_json::Value::as_str);
        let expected_type = match entry.kind {
            crate::ResponseOutputItemKind::Message => "message",
            crate::ResponseOutputItemKind::Reasoning => "reasoning",
            crate::ResponseOutputItemKind::FunctionCall => "function_call",
            crate::ResponseOutputItemKind::FunctionCallOutput => "function_call_output",
            crate::ResponseOutputItemKind::WebSearchCall => "web_search_call",
            crate::ResponseOutputItemKind::CustomToolCall => "custom_tool_call",
            crate::ResponseOutputItemKind::CodeInterpreterCall => "code_interpreter_call",
            crate::ResponseOutputItemKind::Compaction => "compaction",
        };
        if item_type != Some(expected_type)
            || entry.call_id.as_deref() != item.get("call_id").and_then(serde_json::Value::as_str)
            || (item.get("id").and_then(serde_json::Value::as_str).is_some()
                && entry.item_id.as_deref() != item.get("id").and_then(serde_json::Value::as_str))
        {
            return Err("Responses metadata no longer matches serialized input owner".into());
        }
        let object = item
            .as_object_mut()
            .ok_or_else(|| "Responses input item is not an object".to_string())?;
        if let Some(id) = &entry.item_id {
            object.insert("id".into(), serde_json::Value::String(id.clone()));
        }
        if let Some(passthrough) = &entry.internal_chat_message_metadata_passthrough {
            object.insert(
                "internal_chat_message_metadata_passthrough".into(),
                serde_json::to_value(passthrough)
                    .expect("Codex passthrough metadata is serializable"),
            );
        }
    }
    reorder_with_permutation(input, permutation);
    Ok(())
}

fn reorder_response_input<T>(
    input: &mut Vec<T>,
    metadata: &[crate::ResponsesInputItemMetadata],
) -> std::result::Result<(), String> {
    let permutation = response_input_permutation(input.len(), metadata)?;
    reorder_with_permutation(input, permutation);
    Ok(())
}

fn reorder_with_permutation<T>(input: &mut Vec<T>, permutation: Vec<usize>) {
    let mut old = std::mem::take(input)
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>();
    input.extend(permutation.into_iter().map(|old_index| {
        old[old_index]
            .take()
            .expect("validated permutation is unique")
    }));
}

fn response_input_permutation(
    input_len: usize,
    metadata: &[crate::ResponsesInputItemMetadata],
) -> std::result::Result<Vec<usize>, String> {
    let mut bound_input_indices = std::collections::BTreeSet::new();
    let mut groups: std::collections::BTreeMap<&str, Vec<&crate::ResponsesInputItemMetadata>> =
        std::collections::BTreeMap::new();
    for entry in metadata {
        if entry.input_index >= input_len {
            return Err("Responses metadata input index is out of range".into());
        }
        if !bound_input_indices.insert(entry.input_index) {
            return Err("Responses metadata contains duplicate input bindings".into());
        }
        if let Some(order) = &entry.response_order {
            groups
                .entry(order.response_id.as_str())
                .or_default()
                .push(entry);
        }
    }

    let mut claimed_positions = std::collections::BTreeSet::new();
    let mut permutation = (0..input_len).collect::<Vec<_>>();
    for (response_id, entries) in groups {
        if response_id.is_empty() {
            return Err("Responses order binding has an empty response id".into());
        }
        let Some(first_order) = entries
            .first()
            .and_then(|entry| entry.response_order.as_ref())
        else {
            unreachable!("group contains an order binding")
        };
        let output_items = first_order.output_items;
        if output_items == 0 || usize::try_from(output_items).ok() != Some(entries.len()) {
            return Err("Responses order group is missing an output binding".into());
        }
        let mut by_output = std::collections::BTreeMap::new();
        let mut positions = Vec::with_capacity(entries.len());
        for entry in entries {
            let order = entry.response_order.as_ref().expect("grouped order entry");
            if order.output_items != output_items
                || order.response_id != response_id
                || order.output_index >= output_items
                || by_output
                    .insert(order.output_index, entry.input_index)
                    .is_some()
            {
                return Err("Responses order group has duplicate or conflicting bindings".into());
            }
            positions.push(entry.input_index);
        }
        if by_output.keys().copied().ne(0..output_items) {
            return Err("Responses order group has missing or unordered bindings".into());
        }
        positions.sort_unstable();
        if positions
            .windows(2)
            .any(|window| window[1] != window[0] + 1)
        {
            return Err("Responses order group crosses a conversation boundary".into());
        }
        if positions
            .iter()
            .any(|position| !claimed_positions.insert(*position))
        {
            return Err("Responses order groups overlap".into());
        }
        for (target, source) in positions.into_iter().zip(by_output.into_values()) {
            permutation[target] = source;
        }
    }
    Ok(permutation)
}

/// Restore provider message fields after serializing through async-openai.
///
/// `EasyInputMessage` cannot represent the optional provider ID, status, or
/// phase accepted by Codex Responses. Validate alignment before mutating so an
/// internal conversion drift fails closed rather than attaching fields to the
/// wrong message.
pub fn patch_response_message_metadata(
    body: &mut serde_json::Value,
    message_metadata: &[CodexResponseMessageMetadata],
) -> std::result::Result<(), String> {
    // Standard Responses requests carry no Codex side table and must remain
    // untouched even though the shared transport calls this seam.
    if message_metadata.is_empty() {
        return Ok(());
    }
    let input = body
        .get_mut("input")
        .and_then(serde_json::Value::as_array_mut)
        .ok_or_else(|| "Responses request input is not an array".to_string())?;
    let message_indices = input
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            (item.get("type").and_then(serde_json::Value::as_str) == Some("message")
                && matches!(
                    item.get("role").and_then(serde_json::Value::as_str),
                    Some("user" | "assistant")
                ))
            .then_some(index)
        })
        .collect::<Vec<_>>();
    if message_indices.len() != message_metadata.len() {
        return Err("Responses message metadata no longer aligns with serialized input".into());
    }
    for (input_index, metadata) in message_indices.into_iter().zip(message_metadata) {
        let item = &mut input[input_index];
        let role = item.get("role").and_then(serde_json::Value::as_str);
        if (!metadata.status.is_missing() || !metadata.phase.is_missing()) && role != Some("user") {
            return Err("Codex retained user metadata is bound to a non-user message".into());
        }
        let object = item
            .as_object_mut()
            .ok_or_else(|| "Responses message input is not an object".to_string())?;
        if let Some(id) = &metadata.item_id {
            object.insert("id".into(), serde_json::Value::String(id.clone()));
        }
        if !metadata.status.is_missing() {
            object.insert(
                "status".into(),
                serde_json::to_value(&metadata.status)
                    .expect("Codex message status is serializable"),
            );
        }
        if !metadata.phase.is_missing() {
            object.insert(
                "phase".into(),
                serde_json::to_value(&metadata.phase).expect("Codex message phase is serializable"),
            );
        }
    }
    Ok(())
}

fn easy_message_with_response_metadata(
    message: rs::EasyInputMessage,
    metadata: CodexResponseMessageMetadata,
    internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
) -> CodexCompactMessage {
    let content = match message.content {
        rs::EasyInputContent::Text(text) => vec![match message.role {
            rs::Role::Assistant => CodexCompactMessageContent::OutputText { text },
            _ => CodexCompactMessageContent::InputText { text },
        }],
        rs::EasyInputContent::ContentList(parts) => parts
            .into_iter()
            .map(|part| match part {
                rs::InputContent::InputText(text) => {
                    CodexCompactMessageContent::InputText { text: text.text }
                }
                rs::InputContent::InputImage(image) => CodexCompactMessageContent::InputImage {
                    image_url: image.image_url.or(image.file_id).unwrap_or_default(),
                    detail: Some(image.detail),
                },
                rs::InputContent::InputFile(_) => CodexCompactMessageContent::InputText {
                    text: "[retained input file]".into(),
                },
            })
            .collect(),
    };
    CodexCompactMessage {
        r#type: rs::MessageType::Message,
        id: metadata.item_id,
        role: message.role,
        content,
        status: metadata.status,
        phase: metadata.phase,
        internal_chat_message_metadata_passthrough,
    }
}

/// Decode provider replacement history into the durable conversation model.
///
/// Codex currently returns retained user messages plus native reasoning and a
/// final encrypted compaction item. Other item classes are rejected rather
/// than silently flattened into text, because doing so would make the next
/// request differ from the provider's replacement transcript.
pub fn codex_compact_output_to_conversation(
    output: Vec<CodexCompactOutputItem>,
    mut compatibility: NativeCompactionCompatibility,
) -> std::result::Result<Vec<ConversationItem>, String> {
    if compatibility.schema_version != NativeCompactionCompatibility::SCHEMA_VERSION {
        return Err("new native compaction output requires the current manifest schema".into());
    }
    let mut items = Vec::new();
    let mut item_metadata = Vec::new();
    for item in output {
        let (conversation_item, kind, item_id, metadata) = match item {
            CodexCompactOutputItem::Message {
                id,
                role,
                content,
                status,
                phase,
                internal_chat_message_metadata_passthrough,
            } => {
                // Provider-authored instructions may be stale or duplicated.
                // The caller restores the current canonical System item once.
                // Filter them before validating message-only replay fields: no
                // ignored instruction field can affect the installed history.
                if matches!(role, rs::Role::System | rs::Role::Developer) {
                    continue;
                }
                if content
                    .iter()
                    .any(|part| !matches!(part, CodexCompactMessageContent::InputText { .. }))
                {
                    return Err(format!(
                        "unsupported compact message content for {role:?}: only input_text is losslessly replayable"
                    ));
                }
                let content = compact_message_content_to_parts(content)?;
                let item = match role {
                    rs::Role::User => ConversationItem::User(crate::UserItem {
                        content,
                        response_item_id: id.clone(),
                        provider_metadata: Some(UserMessageProviderMetadata::codex(status, phase)),
                        ..Default::default()
                    }),
                    rs::Role::Assistant => {
                        return Err(
                            "unsupported compact assistant message: output_text cannot be replayed exactly"
                                .to_string(),
                        );
                    }
                    rs::Role::System | rs::Role::Developer => unreachable!(),
                };
                (
                    item,
                    NativeCompactionItemKind::Message,
                    id,
                    internal_chat_message_metadata_passthrough,
                )
            }
            CodexCompactOutputItem::Reasoning {
                id,
                summary,
                content,
                encrypted_content,
                status,
                internal_chat_message_metadata_passthrough,
            } => {
                let id = id.ok_or_else(|| "compact reasoning item had no id".to_string())?;
                (
                    ConversationItem::Reasoning(rs::ReasoningItem {
                        id: id.clone(),
                        summary,
                        content,
                        encrypted_content,
                        status,
                    }),
                    NativeCompactionItemKind::Reasoning,
                    Some(id),
                    internal_chat_message_metadata_passthrough,
                )
            }
            CodexCompactOutputItem::Compaction {
                id,
                encrypted_content,
                internal_chat_message_metadata_passthrough,
            } => (
                ConversationItem::Provider(crate::ProviderItem::encrypted_compaction(
                    rs::CompactionSummaryItemParam {
                        id: id.clone(),
                        encrypted_content,
                    },
                )),
                NativeCompactionItemKind::Compaction,
                id,
                internal_chat_message_metadata_passthrough,
            ),
        };
        let user_message_provider_metadata = match &conversation_item {
            ConversationItem::User(user) => user.provider_metadata.clone(),
            _ => None,
        };
        item_metadata.push(NativeCompactionItemMetadata {
            input_index: items.len(),
            kind,
            item_id,
            internal_chat_message_metadata_passthrough: metadata,
            user_message_provider_metadata,
        });
        items.push(conversation_item);
    }
    compatibility.replacement_segment_items = items.len();
    compatibility.item_metadata = item_metadata;
    let mut compaction_indices = items.iter().enumerate().filter_map(|(index, item)| {
        matches!(item, ConversationItem::Provider(provider) if provider.is_encrypted_compaction())
            .then_some(index)
    });
    let compaction_index = compaction_indices
        .next()
        .ok_or_else(|| "compact output contained no encrypted compaction item".to_string())?;
    if compaction_indices.next().is_some() {
        return Err("compact output contained multiple encrypted compaction items".to_string());
    }
    items.insert(
        compaction_index,
        ConversationItem::Provider(crate::ProviderItem::native_compaction_metadata(
            compatibility,
        )),
    );
    Ok(items)
}

fn compact_message_content_to_parts(
    content: Vec<CodexCompactMessageContent>,
) -> std::result::Result<Vec<ContentPart>, String> {
    content
        .into_iter()
        .map(|part| match part {
            CodexCompactMessageContent::InputText { text }
            | CodexCompactMessageContent::OutputText { text } => {
                Ok(ContentPart::Text { text: text.into() })
            }
            CodexCompactMessageContent::InputImage { image_url, .. } => Ok(ContentPart::Image {
                url: image_url.into(),
            }),
            CodexCompactMessageContent::InputAudio { .. } => {
                Err("compact output retained unsupported input_audio".to_string())
            }
        })
        .collect()
}

/// One ordinary Responses output item decoded from raw JSON before the
/// pinned async-openai model can discard provider metadata.
///
/// Grok's shell currently exposes a numeric prompt index, not the trustworthy
/// turn-scoped UUID Codex stamps here. Supplied metadata is therefore preserved
/// exactly, but missing metadata is not synthesized from that wrong identifier.
#[derive(Debug, Clone)]
pub enum CapturedResponseOutputItemValue {
    Typed(rs::OutputItem),
    FunctionCallOutput(rs::FunctionCallOutputItemParam),
}

/// Position-bound raw/typed bridge for an ordinary output item.
#[derive(Debug, Clone)]
pub struct CapturedResponseOutputItem {
    pub output_index: u32,
    pub value: CapturedResponseOutputItemValue,
    pub internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    pub metadata_origin: Option<crate::ResponseMetadataOrigin>,
}

impl CapturedResponseOutputItem {
    pub fn set_metadata_origin(&mut self, origin: Option<&crate::ResponseMetadataOrigin>) {
        self.metadata_origin = origin.cloned();
    }

    pub fn kind(&self) -> Option<crate::ResponseOutputItemKind> {
        use crate::ResponseOutputItemKind as Kind;
        match &self.value {
            CapturedResponseOutputItemValue::FunctionCallOutput(_) => {
                Some(Kind::FunctionCallOutput)
            }
            CapturedResponseOutputItemValue::Typed(item) => match item {
                rs::OutputItem::Message(_) => Some(Kind::Message),
                rs::OutputItem::Reasoning(_) => Some(Kind::Reasoning),
                rs::OutputItem::FunctionCall(_) => Some(Kind::FunctionCall),
                rs::OutputItem::WebSearchCall(_) => Some(Kind::WebSearchCall),
                rs::OutputItem::CustomToolCall(_) => Some(Kind::CustomToolCall),
                rs::OutputItem::CodeInterpreterCall(_) => Some(Kind::CodeInterpreterCall),
                _ => None,
            },
        }
    }

    pub fn item_id(&self) -> Option<&str> {
        match &self.value {
            CapturedResponseOutputItemValue::FunctionCallOutput(item) => item.id.as_deref(),
            CapturedResponseOutputItemValue::Typed(item) => match item {
                rs::OutputItem::Message(item) => Some(item.id.as_str()),
                rs::OutputItem::Reasoning(item) => Some(item.id.as_str()),
                rs::OutputItem::FunctionCall(item) => item.id.as_deref(),
                rs::OutputItem::WebSearchCall(item) => Some(item.id.as_str()),
                rs::OutputItem::CustomToolCall(item) => Some(item.id.as_str()),
                rs::OutputItem::CodeInterpreterCall(item) => Some(item.id.as_str()),
                _ => None,
            },
        }
    }

    pub fn call_id(&self) -> Option<&str> {
        match &self.value {
            CapturedResponseOutputItemValue::FunctionCallOutput(item) => Some(&item.call_id),
            CapturedResponseOutputItemValue::Typed(rs::OutputItem::FunctionCall(item)) => {
                Some(&item.call_id)
            }
            CapturedResponseOutputItemValue::Typed(rs::OutputItem::CustomToolCall(item)) => {
                Some(&item.call_id)
            }
            _ => None,
        }
    }

    pub fn durable_order(&self) -> Option<crate::ResponseOutputItemOrder> {
        Some(crate::ResponseOutputItemOrder {
            output_index: self.output_index,
            kind: self.kind()?,
            item_id: self.item_id().map(str::to_owned),
            call_id: self.call_id().map(str::to_owned),
            internal_chat_message_metadata_passthrough: self
                .internal_chat_message_metadata_passthrough
                .clone(),
        })
    }
}

/// Raw-aware SSE event. Output-item events are represented separately so
/// function-call outputs absent from async-openai's output union remain typed.
#[derive(Debug, Clone)]
pub enum DecodedResponseStreamEvent {
    Event {
        event: rs::ResponseStreamEvent,
        terminal_output: Option<Vec<CapturedResponseOutputItem>>,
    },
    OutputItemAdded(CapturedResponseOutputItem),
    OutputItemDone(CapturedResponseOutputItem),
}

impl From<rs::ResponseStreamEvent> for DecodedResponseStreamEvent {
    fn from(event: rs::ResponseStreamEvent) -> Self {
        match event {
            rs::ResponseStreamEvent::ResponseOutputItemAdded(event) => {
                Self::OutputItemAdded(CapturedResponseOutputItem {
                    output_index: event.output_index,
                    value: CapturedResponseOutputItemValue::Typed(event.item),
                    internal_chat_message_metadata_passthrough: None,
                    metadata_origin: None,
                })
            }
            rs::ResponseStreamEvent::ResponseOutputItemDone(event) => {
                Self::OutputItemDone(CapturedResponseOutputItem {
                    output_index: event.output_index,
                    value: CapturedResponseOutputItemValue::Typed(event.item),
                    internal_chat_message_metadata_passthrough: None,
                    metadata_origin: None,
                })
            }
            event => Self::Event {
                event,
                terminal_output: None,
            },
        }
    }
}

impl DecodedResponseStreamEvent {
    pub fn with_metadata_origin(mut self, origin: Option<&crate::ResponseMetadataOrigin>) -> Self {
        match &mut self {
            Self::Event {
                terminal_output: Some(items),
                ..
            } => {
                for item in items {
                    item.set_metadata_origin(origin);
                }
            }
            Self::OutputItemAdded(item) | Self::OutputItemDone(item) => {
                item.set_metadata_origin(origin);
            }
            Self::Event {
                terminal_output: None,
                ..
            } => {}
        }
        self
    }
}

/// Raw-aware unary `/responses` result.
#[derive(Debug, Clone)]
pub struct DecodedResponse {
    pub response: rs::Response,
    pub output: Vec<CapturedResponseOutputItem>,
}

impl DecodedResponse {
    pub fn set_metadata_origin(&mut self, origin: Option<&crate::ResponseMetadataOrigin>) {
        for item in &mut self.output {
            item.set_metadata_origin(origin);
        }
    }
}

/// Stream-side accumulator for the Responses API: text deltas + position-bound
/// output items. Used for **all** Responses backends (Grok and Codex). When
/// `response.completed.output` is empty after streamed text / `output_item.*`
/// events, [`Self::terminal_output`] rebuilds the assistant turn so empty-response
/// retries do not fire. Codex-only metadata/manifest attachment still requires
/// a `metadata_origin` on captured items.
#[derive(Debug, Default, Clone)]
pub struct ResponsesStreamAccumulator {
    pub text_deltas: String,
    /// Latest items from `response.output_item.added`, keyed by output_index.
    pub added_by_index: std::collections::BTreeMap<u32, CapturedResponseOutputItem>,
    /// Items from `response.output_item.done`, keyed by output_index.
    pub items_by_index: std::collections::BTreeMap<u32, CapturedResponseOutputItem>,
    /// Streamed assistant text keyed by its provider output position. This is
    /// needed when a backend completes tool items but omits the message's
    /// `output_item.done` and then sends an empty terminal output.
    streamed_text_by_index: std::collections::BTreeMap<u32, StreamedOutputText>,
}

#[derive(Debug, Default, Clone)]
struct StreamedOutputText {
    item_id: String,
    text: String,
}

fn rebuilt_message_item(
    output_index: u32,
    streamed_item_id: &str,
    text: &str,
    template: Option<&CapturedResponseOutputItem>,
) -> CapturedResponseOutputItem {
    let template_message = template.and_then(|item| match &item.value {
        CapturedResponseOutputItemValue::Typed(rs::OutputItem::Message(message)) => Some(message),
        _ => None,
    });
    let id = template_message
        .map(|message| message.id.clone())
        .filter(|id| !id.is_empty())
        .or_else(|| (!streamed_item_id.is_empty()).then(|| streamed_item_id.to_string()))
        .unwrap_or_else(|| "msg_responses_rebuilt".to_string());
    let role = template_message
        .map(|message| message.role.clone())
        .unwrap_or(rs::AssistantRole::Assistant);

    CapturedResponseOutputItem {
        output_index,
        value: CapturedResponseOutputItemValue::Typed(rs::OutputItem::Message(rs::OutputMessage {
            content: vec![rs::OutputMessageContent::OutputText(
                rs::OutputTextContent {
                    text: text.to_string(),
                    annotations: vec![],
                    logprobs: None,
                },
            )],
            id,
            role,
            status: rs::OutputStatus::Completed,
        })),
        internal_chat_message_metadata_passthrough: template
            .and_then(|item| item.internal_chat_message_metadata_passthrough.clone()),
        metadata_origin: template.and_then(|item| item.metadata_origin.clone()),
    }
}

/// Backward-compatible alias for [`ResponsesStreamAccumulator`].
pub type CodexStreamAccumulator = ResponsesStreamAccumulator;

impl ResponsesStreamAccumulator {
    pub fn note_text_delta(&mut self, delta: &str) {
        self.text_deltas.push_str(delta);
    }

    pub fn note_indexed_text_delta(&mut self, output_index: u32, item_id: &str, delta: &str) {
        self.note_text_delta(delta);
        let streamed = self.streamed_text_by_index.entry(output_index).or_default();
        if streamed.item_id.is_empty() {
            streamed.item_id = item_id.to_string();
        }
        streamed.text.push_str(delta);
    }

    pub fn note_text_done(&mut self, text: &str) {
        if self.text_deltas.is_empty() && !text.is_empty() {
            self.text_deltas.push_str(text);
        }
    }

    pub fn note_indexed_text_done(&mut self, output_index: u32, item_id: &str, text: &str) {
        self.note_text_done(text);
        let streamed = self.streamed_text_by_index.entry(output_index).or_default();
        if streamed.item_id.is_empty() {
            streamed.item_id = item_id.to_string();
        }
        if streamed.text.is_empty() {
            streamed.text.push_str(text);
        }
    }

    pub fn note_output_item_added(&mut self, item: CapturedResponseOutputItem) {
        self.added_by_index.insert(item.output_index, item);
    }

    pub fn note_captured_output_item_done(&mut self, mut item: CapturedResponseOutputItem) {
        if item.internal_chat_message_metadata_passthrough.is_none()
            && let Some(added) = self.added_by_index.get(&item.output_index)
            && added.kind() == item.kind()
            && added.item_id() == item.item_id()
            && added.call_id() == item.call_id()
        {
            item.internal_chat_message_metadata_passthrough =
                added.internal_chat_message_metadata_passthrough.clone();
            item.metadata_origin = added.metadata_origin.clone();
        }
        self.items_by_index.insert(item.output_index, item);
    }

    /// Compatibility helper for typed callers that have no raw sidecar.
    pub fn note_output_item_done(&mut self, output_index: u32, item: rs::OutputItem) {
        self.note_captured_output_item_done(CapturedResponseOutputItem {
            output_index,
            value: CapturedResponseOutputItemValue::Typed(item),
            internal_chat_message_metadata_passthrough: None,
            metadata_origin: None,
        });
    }

    /// Return the terminal assistant text without double-appending streamed
    /// deltas. Finalized message items are authoritative when present;
    /// otherwise output-text deltas (or an output-text-done fallback) win.
    pub fn final_text(&self) -> String {
        let finalized: String = self
            .items_by_index
            .values()
            .filter_map(|item| match &item.value {
                CapturedResponseOutputItemValue::Typed(rs::OutputItem::Message(message)) => {
                    Some(&message.content)
                }
                _ => None,
            })
            .flat_map(|content| content.iter())
            .filter_map(|part| match part {
                rs::OutputMessageContent::OutputText(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        if finalized.is_empty() {
            self.text_deltas.clone()
        } else {
            finalized
        }
    }

    /// Rebuild position-bound output for an empty terminal response.
    pub fn rebuilt_captured_output(&self) -> Vec<CapturedResponseOutputItem> {
        let mut rebuilt = self.items_by_index.clone();

        for (output_index, streamed) in &self.streamed_text_by_index {
            let has_final_message = rebuilt.get(output_index).is_some_and(|item| {
                matches!(
                    &item.value,
                    CapturedResponseOutputItemValue::Typed(rs::OutputItem::Message(_))
                )
            });
            if has_final_message || streamed.text.is_empty() {
                continue;
            }
            rebuilt.insert(
                *output_index,
                rebuilt_message_item(
                    *output_index,
                    &streamed.item_id,
                    &streamed.text,
                    self.added_by_index.get(output_index),
                ),
            );
        }

        let has_any_message = rebuilt.values().any(|item| {
            matches!(
                &item.value,
                CapturedResponseOutputItemValue::Typed(rs::OutputItem::Message(_))
            )
        });
        if !self.text_deltas.is_empty()
            && self.streamed_text_by_index.is_empty()
            && !has_any_message
        {
            let added_message = self.added_by_index.iter().find(|(index, item)| {
                !rebuilt.contains_key(index)
                    && matches!(
                        &item.value,
                        CapturedResponseOutputItemValue::Typed(rs::OutputItem::Message(_))
                    )
            });
            let (output_index, template) = match added_message {
                Some((index, item)) => (*index, Some(item)),
                None => {
                    let mut index = 0;
                    while rebuilt.contains_key(&index) && index < u32::MAX {
                        index += 1;
                    }
                    (index, None)
                }
            };
            rebuilt.insert(
                output_index,
                rebuilt_message_item(
                    output_index,
                    "msg_responses_rebuilt",
                    &self.text_deltas,
                    template,
                ),
            );
        }

        rebuilt.into_values().collect()
    }

    pub fn rebuilt_output(&self) -> Vec<rs::OutputItem> {
        self.rebuilt_captured_output()
            .into_iter()
            .filter_map(|item| match item.value {
                CapturedResponseOutputItemValue::Typed(item) => Some(item),
                CapturedResponseOutputItemValue::FunctionCallOutput(_) => None,
            })
            .collect()
    }

    /// If typed terminal output is empty, fill it without metadata. Raw-aware
    /// consumers should use [`Self::terminal_output`] instead.
    pub fn fill_empty_response_output(&self, response: &mut rs::Response) {
        if response.output.is_empty() {
            response.output = self.rebuilt_output();
        }
    }

    pub fn terminal_output(
        &self,
        terminal: Option<Vec<CapturedResponseOutputItem>>,
    ) -> Vec<CapturedResponseOutputItem> {
        match terminal {
            Some(mut items) if !items.is_empty() => {
                for item in &mut items {
                    if item.internal_chat_message_metadata_passthrough.is_some() {
                        continue;
                    }
                    let streamed = self
                        .items_by_index
                        .get(&item.output_index)
                        .or_else(|| self.added_by_index.get(&item.output_index));
                    if let Some(streamed) = streamed
                        && streamed.kind() == item.kind()
                        && streamed.item_id() == item.item_id()
                        && streamed.call_id() == item.call_id()
                    {
                        item.internal_chat_message_metadata_passthrough =
                            streamed.internal_chat_message_metadata_passthrough.clone();
                        if item.metadata_origin.is_none() {
                            item.metadata_origin = streamed.metadata_origin.clone();
                        }
                    }
                }
                items
            }
            _ => self.rebuilt_captured_output(),
        }
    }
}

/// Convert a raw-aware ordinary Responses result to durable semantic items and
/// a complete response-group manifest for exact Codex replay ordering.
pub fn captured_response_to_conversation_items(
    response: rs::Response,
    output: Vec<CapturedResponseOutputItem>,
) -> std::result::Result<Vec<ConversationItem>, String> {
    let model_id = response.model.clone();
    let model_fingerprint = response
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("system_fingerprint"))
        .cloned()
        .filter(|value| !value.is_empty());
    let reasoning_effort = response
        .reasoning
        .as_ref()
        .and_then(|reasoning| reasoning.effort.clone())
        .map(crate::ReasoningEffort::from_responses_api);

    let metadata_origin = output
        .iter()
        .find_map(|captured| captured.metadata_origin.clone());
    let response_manifest = if let Some(origin) = metadata_origin {
        if response.id.is_empty() {
            return Err("Codex Responses output has an empty response group id".into());
        }
        let output_items = u32::try_from(output.len())
            .map_err(|_| "Codex Responses output item count exceeds u32")?;
        if output_items == 0 {
            None
        } else {
            let mut items = Vec::with_capacity(output.len());
            for (expected_index, captured) in output.iter().enumerate() {
                if captured.metadata_origin.as_ref() != Some(&origin) {
                    return Err(
                        "Codex Responses output group has missing or conflicting origins".into(),
                    );
                }
                if captured.output_index
                    != u32::try_from(expected_index)
                        .map_err(|_| "Codex Responses output index exceeds u32")?
                {
                    return Err(
                        "Codex Responses output group has missing, duplicate, or unordered indices"
                            .into(),
                    );
                }
                items.push(captured.durable_order().ok_or_else(|| {
                    "Codex Responses output variant cannot be persisted and ordered exactly"
                        .to_string()
                })?);
            }
            Some(crate::ResponseOutputItemMetadata {
                response_id: response.id.clone(),
                output_items,
                items,
                origin: Some(origin),
            })
        }
    } else {
        None
    };
    let require_exact = response_manifest.is_some();

    let mut items = Vec::with_capacity(output.len() + 2);
    let mut content = String::new();
    let mut response_item_id = None;
    let mut exact_message_indices = Vec::new();
    let mut exact_message_ids = std::collections::BTreeSet::new();
    let mut tool_calls = Vec::new();

    for captured in output {
        match captured.value {
            CapturedResponseOutputItemValue::Typed(rs::OutputItem::Message(message)) => {
                let mut message_content = String::new();
                for part in message.content {
                    match part {
                        rs::OutputMessageContent::OutputText(text) => {
                            if !message_content.is_empty() {
                                message_content.push('\n');
                            }
                            message_content.push_str(&text.text);
                        }
                        _ if require_exact => {
                            return Err(
                                "Codex Responses message contains unsupported content".into()
                            );
                        }
                        _ => {}
                    }
                }

                if require_exact {
                    if message.id.is_empty() {
                        return Err("Codex Responses message id is empty".into());
                    }
                    if !exact_message_ids.insert(message.id.clone()) {
                        return Err("Codex Responses message id is duplicated".into());
                    }
                    exact_message_indices.push(items.len());
                    items.push(ConversationItem::Assistant(AssistantItem {
                        content: std::sync::Arc::<str>::from(message_content),
                        response_item_id: Some(message.id),
                        tool_calls: Vec::new(),
                        model_id: None,
                        model_fingerprint: None,
                        reasoning_effort: None,
                    }));
                } else {
                    response_item_id = Some(message.id);
                    if !content.is_empty() && !message_content.is_empty() {
                        content.push('\n');
                    }
                    content.push_str(&message_content);
                }
            }
            CapturedResponseOutputItemValue::Typed(rs::OutputItem::FunctionCall(call)) => {
                tool_calls.push(ToolCall {
                    id: std::sync::Arc::<str>::from(call.call_id),
                    name: call.name,
                    arguments: std::sync::Arc::<str>::from(call.arguments),
                });
            }
            CapturedResponseOutputItemValue::Typed(rs::OutputItem::Reasoning(reasoning)) => {
                items.push(ConversationItem::Reasoning(reasoning));
            }
            CapturedResponseOutputItemValue::Typed(rs::OutputItem::WebSearchCall(call)) => {
                items.push(ConversationItem::BackendToolCall(BackendToolCallItem {
                    kind: BackendToolKind::WebSearch(call),
                }));
            }
            CapturedResponseOutputItemValue::Typed(rs::OutputItem::CustomToolCall(call)) => {
                items.push(ConversationItem::BackendToolCall(BackendToolCallItem {
                    kind: BackendToolKind::XSearch(call),
                }));
            }
            CapturedResponseOutputItemValue::Typed(rs::OutputItem::CodeInterpreterCall(call)) => {
                items.push(ConversationItem::BackendToolCall(BackendToolCallItem {
                    kind: BackendToolKind::CodeInterpreter(call),
                }));
            }
            CapturedResponseOutputItemValue::FunctionCallOutput(output) => {
                let (content, images) = function_call_output_to_durable(&output, require_exact)?;
                items.push(ConversationItem::ToolResult(ToolResultItem {
                    tool_call_id: output.call_id,
                    content: std::sync::Arc::<str>::from(content),
                    images,
                }));
            }
            CapturedResponseOutputItemValue::Typed(_) if require_exact => {
                return Err(
                    "Codex Responses output variant cannot be persisted and replayed".into(),
                );
            }
            CapturedResponseOutputItemValue::Typed(_) => {}
        }
    }

    if require_exact {
        let tool_call_ids: std::collections::HashSet<String> = tool_calls
            .iter()
            .map(|call| call.id.as_ref().to_owned())
            .collect();
        let tool_owner_index =
            if let Some(last_message_index) = exact_message_indices.last().copied() {
                let ConversationItem::Assistant(assistant) = &mut items[last_message_index] else {
                    unreachable!("exact message index must identify an assistant item");
                };
                assistant.tool_calls = tool_calls;
                assistant.model_id = Some(model_id);
                assistant.model_fingerprint = model_fingerprint;
                assistant.reasoning_effort = reasoning_effort;
                last_message_index
            } else {
                items.push(ConversationItem::Assistant(AssistantItem {
                    content: std::sync::Arc::<str>::from(content),
                    response_item_id,
                    tool_calls,
                    model_id: Some(model_id),
                    model_fingerprint,
                    reasoning_effort,
                }));
                items.len() - 1
            };

        if !tool_call_ids.is_empty() {
            // Exact manifests restore provider wire order independently of
            // durable order. Results for calls from an earlier response must
            // remain before this response's tool owner; results for the new
            // calls must follow it. Keep all other provider output siblings
            // before the owner so hosted-tool outputs cannot become generic
            // assistant-message boundaries inside the result run.
            let assistant = items.remove(tool_owner_index);
            let mut previous_results = Vec::new();
            let mut siblings = Vec::new();
            let mut owned_results = Vec::new();
            for item in items {
                match &item {
                    ConversationItem::ToolResult(result)
                        if tool_call_ids.contains(&result.tool_call_id) =>
                    {
                        owned_results.push(item);
                    }
                    ConversationItem::ToolResult(_) => previous_results.push(item),
                    _ => siblings.push(item),
                }
            }
            items = previous_results;
            items.extend(siblings);
            items.push(assistant);
            items.extend(owned_results);
        } else if items
            .iter()
            .any(|item| matches!(item, ConversationItem::ToolResult(_)))
        {
            // With no new function calls, every captured result belongs to an
            // earlier owner. Put those results before this response's semantic
            // outputs, which may project to generic assistant messages.
            let (previous_results, siblings): (Vec<_>, Vec<_>) = items
                .into_iter()
                .partition(|item| matches!(item, ConversationItem::ToolResult(_)));
            items = previous_results;
            items.extend(siblings);
        }
    } else {
        items.push(ConversationItem::Assistant(AssistantItem {
            content: std::sync::Arc::<str>::from(content),
            response_item_id,
            tool_calls,
            model_id: Some(model_id),
            model_fingerprint,
            reasoning_effort,
        }));
    }
    if let Some(manifest) = response_manifest {
        items.insert(
            0,
            ConversationItem::Provider(crate::ProviderItem::response_output_metadata(manifest)),
        );
    }
    Ok(items)
}

fn function_call_output_to_durable(
    item: &rs::FunctionCallOutputItemParam,
    require_exact: bool,
) -> std::result::Result<(String, Vec<ContentPart>), String> {
    match &item.output {
        rs::FunctionCallOutput::Text(text) => Ok((text.clone(), Vec::new())),
        rs::FunctionCallOutput::Content(parts) => {
            let mut text = None;
            let mut images = Vec::new();
            for (index, part) in parts.iter().enumerate() {
                match part {
                    rs::InputContent::InputText(value) if text.is_none() && index == 0 => {
                        text = Some(value.text.clone());
                    }
                    rs::InputContent::InputImage(value) => {
                        let Some(url) = value.image_url.as_ref().or(value.file_id.as_ref()) else {
                            return Err(
                                "function output image has no replayable URL or file id".into()
                            );
                        };
                        images.push(ContentPart::Image {
                            url: std::sync::Arc::<str>::from(url.clone()),
                        });
                    }
                    _ if require_exact => {
                        return Err("Codex function output has unsupported content ordering".into());
                    }
                    _ => {}
                }
            }
            Ok((text.unwrap_or_default(), images))
        }
    }
}

impl DecodedResponse {
    pub fn into_conversation_items(self) -> std::result::Result<Vec<ConversationItem>, String> {
        captured_response_to_conversation_items(self.response, self.output)
    }
}

/// Convert a conversation request into a Codex-compatible CreateResponse.
pub fn conversation_request_to_codex_create_response(
    req: &ConversationRequest,
) -> rs::CreateResponse {
    let mut created: rs::CreateResponse = req.into();
    created.input = crate::conversation::build_codex_responses_input(req);
    normalize_create_response_for_codex(&mut created);
    // Re-apply effort after normalize (normalize may drop empty reasoning).
    if let Some(effort) = req.reasoning_effort {
        apply_reasoning_effort(&mut created, effort);
    }
    created
}

fn apply_reasoning_effort(req: &mut rs::CreateResponse, effort: ReasoningEffort) {
    req.reasoning = Some(rs::Reasoning {
        effort: Some(effort.to_responses_api()),
        summary: Some(rs::ReasoningSummary::Concise),
    });
}

/// Post-process a CreateResponse for Codex wire rules.
pub fn normalize_create_response_for_codex(req: &mut rs::CreateResponse) {
    // Codex requires streaming; non-stream requests 400. It rejects these
    // optional sampling/output-limit controls, including values inherited
    // from client defaults, so they must be removed at the wire boundary.
    req.stream = Some(true);
    req.store = Some(false);
    req.temperature = None;
    req.top_p = None;
    req.max_output_tokens = None;

    // System messages in input → instructions.
    if let rs::InputParam::Items(items) = &mut req.input {
        let mut kept = Vec::new();
        let mut instr = req.instructions.take().unwrap_or_default();
        for item in items.drain(..) {
            match &item {
                rs::InputItem::EasyMessage(m) if matches!(m.role, rs::Role::System) => {
                    let text = match &m.content {
                        rs::EasyInputContent::Text(t) => t.clone(),
                        _ => String::new(),
                    };
                    if !text.trim().is_empty() {
                        if !instr.is_empty() {
                            instr.push_str("\n\n");
                        }
                        instr.push_str(&text);
                    }
                }
                _ => kept.push(item),
            }
        }
        *items = kept;
        if !instr.is_empty() {
            req.instructions = Some(instr);
        }
    }

    if let Some(ref mut reasoning) = req.reasoning {
        if reasoning.effort.is_none() && reasoning.summary.is_none() {
            req.reasoning = None;
        }
    }

    // Drop include entries Codex may not care about beyond encrypted content.
    // Keep ReasoningEncryptedContent when present for multi-turn continuity.
    if req.include.is_none() {
        req.include = Some(vec![rs::IncludeEnum::ReasoningEncryptedContent]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_item_metadata_leaves_standard_responses_input_untouched() {
        let mut body = serde_json::json!({"input": "hi"});

        patch_response_item_metadata_passthrough(&mut body, &[]).unwrap();

        assert_eq!(body["input"], "hi");
    }
    use crate::{ConversationItem, ConversationRequest};

    #[test]
    fn detects_codex_url() {
        assert!(is_codex_backend_url(CODEX_BACKEND_BASE_URL));
        assert!(is_codex_backend_url(
            "https://chatgpt.com/backend-api/codex/"
        ));
        assert!(!is_codex_backend_url("https://api.openai.com/v1"));
        assert!(!is_codex_backend_url("https://cli-chat-proxy.grok.com/v1"));
        assert!(!is_codex_backend_url(
            "https://chatgpt.com.evil.example/backend-api/codex"
        ));
        assert!(!is_codex_backend_url(
            "https://proxy.example/https://chatgpt.com/backend-api/codex"
        ));
    }

    #[test]
    fn system_becomes_instructions_not_input() {
        let req = ConversationRequest {
            items: vec![
                ConversationItem::system("You are helpful."),
                ConversationItem::user("hi"),
            ],
            model: Some("gpt-5.6-sol".into()),
            reasoning_effort: Some(ReasoningEffort::Low),
            ..Default::default()
        };
        let created = conversation_request_to_codex_create_response(&req);
        assert_eq!(created.instructions.as_deref(), Some("You are helpful."));
        match &created.input {
            rs::InputParam::Items(items) => {
                for item in items {
                    if let rs::InputItem::EasyMessage(m) = item {
                        assert!(!matches!(m.role, rs::Role::System));
                    }
                }
            }
            _ => panic!("expected items input"),
        }
        let reasoning = created.reasoning.expect("reasoning");
        assert!(reasoning.summary.is_some());
        assert!(reasoning.effort.is_some());
    }

    #[test]
    fn normalization_strips_unsupported_sampling_and_output_fields() {
        let req = ConversationRequest {
            items: vec![ConversationItem::user("summarize")],
            model: Some("gpt-5.6-sol".into()),
            temperature: Some(1.0),
            top_p: Some(0.9),
            max_output_tokens: Some(4096),
            ..Default::default()
        };
        let created = conversation_request_to_codex_create_response(&req);
        let wire = serde_json::to_value(&created).expect("serialize Codex request");
        for field in ["temperature", "top_p", "max_output_tokens"] {
            assert!(
                wire.get(field).is_none() || wire[field].is_null(),
                "Codex wire request must omit {field}: {wire:#}"
            );
        }
    }

    #[test]
    fn representative_request_keeps_function_tools_and_forces_stream() {
        use crate::ToolSpec;
        let req = ConversationRequest {
            items: vec![
                ConversationItem::system("You are a coding agent."),
                ConversationItem::user("list files"),
            ],
            model: Some("gpt-5.6-sol".into()),
            reasoning_effort: Some(ReasoningEffort::Medium),
            tools: vec![ToolSpec {
                name: "run_terminal_command".into(),
                description: Some("Run a shell command".into()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "command": { "type": "string" } },
                    "required": ["command"]
                }),
            }],
            ..Default::default()
        };
        let created = conversation_request_to_codex_create_response(&req);
        assert_eq!(created.stream, Some(true), "Codex requires stream=true");
        assert_eq!(created.store, Some(false));
        assert_eq!(
            created.instructions.as_deref(),
            Some("You are a coding agent.")
        );
        // System role must not appear in input items.
        if let rs::InputParam::Items(items) = &created.input {
            for item in items {
                if let rs::InputItem::EasyMessage(m) = item {
                    assert!(!matches!(m.role, rs::Role::System));
                }
            }
        } else {
            panic!("expected items input");
        }
        let tools = created
            .tools
            .expect("function tools must survive normalize");
        assert!(
            tools.iter().any(|t| matches!(
                t,
                rs::Tool::Function(f) if f.name == "run_terminal_command"
            )),
            "client function tools must appear on the Codex CreateResponse: {tools:?}"
        );
        let reasoning = created.reasoning.expect("reasoning");
        assert!(reasoning.effort.is_some());
        assert!(
            reasoning.summary.is_some(),
            "Codex wire requests keep reasoning.summary=concise"
        );
    }

    #[test]
    fn native_compact_request_is_strict_and_preserves_realistic_history() {
        use crate::{ToolCall, ToolSpec};

        let reasoning: rs::ReasoningItem = serde_json::from_value(serde_json::json!({
            "type": "reasoning",
            "id": "rs_before_compact",
            "summary": [{"type": "summary_text", "text": "Need to inspect the tree."}],
            "encrypted_content": "encrypted-reasoning-before-compact",
            "status": "completed"
        }))
        .expect("reasoning fixture");
        let req = ConversationRequest {
            items: vec![
                ConversationItem::system("You are a coding agent."),
                ConversationItem::user("Inspect the repository and fix the bug."),
                ConversationItem::Reasoning(reasoning),
                ConversationItem::assistant_tool_calls(vec![ToolCall {
                    id: "call_read".into(),
                    name: "read_file".into(),
                    arguments: r#"{"target_file":"src/lib.rs"}"#.into(),
                }]),
                ConversationItem::tool_result("call_read", "1→pub fn broken() {}"),
                ConversationItem::assistant("The defect is in src/lib.rs."),
            ],
            model: Some("gpt-5.6-sol".into()),
            temperature: Some(0.7),
            top_p: Some(0.8),
            max_output_tokens: Some(8192),
            reasoning_effort: Some(ReasoningEffort::High),
            prompt_cache_key: Some("session-cache-key".into()),
            json_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": false
            })),
            tools: vec![ToolSpec {
                name: "read_file".into(),
                description: Some("Read a repository file".into()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"target_file": {"type": "string"}},
                    "required": ["target_file"]
                }),
            }],
            ..Default::default()
        };

        let compact = conversation_request_to_codex_compact_request_for_origin(&req, None).unwrap();
        let wire = serde_json::to_value(&compact).expect("serialize compact request");
        assert_eq!(wire["model"], "gpt-5.6-sol");
        assert_eq!(wire["instructions"], "You are a coding agent.");
        assert_eq!(wire["prompt_cache_key"], "session-cache-key");
        assert_eq!(wire["parallel_tool_calls"], true);
        assert!(wire["reasoning"].is_object());
        assert!(wire["tools"].is_array());
        assert!(wire["text"].is_object());
        for unsupported in [
            "temperature",
            "top_p",
            "max_output_tokens",
            "stream",
            "stream_options",
            "store",
            "include",
        ] {
            assert!(
                wire.get(unsupported).is_none(),
                "compact wire body must omit {unsupported}: {wire:#}"
            );
        }
        let input = wire["input"].as_array().expect("input array");
        let types = input
            .iter()
            .map(|item| item["type"].as_str().expect("typed input item"))
            .collect::<Vec<_>>();
        assert_eq!(
            types,
            [
                "message",
                "reasoning",
                "function_call",
                "function_call_output",
                "message"
            ]
        );
        assert_eq!(input[1]["id"], "rs_before_compact");
        assert_eq!(
            input[1]["encrypted_content"],
            "encrypted-reasoning-before-compact"
        );
        assert_eq!(input[2]["call_id"], "call_read");
        assert_eq!(input[3]["call_id"], "call_read");
        for message_index in [0, 4] {
            assert!(input[message_index].get("status").is_none());
            assert!(input[message_index].get("phase").is_none());
        }
    }

    #[test]
    fn ordinary_user_messages_stay_default_clean_on_disk_and_both_codex_wires() {
        let ordinary = ConversationItem::user("ordinary");
        let durable = serde_json::to_value(&ordinary).unwrap();
        assert!(durable.get("provider_metadata").is_none());
        assert!(durable.get("status").is_none());
        assert!(durable.get("phase").is_none());

        // A pre-feature persisted user remains readable and serializes without
        // inventing provider metadata.
        let old: ConversationItem = serde_json::from_value(serde_json::json!({
            "type": "user",
            "content": [{"type": "text", "text": "old session"}]
        }))
        .unwrap();
        assert!(
            serde_json::to_value(&old)
                .unwrap()
                .get("provider_metadata")
                .is_none()
        );
        assert!(
            serde_json::from_value::<ConversationItem>(serde_json::json!({
                "type": "user",
                "content": [{"type": "text", "text": "future"}],
                "provider_metadata": {
                    "provider": "codex",
                    "payload": {"status": "completed", "future": true}
                }
            }))
            .is_err(),
            "unknown durable provider fields must fail closed"
        );

        let request = ConversationRequest {
            items: vec![ordinary],
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        let compact =
            serde_json::to_value(conversation_request_to_codex_compact_request(&request).unwrap())
                .unwrap();
        let compact_user = &compact["input"][0];
        assert!(compact_user.get("status").is_none());
        assert!(compact_user.get("phase").is_none());

        let mut responses =
            serde_json::to_value(conversation_request_to_codex_create_response(&request)).unwrap();
        patch_response_message_metadata(
            &mut responses,
            &response_message_metadata(&request).unwrap(),
        )
        .unwrap();
        let responses_user = &responses["input"][0];
        assert!(responses_user.get("status").is_none());
        assert!(responses_user.get("phase").is_none());
    }

    #[test]
    fn native_compact_response_installs_and_reserializes_structured_history() {
        let response: CodexCompactResponse = serde_json::from_value(serde_json::json!({
            "output": [
                {
                    "type": "message",
                    "id": "msg_user_retained",
                    "role": "user",
                    "status": "completed",
                    "phase": "commentary",
                    "content": [{"type": "input_text", "text": "Keep this user turn."}],
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-message-1"}
                },
                {
                    "type": "reasoning",
                    "id": "rs_compacted",
                    "summary": [{"type": "summary_text", "text": "Opaque continuity."}],
                    "encrypted_content": "encrypted-reasoning-after-compact",
                    "status": "completed",
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-reasoning-2"}
                },
                {
                    "type": "message",
                    "id": "msg_retained",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Retained user item."}],
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-message-3"}
                },
                {
                    "type": "compaction",
                    "id": "cmp_123",
                    "encrypted_content": "encrypted-native-compaction",
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-compaction-live"}
                }
            ]
        }))
        .expect("deserialize compact response");

        let replacement = codex_compact_output_to_conversation(
            response.output,
            NativeCompactionCompatibility::codex("gpt-test", Some("acct-test".into())),
        )
        .expect("valid structured replacement history");
        assert_eq!(replacement.len(), 5);
        assert!(matches!(replacement[0], ConversationItem::User(_)));
        assert!(matches!(replacement[1], ConversationItem::Reasoning(_)));
        assert!(matches!(replacement[2], ConversationItem::User(_)));
        assert!(matches!(
            replacement[3],
            ConversationItem::Provider(ref provider) if provider.is_native_compaction_metadata()
        ));
        let ConversationItem::Provider(provider) = &replacement[4] else {
            panic!("expected native compaction item, got {:?}", replacement[4])
        };
        let item = provider
            .as_encrypted_compaction()
            .expect("encrypted native compaction payload");
        assert_eq!(item.id.as_deref(), Some("cmp_123"));
        assert_eq!(item.encrypted_content, "encrypted-native-compaction");

        let persisted = serde_json::to_vec(&replacement).expect("persist replacement history");
        let persisted_json: serde_json::Value =
            serde_json::from_slice(&persisted).expect("inspect durable replacement JSON");
        assert_eq!(
            persisted_json[0]["provider_metadata"],
            serde_json::json!({
                "provider": "codex",
                "payload": {
                    "status": "completed",
                    "phase": "commentary"
                }
            }),
            "durable user owner must retain both provider fields"
        );
        let replacement: Vec<ConversationItem> =
            serde_json::from_slice(&persisted).expect("restart replacement history");
        let restored_identity = crate::native_compaction_compatibility(&replacement)
            .expect("durable descriptor remains well formed")
            .expect("durable descriptor survives restart");
        assert!(restored_identity.matches_origin(
            &crate::ApiBackend::Responses,
            CODEX_BACKEND_BASE_URL,
            "gpt-test",
            Some("acct-test"),
        ));
        let mut mislabeled_as_previous_schema = replacement.clone();
        mislabeled_as_previous_schema
            .iter_mut()
            .find_map(|item| match item {
                ConversationItem::Provider(provider) => {
                    provider.as_native_compaction_metadata_mut()
                }
                _ => None,
            })
            .unwrap()
            .schema_version = NativeCompactionCompatibility::PREVIOUS_SCHEMA_VERSION;
        assert!(
            crate::native_compaction_compatibility(&mislabeled_as_previous_schema).is_err(),
            "schema v2 cannot claim newly represented retained-user fields"
        );

        let next_request = ConversationRequest {
            items: replacement,
            model: Some("gpt-5.6-sol".into()),
            ..Default::default()
        };
        let next =
            conversation_request_to_codex_compact_request_for_origin(&next_request, None).unwrap();
        let wire = serde_json::to_value(next).expect("serialize successor request");
        let input = wire["input"].as_array().expect("input array");
        assert_eq!(input.len(), 4);
        assert_eq!(input[0]["id"], "msg_user_retained");
        assert_eq!(
            input[0]["internal_chat_message_metadata_passthrough"]["turn_id"],
            "turn-message-1"
        );
        assert_eq!(input[0]["status"], "completed");
        assert_eq!(input[0]["phase"], "commentary");
        assert_eq!(input[1]["type"], "reasoning");
        assert_eq!(input[1]["id"], "rs_compacted");
        assert_eq!(
            input[1]["encrypted_content"],
            "encrypted-reasoning-after-compact"
        );
        assert_eq!(
            input[1]["internal_chat_message_metadata_passthrough"]["turn_id"],
            "turn-reasoning-2"
        );
        assert_eq!(input[2]["id"], "msg_retained");
        assert_eq!(
            input[2]["internal_chat_message_metadata_passthrough"]["turn_id"],
            "turn-message-3"
        );
        assert!(input[2].get("status").is_none());
        assert_eq!(input[3]["type"], "compaction");
        assert_eq!(input[3]["id"], "cmp_123");
        assert_eq!(input[3]["encrypted_content"], "encrypted-native-compaction");
        assert_eq!(
            input[3]["internal_chat_message_metadata_passthrough"]["turn_id"],
            "turn-compaction-live"
        );

        // The ordinary successor `/responses` path uses the same installed
        // history. Restore IDs across the pinned async-openai InputMessage gap.
        let created = conversation_request_to_codex_create_response(&next_request);
        let mut successor_wire = serde_json::to_value(created).expect("serialize successor");
        patch_response_message_metadata(
            &mut successor_wire,
            &response_message_metadata(&next_request).unwrap(),
        )
        .unwrap();
        patch_response_item_metadata_passthrough(
            &mut successor_wire,
            &response_item_metadata_passthrough_for_origin(&next_request, None).unwrap(),
        )
        .unwrap();
        let successor_input = successor_wire["input"].as_array().expect("input array");
        assert_eq!(successor_input[0]["id"], "msg_user_retained");
        assert_eq!(successor_input[0]["status"], "completed");
        assert_eq!(successor_input[0]["phase"], "commentary");
        assert_eq!(successor_input[1]["id"], "rs_compacted");
        assert_eq!(successor_input[2]["id"], "msg_retained");
        assert_eq!(successor_input[3]["id"], "cmp_123");
        let replayed_turn_ids = successor_input
            .iter()
            .map(|item| {
                item["internal_chat_message_metadata_passthrough"]["turn_id"]
                    .as_str()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            replayed_turn_ids,
            [
                "turn-message-1",
                "turn-reasoning-2",
                "turn-message-3",
                "turn-compaction-live"
            ]
        );
    }

    #[test]
    fn native_manifest_is_complete_mandatory_and_bound_to_provider_segment() {
        let response: CodexCompactResponse = serde_json::from_value(serde_json::json!({
            "output": [
                {
                    "type": "message",
                    "id": "msg_retained_a",
                    "role": "user",
                    "status": null,
                    "content": [{"type": "input_text", "text": "retained a"}]
                },
                {
                    "type": "message",
                    "id": "msg_retained_b",
                    "role": "user",
                    "phase": "commentary",
                    "content": [{"type": "input_text", "text": "retained b"}]
                },
                {
                    "type": "reasoning",
                    "id": "rs_retained",
                    "summary": [],
                    "encrypted_content": "reasoning-cipher",
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-rs"}
                },
                {
                    "type": "compaction",
                    "id": "cmp_retained",
                    "encrypted_content": "compaction-cipher"
                }
            ]
        }))
        .unwrap();
        let mut replacement = codex_compact_output_to_conversation(
            response.output,
            NativeCompactionCompatibility::codex("gpt-test", Some("acct-test".into())),
        )
        .unwrap();
        replacement.insert(0, ConversationItem::system("canonical system"));

        let manifest = crate::native_compaction_compatibility(&replacement)
            .unwrap()
            .unwrap();
        assert_eq!(manifest.replacement_segment_start, 0);
        assert_eq!(manifest.replacement_segment_items, 4);
        assert_eq!(manifest.item_metadata.len(), 4);
        assert!(
            manifest.item_metadata[0]
                .user_message_provider_metadata
                .is_some()
        );
        assert!(
            manifest.item_metadata[1]
                .user_message_provider_metadata
                .is_some()
        );
        assert!(
            manifest.item_metadata[2]
                .user_message_provider_metadata
                .is_none()
        );
        assert!(
            manifest.item_metadata[3]
                .user_message_provider_metadata
                .is_none()
        );
        assert!(
            manifest.item_metadata[2]
                .internal_chat_message_metadata_passthrough
                .is_some()
        );

        let mut missing_table = serde_json::to_value(&replacement).unwrap();
        let descriptor = missing_table
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|item| item["type"] == "native_compaction_metadata")
            .unwrap();
        descriptor.as_object_mut().unwrap().remove("item_metadata");
        assert!(
            serde_json::from_value::<Vec<ConversationItem>>(missing_table).is_err(),
            "the complete manifest table is never serde-defaulted"
        );

        let mut missing_v3_binding = serde_json::to_value(&replacement).unwrap();
        let descriptor = missing_v3_binding
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|item| item["type"] == "native_compaction_metadata")
            .unwrap();
        descriptor["item_metadata"][0]
            .as_object_mut()
            .unwrap()
            .remove("user_message_provider_metadata");
        let missing_v3_binding: Vec<ConversationItem> =
            serde_json::from_value(missing_v3_binding).unwrap();
        assert!(
            crate::native_compaction_compatibility(&missing_v3_binding).is_err(),
            "schema v3 must fail closed when a retained-user binding key is omitted"
        );

        let mut previous_schema = serde_json::to_value(&replacement).unwrap();
        for item in previous_schema.as_array_mut().unwrap() {
            if item["type"] == "user" {
                item.as_object_mut().unwrap().remove("provider_metadata");
            }
            if item["type"] == "native_compaction_metadata" {
                item["schema_version"] =
                    serde_json::json!(NativeCompactionCompatibility::PREVIOUS_SCHEMA_VERSION);
                for entry in item["item_metadata"].as_array_mut().unwrap() {
                    entry
                        .as_object_mut()
                        .unwrap()
                        .remove("user_message_provider_metadata");
                }
            }
        }
        let previous_schema: Vec<ConversationItem> =
            serde_json::from_value(previous_schema).expect("schema-v2 missing keys deserialize");
        assert!(
            crate::native_compaction_compatibility(&previous_schema).is_ok(),
            "valid schema-v2 history without retained-user replay metadata remains accepted"
        );

        fn manifest_mut(items: &mut [ConversationItem]) -> &mut NativeCompactionCompatibility {
            items
                .iter_mut()
                .find_map(|item| match item {
                    ConversationItem::Provider(provider) => {
                        provider.as_native_compaction_metadata_mut()
                    }
                    _ => None,
                })
                .unwrap()
        }

        let mut removed_user_metadata = replacement.clone();
        let ConversationItem::User(user) = &mut removed_user_metadata[1] else {
            unreachable!()
        };
        user.provider_metadata = None;
        assert!(crate::native_compaction_compatibility(&removed_user_metadata).is_err());

        let mut mutated_user_metadata = replacement.clone();
        let ConversationItem::User(user) = &mut mutated_user_metadata[1] else {
            unreachable!()
        };
        user.provider_metadata = Some(UserMessageProviderMetadata::codex(
            ProviderReplayField::Value(rs::OutputStatus::Completed),
            ProviderReplayField::Missing,
        ));
        assert!(crate::native_compaction_compatibility(&mutated_user_metadata).is_err());

        let mut swapped_user_metadata = replacement.clone();
        let first = match &swapped_user_metadata[1] {
            ConversationItem::User(user) => user.provider_metadata.clone(),
            _ => unreachable!(),
        };
        let second = match &swapped_user_metadata[2] {
            ConversationItem::User(user) => user.provider_metadata.clone(),
            _ => unreachable!(),
        };
        let ConversationItem::User(user) = &mut swapped_user_metadata[1] else {
            unreachable!()
        };
        user.provider_metadata = second;
        let ConversationItem::User(user) = &mut swapped_user_metadata[2] else {
            unreachable!()
        };
        user.provider_metadata = first;
        assert!(crate::native_compaction_compatibility(&swapped_user_metadata).is_err());

        for non_message_index in [2, 3] {
            let mut metadata_on_non_message = replacement.clone();
            let user_binding = manifest_mut(&mut metadata_on_non_message).item_metadata[0]
                .user_message_provider_metadata
                .clone();
            manifest_mut(&mut metadata_on_non_message).item_metadata[non_message_index]
                .user_message_provider_metadata = user_binding;
            assert!(
                crate::native_compaction_compatibility(&metadata_on_non_message).is_err(),
                "user replay fields on non-message manifest entry {non_message_index} must fail"
            );
        }

        let mut missing = replacement.clone();
        manifest_mut(&mut missing).item_metadata.remove(2);
        assert!(crate::native_compaction_compatibility(&missing).is_err());

        let mut extra = replacement.clone();
        let extra_entry = manifest_mut(&mut extra).item_metadata[2].clone();
        manifest_mut(&mut extra).item_metadata.push(extra_entry);
        assert!(crate::native_compaction_compatibility(&extra).is_err());

        let mut duplicate = replacement.clone();
        manifest_mut(&mut duplicate).item_metadata[2].input_index = 0;
        assert!(crate::native_compaction_compatibility(&duplicate).is_err());

        let mut wrong_index = replacement.clone();
        manifest_mut(&mut wrong_index).item_metadata[2].input_index = 3;
        assert!(crate::native_compaction_compatibility(&wrong_index).is_err());

        let mut wrong_kind = replacement.clone();
        manifest_mut(&mut wrong_kind).item_metadata[0].kind = NativeCompactionItemKind::Reasoning;
        assert!(crate::native_compaction_compatibility(&wrong_kind).is_err());

        let mut wrong_id = replacement.clone();
        manifest_mut(&mut wrong_id).item_metadata[3].item_id = Some("cmp-other".into());
        assert!(crate::native_compaction_compatibility(&wrong_id).is_err());

        let mut truncated = replacement.clone();
        truncated.remove(1);
        assert!(crate::native_compaction_compatibility(&truncated).is_err());

        let mut legacy = replacement.clone();
        manifest_mut(&mut legacy).schema_version = 1;
        assert!(crate::native_compaction_compatibility(&legacy).is_err());

        let mut unknown = replacement.clone();
        manifest_mut(&mut unknown).schema_version = 4;
        assert!(crate::native_compaction_compatibility(&unknown).is_err());

        let mut reordered = replacement;
        reordered.swap(1, 2);
        assert!(crate::native_compaction_compatibility(&reordered).is_err());
    }

    #[test]
    fn native_manifest_ignores_later_appends_but_still_binds_replacement_segment() {
        let mut retained = ConversationItem::user("retained");
        if let ConversationItem::User(user) = &mut retained {
            user.response_item_id = Some("msg-retained".into());
        }
        let mut compatibility = NativeCompactionCompatibility::codex("gpt-test", None);
        compatibility.schema_version = NativeCompactionCompatibility::PREVIOUS_SCHEMA_VERSION;
        compatibility.replacement_segment_items = 2;
        compatibility.item_metadata = vec![
            NativeCompactionItemMetadata {
                input_index: 0,
                kind: NativeCompactionItemKind::Message,
                item_id: Some("msg-retained".into()),
                internal_chat_message_metadata_passthrough: None,
                user_message_provider_metadata: None,
            },
            NativeCompactionItemMetadata {
                input_index: 1,
                kind: NativeCompactionItemKind::Compaction,
                item_id: Some("cmp-retained".into()),
                internal_chat_message_metadata_passthrough: None,
                user_message_provider_metadata: None,
            },
        ];
        let replacement = vec![
            ConversationItem::system("canonical system"),
            retained,
            ConversationItem::native_compaction_metadata(compatibility),
            ConversationItem::encrypted_compaction(crate::rs::CompactionSummaryItemParam {
                id: Some("cmp-retained".into()),
                encrypted_content: "cipher".into(),
            }),
        ];

        // Locally/provider-appended successor turns are deliberately outside
        // the two-item provider replacement segment and do not change it.
        let mut appended = replacement.clone();
        appended.push(ConversationItem::assistant("later assistant"));
        appended.push(ConversationItem::user("later user"));
        assert!(crate::native_compaction_compatibility(&appended).is_ok());

        let mut provider_metadata_outside_segment = appended.clone();
        let ConversationItem::User(later_user) =
            provider_metadata_outside_segment.last_mut().unwrap()
        else {
            unreachable!()
        };
        later_user.provider_metadata = Some(UserMessageProviderMetadata::codex(
            ProviderReplayField::Value(rs::OutputStatus::Completed),
            ProviderReplayField::Missing,
        ));
        assert!(
            crate::native_compaction_compatibility(&provider_metadata_outside_segment).is_err(),
            "provider-owned replay fields must remain bound to the immutable native segment"
        );

        let mut removed_inside_segment = replacement.clone();
        removed_inside_segment.remove(1);
        assert!(crate::native_compaction_compatibility(&removed_inside_segment).is_err());

        let mut reordered_inside_segment = replacement;
        reordered_inside_segment.swap(1, 3);
        assert!(crate::native_compaction_compatibility(&reordered_inside_segment).is_err());
    }

    #[test]
    fn native_compact_response_rejects_unknown_structured_items() {
        let error = serde_json::from_value::<CodexCompactResponse>(serde_json::json!({
            "output": [{
                "type": "function_call",
                "id": "fc_unexpected",
                "call_id": "call_unexpected",
                "name": "read_file",
                "arguments": "{}"
            }]
        }))
        .expect_err("unsupported replacement items must not be silently flattened");
        assert!(error.to_string().contains("unknown variant"), "{error}");
    }

    #[test]
    fn native_compact_rejects_official_unreplayable_variants() {
        for kind in ["function_call", "context_compaction", "future_item"] {
            let error = serde_json::from_value::<CodexCompactResponse>(serde_json::json!({
                "output": [{"type": kind, "id": "unsupported"}]
            }))
            .expect_err("unsupported replacement variants must fail before installation");
            assert!(
                error.to_string().contains("unknown variant"),
                "{kind}: {error}"
            );
        }
    }

    #[test]
    fn native_compact_accepts_optional_user_status_and_phase() {
        for (message, expected_status, expected_phase) in [
            (
                serde_json::json!({
                    "type": "message",
                    "id": "msg_status",
                    "role": "user",
                    "status": "completed",
                    "content": [{"type": "input_text", "text": "retained"}]
                }),
                ProviderReplayField::Value(rs::OutputStatus::Completed),
                ProviderReplayField::Missing,
            ),
            (
                serde_json::json!({
                    "type": "message",
                    "id": "msg_phase",
                    "role": "user",
                    "phase": "commentary",
                    "content": [{"type": "input_text", "text": "retained"}]
                }),
                ProviderReplayField::Missing,
                ProviderReplayField::Value("commentary".to_string()),
            ),
            (
                serde_json::json!({
                    "type": "message",
                    "id": "msg_both",
                    "role": "user",
                    "status": "in_progress",
                    "phase": "analysis",
                    "content": [{"type": "input_text", "text": "retained"}]
                }),
                ProviderReplayField::Value(rs::OutputStatus::InProgress),
                ProviderReplayField::Value("analysis".to_string()),
            ),
        ] {
            let response: CodexCompactResponse = serde_json::from_value(serde_json::json!({
                "output": [message, {"type": "compaction", "encrypted_content": "opaque"}]
            }))
            .expect("supported response shape");
            let replacement = codex_compact_output_to_conversation(
                response.output,
                NativeCompactionCompatibility::codex("gpt-test", None),
            )
            .expect("optional retained-user replay fields must be accepted");
            let ConversationItem::User(user) = &replacement[0] else {
                panic!("expected retained user message")
            };
            let metadata = user
                .provider_metadata
                .as_ref()
                .expect("provider metadata must be retained");
            let (status, phase) = metadata.codex_fields();
            assert_eq!(status, &expected_status);
            assert_eq!(phase, &expected_phase);
        }
    }

    #[test]
    fn native_compact_null_presence_survives_durable_reload_and_both_successor_wires() {
        let response: CodexCompactResponse = serde_json::from_value(serde_json::json!({
            "output": [
                {
                    "type": "message", "id": "msg_null_status", "role": "user",
                    "status": null,
                    "content": [{"type": "input_text", "text": "status null only"}]
                },
                {
                    "type": "message", "id": "msg_null_phase", "role": "user",
                    "phase": null,
                    "content": [{"type": "input_text", "text": "phase null only"}]
                },
                {
                    "type": "message", "id": "msg_null_both", "role": "user",
                    "status": null, "phase": null,
                    "content": [{"type": "input_text", "text": "both null"}]
                },
                {"type": "compaction", "id": "cmp_nulls", "encrypted_content": "opaque"}
            ]
        }))
        .unwrap();
        let replacement = codex_compact_output_to_conversation(
            response.output,
            NativeCompactionCompatibility::codex("gpt-null", None),
        )
        .unwrap();

        let durable = serde_json::to_value(&replacement).unwrap();
        for (item_index, status_present, phase_present) in
            [(0, true, false), (1, false, true), (2, true, true)]
        {
            let payload = durable[item_index]["provider_metadata"]["payload"]
                .as_object()
                .unwrap();
            assert_eq!(payload.contains_key("status"), status_present);
            assert_eq!(payload.contains_key("phase"), phase_present);
            if status_present {
                assert!(payload["status"].is_null());
            }
            if phase_present {
                assert!(payload["phase"].is_null());
            }
        }
        let manifest = durable
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["type"] == "native_compaction_metadata")
            .unwrap();
        for (item_index, status_present, phase_present) in
            [(0, true, false), (1, false, true), (2, true, true)]
        {
            let payload =
                manifest["item_metadata"][item_index]["user_message_provider_metadata"]["payload"]
                    .as_object()
                    .unwrap();
            assert_eq!(payload.contains_key("status"), status_present);
            assert_eq!(payload.contains_key("phase"), phase_present);
        }

        let replacement: Vec<ConversationItem> =
            serde_json::from_value(durable).expect("cold reload retained tri-state fields");
        crate::native_compaction_compatibility(&replacement).unwrap();
        let request = ConversationRequest {
            items: replacement,
            model: Some("gpt-null".into()),
            ..Default::default()
        };
        let compact_wire =
            serde_json::to_value(conversation_request_to_codex_compact_request(&request).unwrap())
                .unwrap();
        let mut ordinary_wire =
            serde_json::to_value(conversation_request_to_codex_create_response(&request)).unwrap();
        patch_response_message_metadata(
            &mut ordinary_wire,
            &response_message_metadata(&request).unwrap(),
        )
        .unwrap();
        patch_response_item_metadata_passthrough(
            &mut ordinary_wire,
            &response_item_metadata_passthrough(&request).unwrap(),
        )
        .unwrap();

        for wire in [&compact_wire, &ordinary_wire] {
            let input = wire["input"].as_array().unwrap();
            assert_eq!(input[0]["id"], "msg_null_status");
            assert!(input[0].get("status").unwrap().is_null());
            assert!(input[0].get("phase").is_none());
            assert_eq!(input[1]["id"], "msg_null_phase");
            assert!(input[1].get("status").is_none());
            assert!(input[1].get("phase").unwrap().is_null());
            assert_eq!(input[2]["id"], "msg_null_both");
            assert!(input[2].get("status").unwrap().is_null());
            assert!(input[2].get("phase").unwrap().is_null());
        }

        let unknown_status = serde_json::from_value::<CodexCompactResponse>(serde_json::json!({
            "output": [{
                "type": "message", "role": "user", "status": "future_status",
                "content": [{"type": "input_text", "text": "unknown"}]
            }]
        }));
        assert!(
            unknown_status.is_err(),
            "unknown status must remain fail-closed"
        );
    }

    #[test]
    fn native_compact_still_rejects_images_and_unknown_fields() {
        let response: CodexCompactResponse = serde_json::from_value(serde_json::json!({
            "output": [
                {
                    "type": "message",
                    "id": "msg_image",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "https://example.test/image.png"}]
                },
                {"type": "compaction", "encrypted_content": "opaque"}
            ]
        }))
        .expect("known unsupported content deserializes for an actionable conversion error");
        let error = codex_compact_output_to_conversation(
            response.output,
            NativeCompactionCompatibility::codex("gpt-test", None),
        )
        .expect_err("lossy image projection must fail closed");
        assert!(error.contains("unsupported compact message"), "{error}");

        let error = serde_json::from_value::<CodexCompactResponse>(serde_json::json!({
            "output": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "retained", "future": true}]
            }]
        }))
        .expect_err("unknown content fields must not be silently discarded");
        assert!(error.to_string().contains("unknown field"), "{error}");

        let error = serde_json::from_value::<CodexCompactResponse>(serde_json::json!({
            "output": [{
                "type": "compaction",
                "encrypted_content": "opaque",
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": "turn-known",
                    "future_transport_field": true
                }
            }]
        }))
        .expect_err("unknown passthrough metadata fields must fail closed");
        assert!(error.to_string().contains("unknown field"), "{error}");
    }

    #[test]
    fn native_compact_filters_provider_instructions_before_canonical_reinjection() {
        let response: CodexCompactResponse = serde_json::from_value(serde_json::json!({
            "output": [
                {
                    "type":"message",
                    "role":"developer",
                    "status":"completed",
                    "phase":"commentary",
                    "content":[{"type":"input_image", "image_url":"https://example.test/stale.png"}]
                },
                {"type":"compaction", "id":"cmp", "encrypted_content":"opaque"}
            ]
        }))
        .expect("response shape");
        let replacement = codex_compact_output_to_conversation(
            response.output,
            NativeCompactionCompatibility::codex("gpt-test", Some("acct-test".into())),
        )
        .unwrap();
        assert_eq!(replacement.len(), 2);
        assert!(matches!(
            replacement[0],
            ConversationItem::Provider(ref provider) if provider.is_native_compaction_metadata()
        ));
        assert!(matches!(
            replacement[1],
            ConversationItem::Provider(ref provider) if provider.is_encrypted_compaction()
        ));
    }

    #[test]
    fn native_compact_reasoning_without_id_is_not_installable() {
        let response: CodexCompactResponse = serde_json::from_value(serde_json::json!({
            "output": [{
                "type": "reasoning",
                "summary": [],
                "encrypted_content": "opaque"
            }]
        }))
        .expect("response shape is valid");
        let error = codex_compact_output_to_conversation(
            response.output,
            NativeCompactionCompatibility::codex("gpt-test", Some("acct-test".into())),
        )
        .expect_err("reasoning without an id cannot round-trip");
        assert_eq!(error, "compact reasoning item had no id");
    }

    fn completed_response() -> rs::Response {
        rs::Response {
            background: None,
            billing: None,
            conversation: None,
            created_at: 0,
            completed_at: None,
            error: None,
            id: "resp_metadata".into(),
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            metadata: None,
            model: "gpt-codex".into(),
            object: "response".into(),
            output: vec![],
            parallel_tool_calls: None,
            previous_response_id: None,
            prompt: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            reasoning: None,
            safety_identifier: None,
            service_tier: None,
            status: rs::Status::Completed,
            temperature: None,
            text: None,
            tool_choice: None,
            tools: None,
            top_logprobs: None,
            top_p: None,
            truncation: None,
            usage: None,
        }
    }

    #[test]
    fn empty_completed_rebuild_retains_added_item_metadata() {
        let message = rs::OutputItem::Message(rs::OutputMessage {
            content: vec![rs::OutputMessageContent::OutputText(
                rs::OutputTextContent {
                    text: "retained".into(),
                    annotations: vec![],
                    logprobs: None,
                },
            )],
            id: "msg_added".into(),
            role: rs::AssistantRole::Assistant,
            status: rs::OutputStatus::Completed,
        });
        let mut acc = ResponsesStreamAccumulator::default();
        acc.note_output_item_added(CapturedResponseOutputItem {
            output_index: 0,
            value: CapturedResponseOutputItemValue::Typed(message.clone()),
            internal_chat_message_metadata_passthrough: Some(
                InternalChatMessageMetadataPassthrough {
                    turn_id: Some("turn-added".into()),
                },
            ),
            metadata_origin: crate::ResponseMetadataOrigin::codex(
                CODEX_BACKEND_BASE_URL,
                "gpt-codex",
                None,
            ),
        });
        acc.note_captured_output_item_done(CapturedResponseOutputItem {
            output_index: 0,
            value: CapturedResponseOutputItemValue::Typed(message),
            internal_chat_message_metadata_passthrough: None,
            metadata_origin: None,
        });
        let durable = captured_response_to_conversation_items(
            completed_response(),
            acc.terminal_output(Some(Vec::new())),
        )
        .unwrap();
        assert!(matches!(
            &durable[0],
            ConversationItem::Provider(provider)
                if provider.as_response_output_metadata().is_some_and(|metadata| {
                    metadata.items[0]
                        .internal_chat_message_metadata_passthrough
                        .as_ref()
                        .and_then(|value| value.turn_id.as_deref())
                        == Some("turn-added")
                })
        ));
        let persisted = serde_json::to_string(&durable).unwrap();
        let cold: Vec<ConversationItem> = serde_json::from_str(&persisted).unwrap();
        assert!(matches!(
            cold[0],
            ConversationItem::Provider(ref provider) if provider.is_response_output_metadata()
        ));
    }

    fn captured_item(
        output_index: u32,
        value: serde_json::Value,
        origin: &crate::ResponseMetadataOrigin,
    ) -> CapturedResponseOutputItem {
        let metadata = value
            .get("internal_chat_message_metadata_passthrough")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .unwrap();
        let mut value = value;
        value
            .as_object_mut()
            .unwrap()
            .remove("internal_chat_message_metadata_passthrough");
        let value = if value["type"] == "function_call_output" {
            CapturedResponseOutputItemValue::FunctionCallOutput(
                serde_json::from_value(value).unwrap(),
            )
        } else {
            CapturedResponseOutputItemValue::Typed(serde_json::from_value(value).unwrap())
        };
        CapturedResponseOutputItem {
            output_index,
            value,
            internal_chat_message_metadata_passthrough: metadata,
            metadata_origin: Some(origin.clone()),
        }
    }

    fn ordinary_wire(
        request: &ConversationRequest,
        origin: &crate::ResponseMetadataOrigin,
    ) -> serde_json::Value {
        let mut wire =
            serde_json::to_value(conversation_request_to_codex_create_response(request)).unwrap();
        patch_response_message_metadata(&mut wire, &response_message_metadata(request).unwrap())
            .unwrap();
        let metadata =
            response_item_metadata_passthrough_for_origin(request, Some(origin)).unwrap();
        patch_response_item_metadata_passthrough(&mut wire, &metadata).unwrap();
        wire
    }

    fn input_identity(item: &serde_json::Value) -> (&str, Option<&str>, Option<&str>) {
        (
            item["type"].as_str().unwrap(),
            item.get("id").and_then(serde_json::Value::as_str),
            item.get("call_id").and_then(serde_json::Value::as_str),
        )
    }

    #[test]
    fn interleaved_response_group_round_trips_exactly_on_ordinary_and_compact_wire() {
        let origin =
            crate::ResponseMetadataOrigin::codex(CODEX_BACKEND_BASE_URL, "gpt-codex", None)
                .unwrap();
        let output = vec![
            captured_item(
                0,
                serde_json::json!({
                    "type": "reasoning", "id": "rs_0", "summary": [],
                    "encrypted_content": "cipher-0", "status": "completed",
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-0"}
                }),
                &origin,
            ),
            captured_item(
                1,
                serde_json::json!({
                    "type": "function_call", "id": "fc_1", "call_id": "call_1",
                    "name": "read_file", "arguments": "{}",
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-1"}
                }),
                &origin,
            ),
            captured_item(
                2,
                serde_json::json!({
                    "type": "reasoning", "id": "rs_2", "summary": [],
                    "encrypted_content": "cipher-2", "status": "completed"
                }),
                &origin,
            ),
            captured_item(
                3,
                serde_json::json!({
                    "type": "function_call", "id": "fc_3", "call_id": "call_3",
                    "name": "grep", "arguments": "{}",
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-3"}
                }),
                &origin,
            ),
            captured_item(
                4,
                serde_json::json!({
                    "type": "message", "id": "msg_4", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "done", "annotations": []}],
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-4"}
                }),
                &origin,
            ),
        ];
        let mut response = completed_response();
        response.id = "resp-interleaved".into();
        let durable = captured_response_to_conversation_items(response, output).unwrap();
        let cold: Vec<ConversationItem> =
            serde_json::from_slice(&serde_json::to_vec(&durable).unwrap()).unwrap();
        let request = ConversationRequest {
            items: cold,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };

        let ordinary = ordinary_wire(&request, &origin);
        let compact = serde_json::to_value(
            conversation_request_to_codex_compact_request_for_origin(&request, Some(&origin))
                .unwrap(),
        )
        .unwrap();
        let expected = [
            ("reasoning", Some("rs_0"), None),
            ("function_call", Some("fc_1"), Some("call_1")),
            ("reasoning", Some("rs_2"), None),
            ("function_call", Some("fc_3"), Some("call_3")),
            ("message", Some("msg_4"), None),
        ];
        for wire in [&ordinary, &compact] {
            let input = wire["input"].as_array().unwrap();
            assert_eq!(
                input.iter().map(input_identity).collect::<Vec<_>>(),
                expected
            );
            assert_eq!(
                input[0]["internal_chat_message_metadata_passthrough"]["turn_id"],
                "turn-0"
            );
            assert!(
                input[2]
                    .get("internal_chat_message_metadata_passthrough")
                    .is_none(),
                "nullable manifest entries must still participate in ordering"
            );
            assert_eq!(
                input[4]["internal_chat_message_metadata_passthrough"]["turn_id"],
                "turn-4"
            );
        }
    }

    #[test]
    fn exact_response_keeps_tool_owner_paired_with_later_result() {
        let origin =
            crate::ResponseMetadataOrigin::codex(CODEX_BACKEND_BASE_URL, "gpt-codex", None)
                .unwrap();
        let output = vec![
            captured_item(
                0,
                serde_json::json!({
                    "type": "reasoning", "id": "rs-before", "summary": [],
                    "encrypted_content": "cipher-before", "status": "completed"
                }),
                &origin,
            ),
            captured_item(
                1,
                serde_json::json!({
                    "type": "message", "id": "msg-tool-owner", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "running", "annotations": []}]
                }),
                &origin,
            ),
            captured_item(
                2,
                serde_json::json!({
                    "type": "reasoning", "id": "rs-after", "summary": [],
                    "encrypted_content": "cipher-after", "status": "completed"
                }),
                &origin,
            ),
            captured_item(
                3,
                serde_json::json!({
                    "type": "function_call", "id": "fc-tool", "call_id": "call-tool",
                    "name": "run_terminal_command", "arguments": "{}"
                }),
                &origin,
            ),
        ];
        let mut response = completed_response();
        response.id = "resp-tool-owner".into();
        let mut durable = captured_response_to_conversation_items(response, output).unwrap();

        assert!(matches!(
            durable.as_slice(),
            [
                ConversationItem::Provider(_),
                ConversationItem::Reasoning(_),
                ConversationItem::Reasoning(_),
                ConversationItem::Assistant(_),
            ]
        ));
        let ConversationItem::Assistant(assistant) = durable.last().unwrap() else {
            panic!("tool-owning assistant must end the durable response group");
        };
        assert_eq!(
            assistant.response_item_id.as_deref(),
            Some("msg-tool-owner")
        );
        assert_eq!(assistant.tool_calls[0].id.as_ref(), "call-tool");

        let mut output_response = completed_response();
        output_response.id = "resp-tool-output".into();
        durable.extend(
            captured_response_to_conversation_items(
                output_response,
                vec![
                    captured_item(
                        0,
                        serde_json::json!({
                            "type": "web_search_call", "id": "ws-output-group",
                            "status": "completed",
                            "action": {"type": "search", "query": "status", "sources": []}
                        }),
                        &origin,
                    ),
                    captured_item(
                        1,
                        serde_json::json!({
                            "type": "function_call_output", "id": "fco-tool",
                            "call_id": "call-tool", "output": "exit: 0"
                        }),
                        &origin,
                    ),
                ],
            )
            .unwrap(),
        );
        assert!(matches!(
            durable.as_slice(),
            [
                ConversationItem::Provider(_),
                ConversationItem::Reasoning(_),
                ConversationItem::Reasoning(_),
                ConversationItem::Assistant(_),
                ConversationItem::Provider(_),
                ConversationItem::ToolResult(_),
                ConversationItem::BackendToolCall(_),
                ConversationItem::Assistant(_),
            ]
        ));
        assert!(!crate::has_dangling_tool_calls(&durable));
        assert_eq!(
            crate::repair_dangling_tool_calls(
                &mut durable,
                crate::DanglingToolCallReason::UserCancelled,
            ),
            0,
            "history repair must not synthesize a cancellation inside the response group"
        );

        let cold: Vec<ConversationItem> =
            serde_json::from_slice(&serde_json::to_vec(&durable).unwrap()).unwrap();
        let request = ConversationRequest {
            items: cold,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        let ordinary = ordinary_wire(&request, &origin);
        let compact = serde_json::to_value(
            conversation_request_to_codex_compact_request_for_origin(&request, Some(&origin))
                .unwrap(),
        )
        .unwrap();
        let expected = [
            ("reasoning", Some("rs-before"), None),
            ("message", Some("msg-tool-owner"), None),
            ("reasoning", Some("rs-after"), None),
            ("function_call", Some("fc-tool"), Some("call-tool")),
            ("web_search_call", Some("ws-output-group"), None),
            ("function_call_output", Some("fco-tool"), Some("call-tool")),
        ];
        for wire in [&ordinary, &compact] {
            assert_eq!(
                wire["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(input_identity)
                    .collect::<Vec<_>>(),
                expected
            );
        }

        // Recover histories written before the tool-owning Assistant was moved
        // to the end of its response group. Integrity repair used to insert a
        // synthetic cancellation before the trailing reasoning sibling, then
        // the real result arrived after it.
        let mut poisoned = durable;
        let assistant = poisoned.remove(3);
        poisoned.insert(2, assistant);
        poisoned.insert(
            3,
            ConversationItem::tool_result(
                "call-tool",
                "Tool execution was cancelled by the user (tool `run_terminal_command` was not executed).",
            ),
        );
        assert_eq!(crate::dedup_duplicate_tool_results(&mut poisoned), 1);
        assert!(!crate::has_dangling_tool_calls(&poisoned));
        assert_eq!(
            crate::repair_dangling_tool_calls(
                &mut poisoned,
                crate::DanglingToolCallReason::UserCancelled,
            ),
            0
        );
        let recovered = ConversationRequest {
            items: poisoned,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        assert_eq!(
            ordinary_wire(&recovered, &origin)["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(input_identity)
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn exact_response_places_tool_owner_before_captured_result() {
        let origin =
            crate::ResponseMetadataOrigin::codex(CODEX_BACKEND_BASE_URL, "gpt-codex", None)
                .unwrap();
        let output = vec![
            captured_item(
                0,
                serde_json::json!({
                    "type": "reasoning", "id": "rs-mixed", "summary": [],
                    "encrypted_content": "cipher-mixed", "status": "completed"
                }),
                &origin,
            ),
            captured_item(
                1,
                serde_json::json!({
                    "type": "function_call", "id": "fc-mixed", "call_id": "call-mixed",
                    "name": "read_file", "arguments": "{}"
                }),
                &origin,
            ),
            captured_item(
                2,
                serde_json::json!({
                    "type": "function_call_output", "id": "fco-mixed",
                    "call_id": "call-mixed", "output": "contents"
                }),
                &origin,
            ),
        ];
        let mut response = completed_response();
        response.id = "resp-mixed-call-output".into();
        let mut durable = captured_response_to_conversation_items(response, output).unwrap();

        assert!(matches!(
            durable.as_slice(),
            [
                ConversationItem::Provider(_),
                ConversationItem::Reasoning(_),
                ConversationItem::Assistant(_),
                ConversationItem::ToolResult(_),
            ]
        ));
        assert!(!crate::has_dangling_tool_calls(&durable));
        assert_eq!(
            crate::repair_dangling_tool_calls(
                &mut durable,
                crate::DanglingToolCallReason::UserCancelled,
            ),
            0
        );

        let request = ConversationRequest {
            items: durable,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        let ordinary = ordinary_wire(&request, &origin);
        let compact = serde_json::to_value(
            conversation_request_to_codex_compact_request_for_origin(&request, Some(&origin))
                .unwrap(),
        )
        .unwrap();
        let expected = [
            ("reasoning", Some("rs-mixed"), None),
            ("function_call", Some("fc-mixed"), Some("call-mixed")),
            (
                "function_call_output",
                Some("fco-mixed"),
                Some("call-mixed"),
            ),
        ];
        for wire in [&ordinary, &compact] {
            assert_eq!(
                wire["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(input_identity)
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn exact_mixed_response_keeps_previous_and_new_results_with_their_owners() {
        let origin =
            crate::ResponseMetadataOrigin::codex(CODEX_BACKEND_BASE_URL, "gpt-codex", None)
                .unwrap();
        let mut first_response = completed_response();
        first_response.id = "resp-previous-call".into();
        let mut durable = captured_response_to_conversation_items(
            first_response,
            vec![captured_item(
                0,
                serde_json::json!({
                    "type": "function_call", "id": "fc-previous",
                    "call_id": "call-previous", "name": "read_file", "arguments": "{}"
                }),
                &origin,
            )],
        )
        .unwrap();

        let mut mixed_response = completed_response();
        mixed_response.id = "resp-mixed-owners".into();
        durable.extend(
            captured_response_to_conversation_items(
                mixed_response,
                vec![
                    captured_item(
                        0,
                        serde_json::json!({
                            "type": "function_call", "id": "fc-new",
                            "call_id": "call-new", "name": "grep", "arguments": "{}"
                        }),
                        &origin,
                    ),
                    captured_item(
                        1,
                        serde_json::json!({
                            "type": "function_call_output", "id": "fco-new",
                            "call_id": "call-new", "output": "new result"
                        }),
                        &origin,
                    ),
                    captured_item(
                        2,
                        serde_json::json!({
                            "type": "function_call_output", "id": "fco-previous",
                            "call_id": "call-previous", "output": "previous result"
                        }),
                        &origin,
                    ),
                ],
            )
            .unwrap(),
        );

        assert!(matches!(
            durable.as_slice(),
            [
                ConversationItem::Provider(_),
                ConversationItem::Assistant(_),
                ConversationItem::Provider(_),
                ConversationItem::ToolResult(previous),
                ConversationItem::Assistant(_),
                ConversationItem::ToolResult(new),
            ] if previous.tool_call_id == "call-previous" && new.tool_call_id == "call-new"
        ));
        assert!(!crate::has_dangling_tool_calls(&durable));
        assert_eq!(
            crate::repair_dangling_tool_calls(
                &mut durable,
                crate::DanglingToolCallReason::UserCancelled,
            ),
            0
        );

        let request = ConversationRequest {
            items: durable,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        let expected = [
            ("function_call", Some("fc-previous"), Some("call-previous")),
            ("function_call", Some("fc-new"), Some("call-new")),
            ("function_call_output", Some("fco-new"), Some("call-new")),
            (
                "function_call_output",
                Some("fco-previous"),
                Some("call-previous"),
            ),
        ];
        let ordinary = ordinary_wire(&request, &origin);
        let compact = serde_json::to_value(
            conversation_request_to_codex_compact_request_for_origin(&request, Some(&origin))
                .unwrap(),
        )
        .unwrap();
        for wire in [&ordinary, &compact] {
            assert_eq!(
                wire["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(input_identity)
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn multiple_exact_messages_round_trip_as_distinct_assistants() {
        let origin =
            crate::ResponseMetadataOrigin::codex(CODEX_BACKEND_BASE_URL, "gpt-codex", None)
                .unwrap();
        let output = vec![
            captured_item(
                0,
                serde_json::json!({
                    "type": "reasoning", "id": "rs_multi_0", "summary": [],
                    "encrypted_content": "cipher-0", "status": "completed",
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-0"}
                }),
                &origin,
            ),
            captured_item(
                1,
                serde_json::json!({
                    "type": "message", "id": "msg_multi_1", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "progress", "annotations": []}],
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-1"}
                }),
                &origin,
            ),
            captured_item(
                2,
                serde_json::json!({
                    "type": "function_call", "id": "fc_multi_2", "call_id": "call_multi_2",
                    "name": "read_file", "arguments": "{}",
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-2"}
                }),
                &origin,
            ),
            captured_item(
                3,
                serde_json::json!({
                    "type": "reasoning", "id": "rs_multi_3", "summary": [],
                    "encrypted_content": "cipher-3", "status": "completed"
                }),
                &origin,
            ),
            captured_item(
                4,
                serde_json::json!({
                    "type": "message", "id": "msg_multi_4", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "final", "annotations": []}],
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-4"}
                }),
                &origin,
            ),
        ];
        let mut response = completed_response();
        response.id = "resp-multi-message".into();
        let durable = captured_response_to_conversation_items(response, output).unwrap();
        let assistants = durable
            .iter()
            .filter_map(|item| match item {
                ConversationItem::Assistant(assistant) => Some(assistant),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(assistants.len(), 2);
        assert_eq!(assistants[0].content.as_ref(), "progress");
        assert_eq!(
            assistants[0].response_item_id.as_deref(),
            Some("msg_multi_1")
        );
        assert!(assistants[0].tool_calls.is_empty());
        assert!(assistants[0].model_id.is_none());
        assert_eq!(assistants[1].content.as_ref(), "final");
        assert_eq!(
            assistants[1].response_item_id.as_deref(),
            Some("msg_multi_4")
        );
        assert_eq!(assistants[1].tool_calls.len(), 1);
        assert_eq!(assistants[1].tool_calls[0].id.as_ref(), "call_multi_2");
        assert_eq!(assistants[1].model_id.as_deref(), Some("gpt-codex"));

        let cold: Vec<ConversationItem> =
            serde_json::from_slice(&serde_json::to_vec(&durable).unwrap()).unwrap();
        let request = ConversationRequest {
            items: cold,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        let ordinary = ordinary_wire(&request, &origin);
        let compact = serde_json::to_value(
            conversation_request_to_codex_compact_request_for_origin(&request, Some(&origin))
                .unwrap(),
        )
        .unwrap();
        let expected = [
            ("reasoning", Some("rs_multi_0"), None),
            ("message", Some("msg_multi_1"), None),
            ("function_call", Some("fc_multi_2"), Some("call_multi_2")),
            ("reasoning", Some("rs_multi_3"), None),
            ("message", Some("msg_multi_4"), None),
        ];
        for wire in [&ordinary, &compact] {
            let input = wire["input"].as_array().unwrap();
            assert_eq!(
                input.iter().map(input_identity).collect::<Vec<_>>(),
                expected
            );
            for (index, turn_id) in [(0, "turn-0"), (1, "turn-1"), (2, "turn-2"), (4, "turn-4")] {
                assert_eq!(
                    input[index]["internal_chat_message_metadata_passthrough"]["turn_id"],
                    turn_id
                );
            }
        }
    }

    #[test]
    fn generic_multi_message_conversion_keeps_legacy_collapsing_behavior() {
        let origin =
            crate::ResponseMetadataOrigin::codex(CODEX_BACKEND_BASE_URL, "gpt-codex", None)
                .unwrap();
        let mut output = vec![
            captured_item(
                0,
                serde_json::json!({
                    "type": "message", "id": "msg-generic-0", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "first", "annotations": []}]
                }),
                &origin,
            ),
            captured_item(
                1,
                serde_json::json!({
                    "type": "message", "id": "msg-generic-1", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "second", "annotations": []}]
                }),
                &origin,
            ),
        ];
        for item in &mut output {
            item.metadata_origin = None;
        }
        let durable =
            captured_response_to_conversation_items(completed_response(), output).unwrap();
        assert_eq!(durable.len(), 1);
        let ConversationItem::Assistant(assistant) = &durable[0] else {
            panic!("generic response must retain one collapsed assistant");
        };
        assert_eq!(assistant.content.as_ref(), "first\nsecond");
        assert_eq!(assistant.response_item_id.as_deref(), Some("msg-generic-1"));
    }

    #[test]
    fn duplicate_exact_message_ids_fail_closed() {
        let origin =
            crate::ResponseMetadataOrigin::codex(CODEX_BACKEND_BASE_URL, "gpt-codex", None)
                .unwrap();
        let output = ["first", "second"]
            .into_iter()
            .enumerate()
            .map(|(index, text)| {
                captured_item(
                    index as u32,
                    serde_json::json!({
                        "type": "message", "id": "msg-duplicate", "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]
                    }),
                    &origin,
                )
            })
            .collect();
        let error = captured_response_to_conversation_items(completed_response(), output)
            .expect_err("duplicate provider message ids cannot bind uniquely");
        assert!(error.contains("duplicated"), "{error}");
    }

    #[test]
    fn empty_exact_message_id_fails_closed() {
        let origin =
            crate::ResponseMetadataOrigin::codex(CODEX_BACKEND_BASE_URL, "gpt-codex", None)
                .unwrap();
        let output = vec![captured_item(
            0,
            serde_json::json!({
                "type": "message", "id": "", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}]
            }),
            &origin,
        )];

        let error = captured_response_to_conversation_items(completed_response(), output)
            .expect_err("empty provider message ids cannot bind uniquely");
        assert!(error.contains("id is empty"), "{error}");
    }

    #[test]
    fn empty_identified_message_round_trips_on_ordinary_and_compact_wire() {
        let origin =
            crate::ResponseMetadataOrigin::codex(CODEX_BACKEND_BASE_URL, "gpt-codex", None)
                .unwrap();
        let output = vec![captured_item(
            0,
            serde_json::json!({
                "type": "message", "id": "msg_empty", "role": "assistant",
                "status": "completed",
                "content": []
            }),
            &origin,
        )];
        let durable = captured_response_to_conversation_items(completed_response(), output)
            .expect("identified empty Codex messages are replayable");
        let ConversationItem::Assistant(assistant) = durable
            .iter()
            .find(|item| matches!(item, ConversationItem::Assistant(_)))
            .expect("empty message becomes an assistant")
        else {
            unreachable!();
        };
        assert!(assistant.content.is_empty());
        assert_eq!(assistant.response_item_id.as_deref(), Some("msg_empty"));
        assert!(assistant.tool_calls.is_empty());

        let request = ConversationRequest {
            items: durable,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        let ordinary = ordinary_wire(&request, &origin);
        let compact = serde_json::to_value(
            conversation_request_to_codex_compact_request_for_origin(&request, Some(&origin))
                .unwrap(),
        )
        .unwrap();
        for wire in [&ordinary, &compact] {
            assert_eq!(
                wire["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(input_identity)
                    .collect::<Vec<_>>(),
                [("message", Some("msg_empty"), None)]
            );
        }
        assert_eq!(ordinary["input"][0]["content"], "");
        assert_eq!(compact["input"][0]["content"][0]["type"], "output_text");
        assert_eq!(compact["input"][0]["content"][0]["text"], "");
    }

    #[test]
    fn empty_identified_message_can_own_function_calls() {
        let origin =
            crate::ResponseMetadataOrigin::codex(CODEX_BACKEND_BASE_URL, "gpt-codex", None)
                .unwrap();
        let output = vec![
            captured_item(
                0,
                serde_json::json!({
                    "type": "message", "id": "msg_empty_owner", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "", "annotations": []}]
                }),
                &origin,
            ),
            captured_item(
                1,
                serde_json::json!({
                    "type": "function_call", "id": "fc_empty_owner",
                    "call_id": "call_empty_owner", "name": "todo_write",
                    "arguments": "{}"
                }),
                &origin,
            ),
        ];
        let durable = captured_response_to_conversation_items(completed_response(), output)
            .expect("empty message plus function call is replayable");
        let assistant = durable
            .iter()
            .find_map(|item| match item {
                ConversationItem::Assistant(assistant) => Some(assistant),
                _ => None,
            })
            .expect("function call attaches to the empty message");
        assert!(assistant.content.is_empty());
        assert_eq!(
            assistant.response_item_id.as_deref(),
            Some("msg_empty_owner")
        );
        assert_eq!(assistant.tool_calls.len(), 1);
        assert_eq!(assistant.tool_calls[0].id.as_ref(), "call_empty_owner");

        let request = ConversationRequest {
            items: durable,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        let ordinary = ordinary_wire(&request, &origin);
        assert_eq!(
            ordinary["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(input_identity)
                .collect::<Vec<_>>(),
            [
                ("message", Some("msg_empty_owner"), None),
                (
                    "function_call",
                    Some("fc_empty_owner"),
                    Some("call_empty_owner")
                ),
            ]
        );
    }

    #[test]
    fn standard_responses_input_omits_identified_empty_assistant() {
        let origin =
            crate::ResponseMetadataOrigin::codex(CODEX_BACKEND_BASE_URL, "gpt-codex", None)
                .unwrap();
        let output = vec![
            captured_item(
                0,
                serde_json::json!({
                    "type": "message", "id": "msg_empty_std", "role": "assistant",
                    "status": "completed",
                    "content": []
                }),
                &origin,
            ),
            captured_item(
                1,
                serde_json::json!({
                    "type": "function_call", "id": "fc_empty_std",
                    "call_id": "call_empty_std", "name": "read_file",
                    "arguments": "{}"
                }),
                &origin,
            ),
        ];
        let durable = captured_response_to_conversation_items(completed_response(), output)
            .expect("empty identified message is durable");
        let request = ConversationRequest {
            items: durable,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        let standard = serde_json::to_value(rs::CreateResponse::from(&request)).unwrap();
        assert_eq!(
            standard["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(input_identity)
                .collect::<Vec<_>>(),
            [("function_call", None, Some("call_empty_std"))],
            "standard Responses must not emit an anonymous empty assistant message"
        );
    }

    #[test]
    fn tool_only_exact_response_does_not_emit_a_message() {
        let origin =
            crate::ResponseMetadataOrigin::codex(CODEX_BACKEND_BASE_URL, "gpt-codex", None)
                .unwrap();
        let output = vec![captured_item(
            0,
            serde_json::json!({
                "type": "function_call", "id": "fc_tool_only",
                "call_id": "call_tool_only", "name": "read_file",
                "arguments": "{}"
            }),
            &origin,
        )];
        let durable = captured_response_to_conversation_items(completed_response(), output)
            .expect("function-call-only output remains legal");
        let assistant = durable
            .iter()
            .find_map(|item| match item {
                ConversationItem::Assistant(assistant) => Some(assistant),
                _ => None,
            })
            .expect("tool-only turns still synthesize an assistant");
        assert!(assistant.content.is_empty());
        assert!(assistant.response_item_id.is_none());
        assert!(!assistant.projects_message());

        let request = ConversationRequest {
            items: durable,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        let ordinary = ordinary_wire(&request, &origin);
        assert_eq!(
            ordinary["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(input_identity)
                .collect::<Vec<_>>(),
            [(
                "function_call",
                Some("fc_tool_only"),
                Some("call_tool_only")
            )]
        );
    }

    #[test]
    fn response_group_resets_do_not_cross_tool_result_boundaries() {
        let origin =
            crate::ResponseMetadataOrigin::codex(CODEX_BACKEND_BASE_URL, "gpt-codex", None)
                .unwrap();
        let group =
            |response_id: &str, prefix: &str| {
                let mut response = completed_response();
                response.id = response_id.into();
                captured_response_to_conversation_items(
                response,
                vec![
                    captured_item(0, serde_json::json!({
                        "type": "reasoning", "id": format!("rs-{prefix}"), "summary": [],
                        "encrypted_content": format!("cipher-{prefix}"), "status": "completed"
                    }), &origin),
                    captured_item(1, serde_json::json!({
                        "type": "function_call", "id": format!("fc-{prefix}"),
                        "call_id": format!("call-{prefix}"), "name": "read_file",
                        "arguments": "{}"
                    }), &origin),
                ],
            )
            .unwrap()
            };
        let mut items = group("resp-a", "a");
        items.push(ConversationItem::tool_result("call-a", "result-a"));
        items.extend(group("resp-b", "b"));
        let request = ConversationRequest {
            items,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        let wire = ordinary_wire(&request, &origin);
        let input = wire["input"].as_array().unwrap();
        assert_eq!(
            input.iter().map(input_identity).collect::<Vec<_>>(),
            [
                ("reasoning", Some("rs-a"), None),
                ("function_call", Some("fc-a"), Some("call-a")),
                ("function_call_output", None, Some("call-a")),
                ("reasoning", Some("rs-b"), None),
                ("function_call", Some("fc-b"), Some("call-b")),
            ]
        );
    }

    #[test]
    fn corrupt_response_order_manifest_fails_before_wire_use() {
        let origin =
            crate::ResponseMetadataOrigin::codex(CODEX_BACKEND_BASE_URL, "gpt-codex", None)
                .unwrap();
        let mut response = completed_response();
        response.id = "resp-corrupt".into();
        let mut items = captured_response_to_conversation_items(
            response,
            vec![
                captured_item(
                    0,
                    serde_json::json!({
                        "type": "reasoning", "id": "rs-corrupt", "summary": [],
                        "encrypted_content": "cipher", "status": "completed"
                    }),
                    &origin,
                ),
                captured_item(
                    1,
                    serde_json::json!({
                        "type": "function_call", "id": "fc-corrupt",
                        "call_id": "call-corrupt", "name": "read_file", "arguments": "{}"
                    }),
                    &origin,
                ),
            ],
        )
        .unwrap();
        let mut duplicate = items.clone();
        let ConversationItem::Provider(provider) = &mut duplicate[0] else {
            panic!("manifest")
        };
        let manifest = provider
            .as_response_output_metadata_mut()
            .expect("ordinary response manifest");
        manifest.items[1].output_index = 0;
        let duplicate_request = ConversationRequest {
            items: duplicate,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        let error =
            response_item_metadata_passthrough_for_origin(&duplicate_request, Some(&origin))
                .expect_err("duplicate order entry must fail before HTTP");
        assert!(error.contains("duplicate indices"), "{error}");

        let ConversationItem::Provider(provider) = &mut items[0] else {
            panic!("manifest")
        };
        provider
            .as_response_output_metadata_mut()
            .expect("ordinary response manifest")
            .items
            .remove(0);
        let request = ConversationRequest {
            items,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        let error = response_item_metadata_passthrough_for_origin(&request, Some(&origin))
            .expect_err("missing order entry must fail before HTTP");
        assert!(error.contains("manifest length"), "{error}");
    }

    #[test]
    fn native_manifest_and_later_ordinary_metadata_replay_on_both_endpoints() {
        let native_metadata = InternalChatMessageMetadataPassthrough {
            turn_id: Some("turn-native".into()),
        };
        let mut compatibility = NativeCompactionCompatibility::codex("gpt-codex", None);
        compatibility.replacement_segment_items = 1;
        compatibility.item_metadata = vec![NativeCompactionItemMetadata {
            input_index: 0,
            kind: NativeCompactionItemKind::Compaction,
            item_id: Some("cmp_1".into()),
            internal_chat_message_metadata_passthrough: Some(native_metadata),
            user_message_provider_metadata: None,
        }];
        let origin =
            crate::ResponseMetadataOrigin::codex(CODEX_BACKEND_BASE_URL, "gpt-codex", None)
                .unwrap();
        let ordinary = crate::ResponseOutputItemMetadata {
            response_id: "resp-later".into(),
            output_items: 1,
            items: vec![crate::ResponseOutputItemOrder {
                output_index: 0,
                kind: crate::ResponseOutputItemKind::Message,
                item_id: Some("msg_later".into()),
                call_id: None,
                internal_chat_message_metadata_passthrough: Some(
                    InternalChatMessageMetadataPassthrough {
                        turn_id: Some("turn-later".into()),
                    },
                ),
            }],
            origin: Some(origin.clone()),
        };
        let request = ConversationRequest {
            items: vec![
                ConversationItem::system("system"),
                ConversationItem::native_compaction_metadata(compatibility),
                ConversationItem::encrypted_compaction(rs::CompactionSummaryItemParam {
                    id: Some("cmp_1".into()),
                    encrypted_content: "cipher".into(),
                }),
                ConversationItem::response_output_metadata(ordinary),
                ConversationItem::Assistant(AssistantItem {
                    content: "later answer".into(),
                    response_item_id: Some("msg_later".into()),
                    tool_calls: vec![],
                    model_id: Some("gpt-codex".into()),
                    model_fingerprint: None,
                    reasoning_effort: None,
                }),
            ],
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        crate::native_compaction_compatibility(&request.items).unwrap();
        let metadata =
            response_item_metadata_passthrough_for_origin(&request, Some(&origin)).unwrap();
        let mut ordinary_wire =
            serde_json::to_value(conversation_request_to_codex_create_response(&request)).unwrap();
        patch_response_message_metadata(
            &mut ordinary_wire,
            &response_message_metadata(&request).unwrap(),
        )
        .unwrap();
        patch_response_item_metadata_passthrough(&mut ordinary_wire, &metadata).unwrap();
        let ordinary_input = ordinary_wire["input"].as_array().unwrap();
        assert_eq!(ordinary_input[0]["type"], "compaction");
        assert_eq!(
            ordinary_input[0]["internal_chat_message_metadata_passthrough"]["turn_id"],
            "turn-native"
        );
        assert_eq!(ordinary_input[1]["id"], "msg_later");
        assert_eq!(
            ordinary_input[1]["internal_chat_message_metadata_passthrough"]["turn_id"],
            "turn-later"
        );

        let compact_wire = serde_json::to_value(
            conversation_request_to_codex_compact_request_for_origin(&request, Some(&origin))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            compact_wire["input"][0]["internal_chat_message_metadata_passthrough"]["turn_id"],
            "turn-native"
        );
        assert_eq!(
            compact_wire["input"][1]["internal_chat_message_metadata_passthrough"]["turn_id"],
            "turn-later"
        );

        let non_codex = serde_json::to_value(rs::CreateResponse::from(&request)).unwrap();
        assert!(non_codex["input"].as_array().unwrap().iter().all(|item| {
            item.get("internal_chat_message_metadata_passthrough")
                .is_none()
        }));
    }

    #[test]
    fn ordinary_streamed_parallel_calls_and_outputs_keep_exact_captured_compact_body() {
        let origin = crate::ResponseMetadataOrigin::codex(
            CODEX_BACKEND_BASE_URL,
            "gpt-codex",
            Some("acct-a".into()),
        )
        .unwrap();
        let calls = [
            ("fc_provider_1", "call_parallel_1", "turn-call-1"),
            ("fc_provider_2", "call_parallel_2", "turn-call-2"),
        ];
        let mut call_accumulator = ResponsesStreamAccumulator::default();
        for (output_index, (item_id, call_id, turn_id)) in calls.iter().enumerate() {
            let item: rs::OutputItem = serde_json::from_value(serde_json::json!({
                "type": "function_call",
                "id": item_id,
                "call_id": call_id,
                "name": "read_file",
                "arguments": format!(r#"{{"target_file":"{call_id}"}}"#)
            }))
            .unwrap();
            call_accumulator.note_output_item_added(CapturedResponseOutputItem {
                output_index: output_index as u32,
                value: CapturedResponseOutputItemValue::Typed(item.clone()),
                internal_chat_message_metadata_passthrough: Some(
                    InternalChatMessageMetadataPassthrough {
                        turn_id: Some((*turn_id).into()),
                    },
                ),
                metadata_origin: Some(origin.clone()),
            });
            // `added` and `done` are duplicate stream representations of one
            // provider item. The accumulator must retain one ordered owner and
            // copy the metadata envelope from `added` to `done`.
            call_accumulator.note_captured_output_item_done(CapturedResponseOutputItem {
                output_index: output_index as u32,
                value: CapturedResponseOutputItemValue::Typed(item),
                internal_chat_message_metadata_passthrough: None,
                metadata_origin: Some(origin.clone()),
            });
        }
        let mut durable = captured_response_to_conversation_items(
            completed_response(),
            call_accumulator.terminal_output(None),
        )
        .unwrap();

        let mut output_accumulator = ResponsesStreamAccumulator::default();
        for (output_index, (item_id, call_id, turn_id)) in [
            ("fco_provider_1", "call_parallel_1", "turn-output-1"),
            ("fco_provider_2", "call_parallel_2", "turn-output-2"),
        ]
        .into_iter()
        .enumerate()
        {
            let item: rs::FunctionCallOutputItemParam = serde_json::from_value(serde_json::json!({
                "type": "function_call_output",
                "id": item_id,
                "call_id": call_id,
                "output": format!("result-{call_id}")
            }))
            .unwrap();
            output_accumulator.note_output_item_added(CapturedResponseOutputItem {
                output_index: output_index as u32,
                value: CapturedResponseOutputItemValue::FunctionCallOutput(item.clone()),
                internal_chat_message_metadata_passthrough: Some(
                    InternalChatMessageMetadataPassthrough {
                        turn_id: Some(turn_id.into()),
                    },
                ),
                metadata_origin: Some(origin.clone()),
            });
            output_accumulator.note_captured_output_item_done(CapturedResponseOutputItem {
                output_index: output_index as u32,
                value: CapturedResponseOutputItemValue::FunctionCallOutput(item),
                internal_chat_message_metadata_passthrough: None,
                metadata_origin: Some(origin.clone()),
            });
        }
        let mut output_response = completed_response();
        output_response.id = "resp_metadata_outputs".into();
        durable.extend(
            captured_response_to_conversation_items(
                output_response,
                output_accumulator.terminal_output(None),
            )
            .unwrap(),
        );

        let persisted = serde_json::to_vec(&durable).unwrap();
        let cold: Vec<ConversationItem> = serde_json::from_slice(&persisted).unwrap();
        let request = ConversationRequest {
            items: cold,
            model: Some("gpt-codex".into()),
            ..Default::default()
        };
        let compact = serde_json::to_value(
            conversation_request_to_codex_compact_request_for_origin(&request, Some(&origin))
                .unwrap(),
        )
        .unwrap();
        let ordinary = ordinary_wire(&request, &origin);
        for wire in [&ordinary, &compact] {
            let input = wire["input"].as_array().unwrap();
            let exact = input
                .iter()
                .map(|item| {
                    (
                        item["type"].as_str().unwrap(),
                        item["id"].as_str().unwrap(),
                        item["call_id"].as_str().unwrap(),
                        item["internal_chat_message_metadata_passthrough"]["turn_id"]
                            .as_str()
                            .unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(
                exact,
                [
                    (
                        "function_call",
                        "fc_provider_1",
                        "call_parallel_1",
                        "turn-call-1"
                    ),
                    (
                        "function_call",
                        "fc_provider_2",
                        "call_parallel_2",
                        "turn-call-2"
                    ),
                    (
                        "function_call_output",
                        "fco_provider_1",
                        "call_parallel_1",
                        "turn-output-1",
                    ),
                    (
                        "function_call_output",
                        "fco_provider_2",
                        "call_parallel_2",
                        "turn-output-2",
                    ),
                ]
            );
        }
    }

    #[test]
    fn ordinary_metadata_origin_is_exact_portable_and_secret_free() {
        let origin = crate::ResponseMetadataOrigin::codex(
            "https://chat.openai.com/backend-api/codex/bearer-secret/access_token",
            "gpt-codex-a",
            Some("acct-a".into()),
        )
        .unwrap();
        let metadata = crate::ResponseOutputItemMetadata {
            response_id: "resp-provider".into(),
            output_items: 1,
            items: vec![crate::ResponseOutputItemOrder {
                output_index: 0,
                kind: crate::ResponseOutputItemKind::Message,
                item_id: Some("msg-provider".into()),
                call_id: None,
                internal_chat_message_metadata_passthrough: Some(
                    InternalChatMessageMetadataPassthrough {
                        turn_id: Some("turn-provider".into()),
                    },
                ),
            }],
            origin: Some(origin.clone()),
        };
        let request = ConversationRequest {
            items: vec![
                ConversationItem::response_output_metadata(metadata),
                ConversationItem::Assistant(AssistantItem {
                    content: "portable answer".into(),
                    response_item_id: Some("msg-provider".into()),
                    tool_calls: vec![],
                    model_id: Some("gpt-codex-a".into()),
                    model_fingerprint: None,
                    reasoning_effort: None,
                }),
            ],
            model: Some("gpt-codex-a".into()),
            ..Default::default()
        };

        let cold_items: Vec<ConversationItem> =
            serde_json::from_slice(&serde_json::to_vec(&request.items).unwrap()).unwrap();
        let cold = ConversationRequest {
            items: cold_items,
            model: request.model.clone(),
            ..Default::default()
        };
        assert_eq!(
            response_item_metadata_passthrough_for_origin(&cold, Some(&origin))
                .unwrap()
                .len(),
            1,
            "the exact identity must replay after a cold resume"
        );
        let same_compact = serde_json::to_value(
            conversation_request_to_codex_compact_request_for_origin(&cold, Some(&origin)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            same_compact["input"][0]["internal_chat_message_metadata_passthrough"]["turn_id"],
            "turn-provider"
        );
        let different_model = crate::ResponseMetadataOrigin::codex(
            CODEX_BACKEND_BASE_URL,
            "gpt-codex-b",
            Some("acct-a".into()),
        )
        .unwrap();
        let mismatched_compact = serde_json::to_value(
            conversation_request_to_codex_compact_request_for_origin(&cold, Some(&different_model))
                .unwrap(),
        )
        .unwrap();
        assert!(
            mismatched_compact["input"][0]
                .get("internal_chat_message_metadata_passthrough")
                .is_none(),
            "request-time compact defense must not emit stale metadata"
        );
        for incompatible in [
            crate::ResponseMetadataOrigin::codex(
                CODEX_BACKEND_BASE_URL,
                "gpt-codex-b",
                Some("acct-a".into()),
            ),
            crate::ResponseMetadataOrigin::codex(
                CODEX_BACKEND_BASE_URL,
                "gpt-codex-a",
                Some("acct-b".into()),
            ),
        ] {
            assert!(
                response_item_metadata_passthrough_for_origin(&cold, incompatible.as_ref())
                    .unwrap()
                    .is_empty()
            );
        }
        assert!(
            response_item_metadata_passthrough_for_origin(&cold, None)
                .unwrap()
                .is_empty(),
            "non-Codex Responses providers must omit Codex metadata"
        );

        let mut same_identity = cold.items.clone();
        assert!(!crate::strip_incompatible_response_metadata(
            &mut same_identity,
            &crate::ApiBackend::Responses,
            CODEX_BACKEND_BASE_URL,
            "gpt-codex-a",
            Some("acct-a"),
        ));
        assert_eq!(same_identity.len(), cold.items.len());

        let mut non_codex = cold.items.clone();
        assert!(crate::strip_incompatible_response_metadata(
            &mut non_codex,
            &crate::ApiBackend::Responses,
            "https://api.openai.com/v1",
            "gpt-codex-a",
            Some("acct-a"),
        ));
        assert!(matches!(
            non_codex.as_slice(),
            [ConversationItem::Assistant(_)]
        ));

        let mut migrated = cold.items.clone();
        assert!(crate::strip_incompatible_response_metadata(
            &mut migrated,
            &crate::ApiBackend::Responses,
            CODEX_BACKEND_BASE_URL,
            "gpt-codex-b",
            Some("acct-a"),
        ));
        assert!(
            migrated
                .iter()
                .all(|item| {
                    !matches!(item, ConversationItem::Provider(provider) if provider.is_response_output_metadata())
                })
        );
        assert!(matches!(
            migrated.as_slice(),
            [ConversationItem::Assistant(_)]
        ));

        let persisted = serde_json::to_string(&cold.items).unwrap();
        assert!(persisted.contains(CODEX_BACKEND_BASE_URL));
        assert!(!persisted.contains("chat.openai.com"));
        assert!(!persisted.contains("bearer-secret"));
        assert!(!persisted.contains("access_token"));
    }

    #[test]
    fn accumulator_rebuilds_empty_completed_output() {
        let mut acc = ResponsesStreamAccumulator::default();
        acc.note_text_delta("po");
        acc.note_text_delta("ng");
        let rebuilt = acc.rebuilt_output();
        assert_eq!(rebuilt.len(), 1);
        match &rebuilt[0] {
            rs::OutputItem::Message(m) => match &m.content[0] {
                rs::OutputMessageContent::OutputText(t) => assert_eq!(t.text, "pong"),
                _ => panic!("expected output_text"),
            },
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn accumulator_prefers_output_item_done() {
        let mut acc = ResponsesStreamAccumulator::default();
        acc.note_text_delta("ignored-if-items");
        acc.note_output_item_done(
            1,
            rs::OutputItem::Message(rs::OutputMessage {
                content: vec![rs::OutputMessageContent::OutputText(
                    rs::OutputTextContent {
                        text: "from-item".into(),
                        annotations: vec![],
                        logprobs: None,
                    },
                )],
                id: "msg_1".into(),
                role: rs::AssistantRole::Assistant,
                status: rs::OutputStatus::Completed,
            }),
        );
        let rebuilt = acc.rebuilt_output();
        match &rebuilt[0] {
            rs::OutputItem::Message(m) => match &m.content[0] {
                rs::OutputMessageContent::OutputText(t) => assert_eq!(t.text, "from-item"),
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn accumulator_preserves_streamed_text_with_completed_tool_item() {
        let mut acc = ResponsesStreamAccumulator::default();
        acc.note_indexed_text_delta(0, "msg_1", "hello");
        acc.note_output_item_done(
            1,
            rs::OutputItem::FunctionCall(rs::FunctionToolCall {
                arguments: "{}".into(),
                call_id: "call_1".into(),
                name: "lookup".into(),
                id: Some("fc_1".into()),
                status: Some(rs::OutputStatus::Completed),
            }),
        );

        let rebuilt = acc.rebuilt_captured_output();
        assert_eq!(
            rebuilt
                .iter()
                .map(|item| item.output_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        let message = match &rebuilt[0].value {
            CapturedResponseOutputItemValue::Typed(rs::OutputItem::Message(message)) => message,
            other => panic!("expected rebuilt message, got {other:?}"),
        };
        assert_eq!(message.id, "msg_1");
        assert!(matches!(
            &message.content[0],
            rs::OutputMessageContent::OutputText(text) if text.text == "hello"
        ));
        assert!(matches!(
            &rebuilt[1].value,
            CapturedResponseOutputItemValue::Typed(rs::OutputItem::FunctionCall(_))
        ));
    }

    #[test]
    fn accumulator_does_not_duplicate_completed_message() {
        let mut acc = ResponsesStreamAccumulator::default();
        acc.note_indexed_text_delta(0, "msg_done", "hello");
        acc.note_output_item_done(
            0,
            rs::OutputItem::Message(rs::OutputMessage {
                content: vec![rs::OutputMessageContent::OutputText(
                    rs::OutputTextContent {
                        text: "hello".into(),
                        annotations: vec![],
                        logprobs: None,
                    },
                )],
                id: "msg_done".into(),
                role: rs::AssistantRole::Assistant,
                status: rs::OutputStatus::Completed,
            }),
        );

        let rebuilt = acc.rebuilt_captured_output();
        assert_eq!(rebuilt.len(), 1);
        assert_eq!(acc.final_text(), "hello");
    }

    #[test]
    fn accumulator_rebuilt_message_retains_added_identity_and_metadata() {
        let mut acc = ResponsesStreamAccumulator::default();
        acc.note_output_item_added(CapturedResponseOutputItem {
            output_index: 2,
            value: CapturedResponseOutputItemValue::Typed(rs::OutputItem::Message(
                rs::OutputMessage {
                    content: vec![],
                    id: "msg_added".into(),
                    role: rs::AssistantRole::Assistant,
                    status: rs::OutputStatus::InProgress,
                },
            )),
            internal_chat_message_metadata_passthrough: Some(
                InternalChatMessageMetadataPassthrough {
                    turn_id: Some("turn-added".into()),
                },
            ),
            metadata_origin: crate::ResponseMetadataOrigin::codex(
                CODEX_BACKEND_BASE_URL,
                "gpt-codex",
                None,
            ),
        });
        acc.note_indexed_text_delta(2, "msg_added", "recovered");

        let rebuilt = acc.rebuilt_captured_output();
        assert_eq!(rebuilt.len(), 1);
        assert_eq!(rebuilt[0].output_index, 2);
        assert_eq!(rebuilt[0].item_id(), Some("msg_added"));
        assert_eq!(
            rebuilt[0]
                .internal_chat_message_metadata_passthrough
                .as_ref()
                .and_then(|metadata| metadata.turn_id.as_deref()),
            Some("turn-added")
        );
        assert!(rebuilt[0].metadata_origin.is_some());
    }

    #[test]
    fn accumulator_terminal_item_does_not_double_append_deltas() {
        let mut acc = ResponsesStreamAccumulator::default();
        acc.note_text_delta("<summary>");
        acc.note_text_delta("done</summary>");
        acc.note_text_done("<summary>done</summary>");
        acc.note_output_item_done(
            0,
            rs::OutputItem::Message(rs::OutputMessage {
                content: vec![rs::OutputMessageContent::OutputText(
                    rs::OutputTextContent {
                        text: "<summary>done</summary>".into(),
                        annotations: vec![],
                        logprobs: None,
                    },
                )],
                id: "msg_done".into(),
                role: rs::AssistantRole::Assistant,
                status: rs::OutputStatus::Completed,
            }),
        );
        assert_eq!(acc.final_text(), "<summary>done</summary>");
    }

    #[test]
    fn accumulator_concatenates_all_finalized_messages_in_output_order() {
        let message = |id: &str, text: &str| {
            rs::OutputItem::Message(rs::OutputMessage {
                content: vec![rs::OutputMessageContent::OutputText(
                    rs::OutputTextContent {
                        text: text.into(),
                        annotations: vec![],
                        logprobs: None,
                    },
                )],
                id: id.into(),
                role: rs::AssistantRole::Assistant,
                status: rs::OutputStatus::Completed,
            })
        };
        let mut acc = ResponsesStreamAccumulator::default();
        acc.note_text_delta("first second");
        acc.note_output_item_done(2, message("msg_2", "second"));
        acc.note_output_item_done(0, message("msg_0", "first "));
        assert_eq!(acc.final_text(), "first second");
    }

    #[test]
    fn accumulator_uses_output_text_done_when_no_deltas_or_items() {
        let mut acc = ResponsesStreamAccumulator::default();
        acc.note_text_done("terminal-only");
        assert_eq!(acc.final_text(), "terminal-only");
    }

    #[test]
    fn fill_empty_response_output_via_rebuilt() {
        let mut acc = ResponsesStreamAccumulator::default();
        acc.note_text_delta("hi");
        assert!(!acc.rebuilt_output().is_empty());
        // Non-empty output is left alone: simulate by checking rebuild only.
        assert_eq!(
            match &acc.rebuilt_output()[0] {
                rs::OutputItem::Message(m) => match &m.content[0] {
                    rs::OutputMessageContent::OutputText(t) => t.text.as_str(),
                    _ => "",
                },
                _ => "",
            },
            "hi"
        );
    }
}
