//! Pure data types for the xAI sampling / chat-completion API layer.
//!
//! This crate contains the API-agnostic conversation types, chat completion
//! request/response types, streaming types, and error types used across the
//! xAI agent stack.  It intentionally contains **no I/O** (no HTTP clients,
//! no file system access) so it can be depended on by downstream crates
//! (e.g., `xai-chat-state`) without pulling in the full `xai-grok-shell`.

pub mod codex;
pub mod conversation;
pub mod doom_loop;
pub mod error;
pub mod messages;
pub mod provider_capabilities;
pub mod provider_history;
pub mod provider_history_policy;
pub mod sampling_identity;
pub mod serde_helpers;
pub mod tool_overrides;
pub mod types;

pub use self::codex::{
    CHATGPT_ACCOUNT_ID_HEADER, CODEX_ACCOUNT_USAGE_URL, CODEX_BACKEND_BASE_URL,
    CODEX_OAUTH_CLIENT_ID, CODEX_OAUTH_TOKEN_URL, CapturedResponseOutputItem,
    CapturedResponseOutputItemValue, CodexCompactInputItem, CodexCompactMessage,
    CodexCompactMessageContent, CodexCompactOutputItem, CodexCompactRequest, CodexCompactResponse,
    CodexCompactTypedInputItem, CodexStreamAccumulator, DecodedResponse,
    DecodedResponseStreamEvent, ResponsesStreamAccumulator,
    captured_response_to_conversation_items, codex_compact_output_to_conversation,
    conversation_request_to_codex_compact_request,
    conversation_request_to_codex_compact_request_for_origin,
    conversation_request_to_codex_create_response, is_codex_backend_url,
    model_rejects_reasoning_summary, normalize_create_response_for_codex,
    patch_response_item_metadata_passthrough, patch_response_message_metadata,
    response_item_metadata_passthrough, response_item_metadata_passthrough_for_origin,
    response_message_metadata,
};
pub use self::conversation::*;
pub use self::doom_loop::{
    DOOM_LOOP_CHECK_EVENT_TYPE, DOOM_LOOP_CHECK_HEADER, DoomLoopPeek, DoomLoopRecoveryPolicy,
    DoomLoopSignal, DoomLoopSignalKind, is_check_event, peek_doom_loop,
};
pub use self::error::{
    EmptyReason, EmptyResponseContext, ResponseModelMetadata, Result, SamplingError,
    is_context_length_error, status_user_message, structured_stream_error_status,
    user_facing_api_error_message,
};
pub use self::provider_capabilities::{
    AutoCompactSafety, HostedToolPolicy, NativeCompactionKind, ProtocolIdentity,
    ProviderCapabilities, ProviderId, ResolvedProvider, ResponsesWireProtocol, TurnRoutingPolicy,
    capabilities_for_protocol, resolve_provider,
};
pub use self::provider_history::{
    CodexResponseMessageMetadata, InternalChatMessageMetadataPassthrough,
    NativeCompactionCompatibility, NativeCompactionItemKind, NativeCompactionItemMetadata,
    ProviderItem, ProviderReplayField, ResponseMetadataOrigin, ResponseOutputItemKind,
    ResponseOutputItemMetadata, ResponseOutputItemOrder, ResponsesInputItemMetadata,
    ResponsesInputItemOrder, UserMessageProviderMetadata,
};
pub use self::provider_history_policy::{
    SamplingIdentityHistoryError, native_compaction_compatibility,
    prepare_history_for_sampling_identity, strip_incompatible_response_metadata,
    strip_incompatible_response_metadata_for_identity, validate_history_for_sampling_identity,
};
pub use self::sampling_identity::{
    ResolvedSamplingTarget, SamplingIdentity, SamplingTargetMismatch,
    chatgpt_account_id_from_headers,
};
pub use self::tool_overrides::{
    ClearableField, SearchDateBound, SearchDateBoundError, ToolOverrides, ToolOverridesUpdate,
    WebSearchOptions, XSearchOptions,
};
pub use self::types::*;

// Re-export async-openai crate Responses API types under `rs` namespace
pub use async_openai::types::responses as rs;
