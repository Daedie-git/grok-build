//! Sealed provider-owned durable history metadata.
//!
//! Provider payload enums remain private. Neutral callers use semantic
//! constructors/accessors, while conversation persistence uses the crate-private
//! projections to preserve existing JSONL tags and retain exact replay fields.

use serde::{Deserialize, Serialize};

use crate::rs;
use crate::sampling_identity::SamplingIdentity;
use crate::types::ApiBackend;

/// Internal Responses transport metadata that Codex requires clients to replay
/// unchanged with provider-authored compact output items.
///
/// Keep this strongly typed and fail closed when the provider expands the
/// payload: unlike arbitrary response fields, this is the one explicitly
/// approved opaque metadata envelope we persist.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InternalChatMessageMetadataPassthrough {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
}

/// Lossless presence state for a provider replay field.
///
/// Container fields must use [`Self::is_missing`] as their serde skip predicate:
/// a missing key is omitted, [`Self::Null`] is serialized as JSON `null`, and a
/// concrete value is serialized without transformation. During deserialization,
/// the container's serde default produces [`Self::Missing`], while a present
/// JSON `null` reaches this type and produces [`Self::Null`].
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ProviderReplayField<T> {
    #[default]
    Missing,
    Null,
    Value(T),
}

impl<T> ProviderReplayField<T> {
    pub fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
    }
}

impl<T: Serialize> Serialize for ProviderReplayField<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Missing | Self::Null => serializer.serialize_none(),
            Self::Value(value) => value.serialize(serializer),
        }
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for ProviderReplayField<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(match Option::<T>::deserialize(deserializer)? {
            Some(value) => Self::Value(value),
            None => Self::Null,
        })
    }
}

/// Provider-owned replay fields attached to a retained semantic user message.
///
/// The provider taxonomy stays sealed so neutral conversation consumers do not
/// grow Codex-specific branches. Ordinary/local user messages leave this unset;
/// every schema-v3 retained user owns an envelope, even when both fields were
/// missing, so provenance and field presence remain explicit.
#[derive(Debug, Clone, PartialEq)]
pub struct UserMessageProviderMetadata {
    inner: UserMessageProviderMetadataInner,
}

#[derive(Debug, Clone, PartialEq)]
enum UserMessageProviderMetadataInner {
    Codex(CodexUserMessageMetadata),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CodexUserMessageMetadata {
    #[serde(default, skip_serializing_if = "ProviderReplayField::is_missing")]
    status: ProviderReplayField<rs::OutputStatus>,
    #[serde(default, skip_serializing_if = "ProviderReplayField::is_missing")]
    phase: ProviderReplayField<String>,
}

impl UserMessageProviderMetadata {
    /// Retain Codex response-only message fields for exact input replay.
    pub fn codex(
        status: ProviderReplayField<rs::OutputStatus>,
        phase: ProviderReplayField<String>,
    ) -> Self {
        Self {
            inner: UserMessageProviderMetadataInner::Codex(CodexUserMessageMetadata {
                status,
                phase,
            }),
        }
    }

    pub(crate) fn codex_fields(
        &self,
    ) -> (
        &ProviderReplayField<rs::OutputStatus>,
        &ProviderReplayField<String>,
    ) {
        match &self.inner {
            UserMessageProviderMetadataInner::Codex(metadata) => {
                (&metadata.status, &metadata.phase)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum UserMessageProviderName {
    Codex,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct UserMessageProviderMetadataSerialize<'a> {
    provider: UserMessageProviderName,
    payload: &'a CodexUserMessageMetadata,
}

impl Serialize for UserMessageProviderMetadata {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match &self.inner {
            UserMessageProviderMetadataInner::Codex(metadata) => {
                UserMessageProviderMetadataSerialize {
                    provider: UserMessageProviderName::Codex,
                    payload: metadata,
                }
                .serialize(serializer)
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UserMessageProviderMetadataDeserialize {
    provider: UserMessageProviderName,
    payload: CodexUserMessageMetadata,
}

impl<'de> Deserialize<'de> for UserMessageProviderMetadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let UserMessageProviderMetadataDeserialize { provider, payload } =
            UserMessageProviderMetadataDeserialize::deserialize(deserializer)?;
        match provider {
            UserMessageProviderName::Codex => Ok(Self {
                inner: UserMessageProviderMetadataInner::Codex(payload),
            }),
        }
    }
}

/// Codex fields aligned one-for-one with serialized user/assistant messages.
/// This bridges message fields absent from async-openai's `EasyInputMessage`.
#[derive(Debug, Clone, PartialEq)]
pub struct CodexResponseMessageMetadata {
    pub item_id: Option<String>,
    pub status: ProviderReplayField<rs::OutputStatus>,
    pub phase: ProviderReplayField<String>,
}

/// Ordinary Responses item shape used to bind provider metadata to the
/// durable owner and to the exact subsequent input item.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseOutputItemKind {
    Message,
    Reasoning,
    FunctionCall,
    FunctionCallOutput,
    WebSearchCall,
    CustomToolCall,
    CodeInterpreterCall,
    Compaction,
}

/// Exact provider identity under which ordinary Codex Responses transport
/// metadata may be replayed. The canonical URL is a public backend identifier;
/// credentials and bearer tokens must never be stored here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResponseMetadataOrigin {
    pub schema_version: u8,
    pub backend_family: String,
    pub base_url: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chatgpt_account_id: Option<String>,
}

impl ResponseMetadataOrigin {
    pub const SCHEMA_VERSION: u8 = 1;
    pub const CODEX_RESPONSES_FAMILY: &'static str = "codex_responses";

    pub fn codex(
        base_url: &str,
        model: impl Into<String>,
        chatgpt_account_id: Option<String>,
    ) -> Option<Self> {
        let identity =
            SamplingIdentity::new(ApiBackend::Responses, base_url, model, chatgpt_account_id);
        Self::from_sampling_identity(&identity)
    }

    pub fn from_sampling_identity(identity: &SamplingIdentity) -> Option<Self> {
        identity.is_codex_responses().then(|| Self {
            schema_version: Self::SCHEMA_VERSION,
            backend_family: Self::CODEX_RESPONSES_FAMILY.into(),
            base_url: crate::CODEX_BACKEND_BASE_URL.into(),
            model: identity.model.clone(),
            chatgpt_account_id: identity.chatgpt_account_id.clone(),
        })
    }

    pub fn matches_identity(&self, identity: &SamplingIdentity) -> bool {
        self.schema_version == Self::SCHEMA_VERSION
            && self.backend_family == Self::CODEX_RESPONSES_FAMILY
            && self.base_url == crate::CODEX_BACKEND_BASE_URL
            && identity.is_codex_responses()
            && self.model == identity.model
            && self.chatgpt_account_id == identity.chatgpt_account_id
    }

    pub fn matches(
        &self,
        api_backend: &ApiBackend,
        base_url: &str,
        model: &str,
        chatgpt_account_id: Option<&str>,
    ) -> bool {
        self.matches_identity(&SamplingIdentity::new(
            api_backend.clone(),
            base_url,
            model,
            chatgpt_account_id.map(str::to_owned),
        ))
    }
}

/// Complete ordered manifest for one ordinary Codex Responses output group.
///
/// The semantic conversation model intentionally keeps function calls grouped
/// on an `AssistantItem`. This persistence-only sidecar records the original
/// provider order so Codex wire serialization can restore it without changing
/// tool execution or UI grouping. `output_items` is stored independently from
/// the entries so deleted entries fail closed after a cold resume.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResponseOutputItemMetadata {
    /// Defaults retain readability of the short-lived legacy per-item sidecar;
    /// a matching Codex request rejects empty/incomplete group fields.
    #[serde(default)]
    pub response_id: String,
    #[serde(default)]
    pub output_items: u32,
    #[serde(default)]
    pub items: Vec<ResponseOutputItemOrder>,
    /// `None` represents history written before origin scoping. Such metadata
    /// remains readable for migration but is never replayed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<ResponseMetadataOrigin>,
}

/// One position and owner binding in an ordinary Responses output manifest.
/// The passthrough field is a required nullable field: every supported output
/// item has an entry even when the provider supplied no metadata envelope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResponseOutputItemOrder {
    pub output_index: u32,
    pub kind: ResponseOutputItemKind,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub item_id: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub call_id: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
}

/// Request-side binding after durable conversation items have expanded back
/// into the Responses input array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponsesInputItemMetadata {
    pub input_index: usize,
    pub kind: ResponseOutputItemKind,
    pub item_id: Option<String>,
    pub call_id: Option<String>,
    pub internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    /// Present only for ordinary output items. Native compact replacement
    /// bindings use their separate immutable segment manifest.
    pub response_order: Option<ResponsesInputItemOrder>,
}

/// Original position of an ordinary output item within one provider response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponsesInputItemOrder {
    pub response_id: String,
    pub output_index: u32,
    pub output_items: u32,
}

/// Compact-output item kind used to bind passthrough metadata to its exact
/// replay position without weakening the pinned Responses request types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NativeCompactionItemKind {
    Message,
    Reasoning,
    Compaction,
}

/// Complete manifest entry for one provider-authored compact output item.
/// `input_index` is the item's zero-based position in the subsequent Responses
/// `input` array (canonical system instructions are excluded). Entries with no
/// provider metadata are retained so deletion and reordering remain detectable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NativeCompactionItemMetadata {
    pub input_index: usize,
    pub kind: NativeCompactionItemKind,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub item_id: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    /// Required on every schema-v3 message entry and forbidden on every
    /// non-message entry. Schema v2 predates this owner binding, so omission
    /// remains deserializable and is interpreted according to the manifest's
    /// schema version during whole-history validation.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_non_null"
    )]
    pub user_message_provider_metadata: Option<UserMessageProviderMetadata>,
}

fn deserialize_required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

fn deserialize_optional_non_null<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

/// Exact identity under which opaque native Codex history may be replayed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NativeCompactionCompatibility {
    pub schema_version: u8,
    pub backend_family: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chatgpt_account_id: Option<String>,
    /// First Responses input position owned by the compact replacement. This is
    /// zero today, but persisting it makes the segment boundary explicit.
    pub replacement_segment_start: usize,
    /// Number of provider-authored replay items in the replacement segment.
    pub replacement_segment_items: usize,
    /// Complete ordered manifest for the replacement segment, including items
    /// whose passthrough metadata is `None`. This field is intentionally not
    /// serde-defaulted: schema-v2 history without it is invalid.
    pub item_metadata: Vec<NativeCompactionItemMetadata>,
}

impl NativeCompactionCompatibility {
    /// Version 2 added the mandatory passthrough-metadata side table. Version 3
    /// signals that retained user status/phase may live on their durable owner.
    /// New readers still accept exact version-2 manifests; old readers reject
    /// version 3 before they can replay those newly represented fields lossily.
    pub const SCHEMA_VERSION: u8 = 3;
    pub const PREVIOUS_SCHEMA_VERSION: u8 = 2;
    pub const LEGACY_SCHEMA_VERSION: u8 = 1;
    pub const CODEX_RESPONSES_FAMILY: &'static str = "codex_responses";

    pub fn codex(model: impl Into<String>, chatgpt_account_id: Option<String>) -> Self {
        Self {
            schema_version: Self::SCHEMA_VERSION,
            backend_family: Self::CODEX_RESPONSES_FAMILY.into(),
            model: model.into(),
            chatgpt_account_id,
            replacement_segment_start: 0,
            replacement_segment_items: 0,
            item_metadata: Vec::new(),
        }
    }

    pub fn has_supported_replay_schema(&self) -> bool {
        matches!(
            self.schema_version,
            Self::PREVIOUS_SCHEMA_VERSION | Self::SCHEMA_VERSION
        )
    }

    /// Whether a proposed sampler identity can safely replay this opaque
    /// history. Every known identity component is exact-match only.
    pub fn matches_identity(&self, identity: &SamplingIdentity) -> bool {
        self.has_supported_replay_schema()
            && self.backend_family == Self::CODEX_RESPONSES_FAMILY
            && identity.is_codex_responses()
            && self.model == identity.model
            && self.chatgpt_account_id == identity.chatgpt_account_id
    }

    pub fn matches_origin(
        &self,
        api_backend: &ApiBackend,
        base_url: &str,
        model: &str,
        chatgpt_account_id: Option<&str>,
    ) -> bool {
        self.matches_identity(&SamplingIdentity::new(
            api_backend.clone(),
            base_url,
            model,
            chatgpt_account_id.map(str::to_owned),
        ))
    }
}

/// Sealed, strongly typed provider-owned durable history envelope.
///
/// Provider and payload enums are intentionally private. External crates use
/// the typed constructors and accessors below rather than depending on a
/// provider's internal variant taxonomy.
#[derive(Debug, Clone)]
pub struct ProviderItem {
    inner: ProviderItemInner,
}

#[derive(Debug, Clone)]
enum ProviderItemInner {
    Codex(CodexProviderItem),
}

#[derive(Debug, Clone)]
enum CodexProviderItem {
    ResponseOutputMetadata(ResponseOutputItemMetadata),
    NativeCompactionMetadata(NativeCompactionCompatibility),
    EncryptedCompaction(rs::CompactionSummaryItemParam),
}

impl ProviderItem {
    pub fn response_output_metadata(metadata: ResponseOutputItemMetadata) -> Self {
        Self {
            inner: ProviderItemInner::Codex(CodexProviderItem::ResponseOutputMetadata(metadata)),
        }
    }

    pub fn native_compaction_metadata(metadata: NativeCompactionCompatibility) -> Self {
        Self {
            inner: ProviderItemInner::Codex(CodexProviderItem::NativeCompactionMetadata(metadata)),
        }
    }

    pub fn encrypted_compaction(item: rs::CompactionSummaryItemParam) -> Self {
        Self {
            inner: ProviderItemInner::Codex(CodexProviderItem::EncryptedCompaction(item)),
        }
    }

    pub fn as_response_output_metadata(&self) -> Option<&ResponseOutputItemMetadata> {
        match &self.inner {
            ProviderItemInner::Codex(CodexProviderItem::ResponseOutputMetadata(metadata)) => {
                Some(metadata)
            }
            _ => None,
        }
    }

    pub fn as_response_output_metadata_mut(&mut self) -> Option<&mut ResponseOutputItemMetadata> {
        match &mut self.inner {
            ProviderItemInner::Codex(CodexProviderItem::ResponseOutputMetadata(metadata)) => {
                Some(metadata)
            }
            _ => None,
        }
    }

    pub fn as_native_compaction_metadata(&self) -> Option<&NativeCompactionCompatibility> {
        match &self.inner {
            ProviderItemInner::Codex(CodexProviderItem::NativeCompactionMetadata(metadata)) => {
                Some(metadata)
            }
            _ => None,
        }
    }

    pub fn as_native_compaction_metadata_mut(
        &mut self,
    ) -> Option<&mut NativeCompactionCompatibility> {
        match &mut self.inner {
            ProviderItemInner::Codex(CodexProviderItem::NativeCompactionMetadata(metadata)) => {
                Some(metadata)
            }
            _ => None,
        }
    }

    pub fn as_encrypted_compaction(&self) -> Option<&rs::CompactionSummaryItemParam> {
        match &self.inner {
            ProviderItemInner::Codex(CodexProviderItem::EncryptedCompaction(item)) => Some(item),
            _ => None,
        }
    }

    pub fn as_encrypted_compaction_mut(&mut self) -> Option<&mut rs::CompactionSummaryItemParam> {
        match &mut self.inner {
            ProviderItemInner::Codex(CodexProviderItem::EncryptedCompaction(item)) => Some(item),
            _ => None,
        }
    }

    pub fn is_response_output_metadata(&self) -> bool {
        self.as_response_output_metadata().is_some()
    }

    pub fn is_native_compaction_metadata(&self) -> bool {
        self.as_native_compaction_metadata().is_some()
    }

    pub fn is_encrypted_compaction(&self) -> bool {
        self.as_encrypted_compaction().is_some()
    }

    /// True for either half of the native compact descriptor/payload pair.
    pub fn is_native_compaction_item(&self) -> bool {
        self.is_native_compaction_metadata() || self.is_encrypted_compaction()
    }
}

/// Borrowed legacy Codex payload used only by `ConversationItem` persistence.
/// Keeping this projection here prevents the neutral conversation serializer
/// from reaching into provider-private payload enums.
pub(crate) enum LegacyProviderItemRef<'a> {
    ResponseOutputMetadata(&'a ResponseOutputItemMetadata),
    NativeCompactionMetadata(&'a NativeCompactionCompatibility),
    EncryptedCompaction(&'a rs::CompactionSummaryItemParam),
}

impl ProviderItem {
    pub(crate) fn legacy_persistence_projection(&self) -> LegacyProviderItemRef<'_> {
        match &self.inner {
            ProviderItemInner::Codex(CodexProviderItem::ResponseOutputMetadata(metadata)) => {
                LegacyProviderItemRef::ResponseOutputMetadata(metadata)
            }
            ProviderItemInner::Codex(CodexProviderItem::NativeCompactionMetadata(metadata)) => {
                LegacyProviderItemRef::NativeCompactionMetadata(metadata)
            }
            ProviderItemInner::Codex(CodexProviderItem::EncryptedCompaction(item)) => {
                LegacyProviderItemRef::EncryptedCompaction(item)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ProviderName {
    Codex,
}

#[derive(Serialize)]
struct ProviderItemSerialize<'a> {
    provider: ProviderName,
    #[serde(flatten)]
    item: CodexProviderItemSerialize<'a>,
}

#[derive(Serialize)]
#[serde(tag = "kind", content = "payload", rename_all = "snake_case")]
enum CodexProviderItemSerialize<'a> {
    ResponseOutputMetadata(&'a ResponseOutputItemMetadata),
    NativeCompactionMetadata(&'a NativeCompactionCompatibility),
    #[serde(rename = "compaction")]
    EncryptedCompaction(&'a rs::CompactionSummaryItemParam),
}

impl Serialize for ProviderItem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let item = match &self.inner {
            ProviderItemInner::Codex(CodexProviderItem::ResponseOutputMetadata(metadata)) => {
                CodexProviderItemSerialize::ResponseOutputMetadata(metadata)
            }
            ProviderItemInner::Codex(CodexProviderItem::NativeCompactionMetadata(metadata)) => {
                CodexProviderItemSerialize::NativeCompactionMetadata(metadata)
            }
            ProviderItemInner::Codex(CodexProviderItem::EncryptedCompaction(item)) => {
                CodexProviderItemSerialize::EncryptedCompaction(item)
            }
        };
        ProviderItemSerialize {
            provider: ProviderName::Codex,
            item,
        }
        .serialize(serializer)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderItemDeserialize {
    provider: ProviderName,
    #[serde(flatten)]
    item: CodexProviderItemDeserialize,
}

#[derive(Deserialize)]
#[serde(
    tag = "kind",
    content = "payload",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum CodexProviderItemDeserialize {
    ResponseOutputMetadata(ResponseOutputItemMetadata),
    NativeCompactionMetadata(NativeCompactionCompatibility),
    #[serde(rename = "compaction")]
    EncryptedCompaction(rs::CompactionSummaryItemParam),
}

impl<'de> Deserialize<'de> for ProviderItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let ProviderItemDeserialize { provider, item } =
            ProviderItemDeserialize::deserialize(deserializer)?;
        match (provider, item) {
            (
                ProviderName::Codex,
                CodexProviderItemDeserialize::ResponseOutputMetadata(metadata),
            ) => Ok(Self::response_output_metadata(metadata)),
            (
                ProviderName::Codex,
                CodexProviderItemDeserialize::NativeCompactionMetadata(metadata),
            ) => Ok(Self::native_compaction_metadata(metadata)),
            (ProviderName::Codex, CodexProviderItemDeserialize::EncryptedCompaction(item)) => {
                Ok(Self::encrypted_compaction(item))
            }
        }
    }
}
