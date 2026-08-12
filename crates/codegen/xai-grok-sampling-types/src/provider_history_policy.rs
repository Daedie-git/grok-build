//! Whole-conversation policy for provider-owned durable history.

use crate::conversation::ConversationItem;
use crate::provider_history::{NativeCompactionCompatibility, NativeCompactionItemKind};
use crate::sampling_identity::SamplingIdentity;
use crate::types::ApiBackend;

/// Remove ordinary provider transport sidecars that do not belong to the
/// proposed sampler identity. Semantic messages/tool history remain portable;
/// opaque native compaction descriptors are deliberately untouched and retain
/// their separate fail-closed compatibility checks.
pub fn strip_incompatible_response_metadata_for_identity(
    items: &mut Vec<ConversationItem>,
    identity: &SamplingIdentity,
) -> bool {
    let original_len = items.len();
    items.retain(|item| {
        !matches!(
            item,
            ConversationItem::Provider(provider)
                if provider.as_response_output_metadata().is_some_and(|metadata| {
                    !metadata
                        .origin
                        .as_ref()
                        .is_some_and(|origin| origin.matches_identity(identity))
                })
        )
    });
    items.len() != original_len
}

/// Backward-compatible field-wise wrapper. New state-transition code should
/// construct one [`SamplingIdentity`] and use the identity-based policy.
pub fn strip_incompatible_response_metadata(
    items: &mut Vec<ConversationItem>,
    api_backend: &ApiBackend,
    base_url: &str,
    model: &str,
    chatgpt_account_id: Option<&str>,
) -> bool {
    strip_incompatible_response_metadata_for_identity(
        items,
        &SamplingIdentity::new(
            api_backend.clone(),
            base_url,
            model,
            chatgpt_account_id.map(str::to_owned),
        ),
    )
}

/// Return the single durable native identity, rejecting malformed/legacy
/// opaque history rather than guessing whether it is portable.
pub fn native_compaction_compatibility(
    items: &[ConversationItem],
) -> Result<Option<&NativeCompactionCompatibility>, String> {
    let mut descriptor = None;
    let mut metadata_pending = false;
    for item in items {
        match item {
            ConversationItem::Provider(provider) => {
                if let Some(value) = provider.as_native_compaction_metadata() {
                    if metadata_pending || descriptor.is_some() {
                        return Err(
                            "native compaction history contains conflicting identity metadata"
                                .into(),
                        );
                    }
                    descriptor = Some(value);
                    metadata_pending = true;
                } else if provider.is_encrypted_compaction() {
                    if !metadata_pending {
                        return Err("native compaction history is missing durable identity metadata; resume with the original client or start a new session".into());
                    }
                    metadata_pending = false;
                } else if metadata_pending {
                    return Err(
                        "native compaction identity metadata is not adjacent to its opaque item"
                            .into(),
                    );
                }
            }
            _ if metadata_pending => {
                return Err(
                    "native compaction identity metadata is not adjacent to its opaque item".into(),
                );
            }
            _ => {}
        }
    }
    if metadata_pending {
        return Err("native compaction identity metadata has no opaque item".into());
    }
    if let Some(descriptor) = descriptor {
        if !descriptor.has_supported_replay_schema() {
            return Err(
                "legacy or unknown native compaction history cannot be replayed safely".into(),
            );
        }
        if descriptor.replacement_segment_start != 0 {
            return Err("native compaction manifest has an invalid segment start".into());
        }
        if descriptor.replacement_segment_items == 0 {
            return Err("native compaction manifest has an empty replacement segment".into());
        }
        if descriptor.item_metadata.len() != descriptor.replacement_segment_items {
            return Err("native compaction manifest length does not match its segment".into());
        }

        let mut replay_items = Vec::with_capacity(descriptor.replacement_segment_items);
        for item in items {
            // The manifest binds only the immutable provider-authored
            // replacement prefix. Turns appended after compaction are outside
            // that segment and must never shift or extend this binding.
            if replay_items.len() == descriptor.replacement_segment_items {
                break;
            }
            match item {
                ConversationItem::System(_) => {}
                ConversationItem::User(user) => replay_items.push((
                    NativeCompactionItemKind::Message,
                    user.response_item_id.as_deref(),
                    user.provider_metadata.as_ref(),
                )),
                ConversationItem::Reasoning(reasoning) => replay_items.push((
                    NativeCompactionItemKind::Reasoning,
                    Some(reasoning.id.as_str()),
                    None,
                )),
                ConversationItem::Provider(provider) => {
                    if provider.is_native_compaction_metadata() {
                        continue;
                    }
                    let Some(compaction) = provider.as_encrypted_compaction() else {
                        return Err(
                            "native compaction manifest crosses an unsupported replay item".into(),
                        );
                    };
                    replay_items.push((
                        NativeCompactionItemKind::Compaction,
                        compaction.id.as_deref(),
                        None,
                    ));
                }
                _ => {
                    return Err(
                        "native compaction manifest crosses an unsupported replay item".into(),
                    );
                }
            }
        }
        if replay_items.len() != descriptor.replacement_segment_items {
            return Err("native compaction replacement segment is truncated".into());
        }
        if replay_items
            .iter()
            .filter(|(kind, _, _)| *kind == NativeCompactionItemKind::Compaction)
            .count()
            != 1
        {
            return Err("native compaction segment must contain exactly one opaque item".into());
        }

        let mut seen_indices = std::collections::BTreeSet::new();
        for (expected_index, metadata) in descriptor.item_metadata.iter().enumerate() {
            if !seen_indices.insert(metadata.input_index) {
                return Err("native compaction manifest contains duplicate input indices".into());
            }
            if metadata.input_index != descriptor.replacement_segment_start + expected_index {
                return Err(
                    "native compaction manifest has missing, extra, or unordered indices".into(),
                );
            }
            let Some((kind, item_id, user_provider_metadata)) = replay_items.get(expected_index)
            else {
                return Err("native compaction manifest input index is out of range".into());
            };
            if *kind != metadata.kind || *item_id != metadata.item_id.as_deref() {
                return Err("native compaction manifest does not match its replay item".into());
            }
            match descriptor.schema_version {
                NativeCompactionCompatibility::SCHEMA_VERSION => {
                    if *kind == NativeCompactionItemKind::Message {
                        let Some(owner_metadata) = *user_provider_metadata else {
                            return Err(
                                "schema-v3 retained user is missing provider replay metadata"
                                    .into(),
                            );
                        };
                        if metadata.user_message_provider_metadata.as_ref() != Some(owner_metadata)
                        {
                            return Err(
                                "native compaction manifest does not match retained user replay metadata"
                                    .into(),
                            );
                        }
                    } else if metadata.user_message_provider_metadata.is_some() {
                        return Err(
                            "native compaction manifest binds user replay fields to a non-message entry"
                                .into(),
                        );
                    }
                }
                NativeCompactionCompatibility::PREVIOUS_SCHEMA_VERSION => {
                    if user_provider_metadata.is_some()
                        || metadata.user_message_provider_metadata.is_some()
                    {
                        return Err(
                            "schema-v2 native compaction cannot contain retained user replay metadata"
                                .into(),
                        );
                    }
                }
                _ => unreachable!("supported schema checked above"),
            }
        }
        let all_user_metadata = items
            .iter()
            .filter(|item| {
                matches!(
                    item,
                    ConversationItem::User(user) if user.provider_metadata.is_some()
                )
            })
            .count();
        let replacement_user_metadata = replay_items
            .iter()
            .filter(|(_, _, user_provider_metadata)| user_provider_metadata.is_some())
            .count();
        if all_user_metadata != replacement_user_metadata {
            return Err(
                "retained user provider metadata sits outside the native replacement segment"
                    .into(),
            );
        }
    } else if items.iter().any(|item| {
        matches!(
            item,
            ConversationItem::User(user) if user.provider_metadata.is_some()
        )
    }) {
        return Err("retained user provider metadata has no native compaction manifest".into());
    }
    Ok(descriptor)
}

/// Why history cannot transition to a proposed sampling identity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SamplingIdentityHistoryError {
    #[error(
        "native Codex compaction history has missing or malformed durable identity metadata: {0}; history was not modified"
    )]
    MalformedNativeHistory(String),
    #[error(
        "session contains identity-bound native Codex compaction history; backend, API, model, and ChatGPT account must exactly match its origin"
    )]
    IncompatibleNativeHistory,
}

/// Validate opaque native history against a proposed identity without mutating
/// caller-owned history. Samplers use this as defense in depth at every API
/// entry point.
pub fn validate_history_for_sampling_identity(
    items: &[ConversationItem],
    identity: &SamplingIdentity,
) -> Result<(), SamplingIdentityHistoryError> {
    let compatibility = native_compaction_compatibility(items)
        .map_err(SamplingIdentityHistoryError::MalformedNativeHistory)?;
    if compatibility.is_some_and(|expected| !expected.matches_identity(identity)) {
        return Err(SamplingIdentityHistoryError::IncompatibleNativeHistory);
    }
    Ok(())
}

/// Prepare history for an authoritative sampling-identity transition.
///
/// Native history is fully parsed and identity-validated before this function
/// mutates anything. Only after that succeeds are incompatible ordinary
/// response sidecars stripped. The returned boolean reports whether history
/// changed.
pub fn prepare_history_for_sampling_identity(
    items: &mut Vec<ConversationItem>,
    identity: &SamplingIdentity,
) -> Result<bool, SamplingIdentityHistoryError> {
    validate_history_for_sampling_identity(items, identity)?;
    Ok(strip_incompatible_response_metadata_for_identity(
        items, identity,
    ))
}
