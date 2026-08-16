//! Conversation types: re-exports the canonical set from
//! `xai_grok_sampling_types` plus grok-shell-specific additions.

use std::collections::HashSet;

pub use xai_grok_sampling_types::conversation::*;

#[cfg(test)]
#[path = "conversation_tests.rs"]
mod tests;

/// Tracing context for conversation requests; satisfies `TraceContext`
/// through its blanket impl. Lives in grok-shell because it references
/// shell-internal config and upload types.
#[derive(Debug, Clone)]
pub struct ConversationRequestTrace {
    pub gcs_config: crate::session::repo_changes::TraceExportConfig,
    #[expect(
        dead_code,
        reason = "retained for snapshot compat; wire when sampler path uploads traces"
    )]
    pub(crate) artifact_tracker: Option<crate::upload::manifest::ArtifactTracker>,
}

/// Fork-safety filter for copied chat history: drops synthetic user messages,
/// then truncates at the last complete turn so the child never sees a partial
/// one. A turn is complete when the Assistant's tool calls are all answered;
/// Reasoning and BackendToolCall items are transparent to the scan.
///
/// NOTE: keep the "complete turn" definition in sync with
/// `count_complete_turns` in `xai-grok-subagent-resolution/src/context.rs`.
pub(crate) fn fork_filter_chat(items: &mut Vec<ConversationItem>) {
    items.retain(|item| match item {
        ConversationItem::User(u) => u.synthetic_reason.is_none(),
        _ => true,
    });

    // A validated native replacement is an atomic completed boundary: it
    // often ends in metadata + encrypted compaction with no assistant, and
    // must not be silently truncated to the system message.
    let mut last_complete_end = native_replacement_boundary(items).unwrap_or(0);

    // Only Assistant advances the ordinary-turn boundary; everything else
    // after the native prefix is transparent.
    let mut i = last_complete_end;
    while i < items.len() {
        match &items[i] {
            ConversationItem::System(_) => {
                last_complete_end = i + 1;
                i += 1;
            }
            ConversationItem::Assistant(asst) => {
                let expected: HashSet<&str> =
                    asst.tool_calls.iter().map(|tc| tc.id.as_ref()).collect();
                let mut found = HashSet::new();
                let mut j = i + 1;
                while j < items.len() {
                    match &items[j] {
                        ConversationItem::ToolResult(tr) => {
                            if expected.contains(tr.tool_call_id.as_str()) {
                                found.insert(tr.tool_call_id.as_str());
                            }
                            j += 1;
                        }
                        ConversationItem::Reasoning(_) | ConversationItem::BackendToolCall(_) => {
                            j += 1;
                        }
                        ConversationItem::Provider(provider)
                            if provider.is_response_output_metadata() =>
                        {
                            j += 1;
                        }
                        _ => break,
                    }
                }
                if found == expected {
                    last_complete_end = j;
                    i = j;
                } else {
                    break; // dangling tool calls -> stop at the last complete boundary
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    items.truncate(last_complete_end);
}

/// End index of a validated native compaction replacement segment, counting
/// the leading system message and the provider-authored prefix. `None` when
/// the history has no native segment. Malformed native history is left
/// untouched here — identity conversion happens at the sampling boundary.
fn native_replacement_boundary(items: &[ConversationItem]) -> Option<usize> {
    let descriptor = xai_grok_sampling_types::native_compaction_compatibility(items).ok()??;
    let mut counted = 0usize;
    let mut end = 0usize;
    for (index, item) in items.iter().enumerate() {
        match item {
            ConversationItem::System(_) => {
                end = index + 1;
            }
            ConversationItem::Provider(provider) if provider.is_native_compaction_metadata() => {
                // Manifest sidecar is adjacent to the segment but not one of
                // the counted replacement items.
                end = index + 1;
            }
            _ => {
                if counted >= descriptor.replacement_segment_items {
                    break;
                }
                counted += 1;
                end = index + 1;
            }
        }
    }
    (counted == descriptor.replacement_segment_items).then_some(end)
}
