use super::*;
use crate::extensions::notification::{
    CompactionCheckpointFile, CompactionCheckpointInfo,
    FINALIZED_COMPACTION_CHECKPOINT_SCHEMA_VERSION, LEGACY_COMPACTION_CHECKPOINT_SCHEMA_VERSION,
    SessionNotification, SessionUpdate as XaiSessionUpdate,
};
use crate::session::info::Info;
use crate::session::persistence::default_model_id;
use crate::session::storage::{SessionUpdate, StorageAdapter};

fn info() -> Info {
    Info {
        id: acp::SessionId::new("durable-jsonl"),
        cwd: "/test".into(),
    }
}

fn update(info: &Info, text: String) -> SessionUpdate {
    SessionUpdate::Acp(Box::new(acp::SessionNotification::new(
        info.id.clone(),
        acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(acp::ContentBlock::Text(
            acp::TextContent::new(text),
        ))),
    )))
}

fn compaction_fixture(info: &Info) -> (CompactionCheckpointFile, SessionUpdate) {
    let mut compatibility =
        xai_grok_sampling_types::NativeCompactionCompatibility::codex("gpt-recovery", None);
    compatibility.replacement_segment_items = 1;
    compatibility.item_metadata = vec![xai_grok_sampling_types::NativeCompactionItemMetadata {
        input_index: 0,
        kind: xai_grok_sampling_types::NativeCompactionItemKind::Compaction,
        item_id: Some("cmp-recovery".into()),
        internal_chat_message_metadata_passthrough: None,
    }];
    let checkpoint = CompactionCheckpointFile {
        checkpoint_id: "recovery-checkpoint".into(),
        prompt_index_at_compaction: 2,
        compacted_history: vec![
            ConversationItem::system("authoritative checkpoint"),
            ConversationItem::NativeCompactionMetadata(compatibility),
            ConversationItem::Compaction(xai_grok_sampling_types::rs::CompactionSummaryItemParam {
                id: Some("cmp-recovery".into()),
                encrypted_content: "cipher-recovery".into(),
            }),
        ],
        schema_version: 1,
        created_at: "2026-01-01T00:00:00Z".into(),
        original_user_info: None,
        reread_file_paths: vec![],
    };
    let marker = SessionUpdate::Xai(Box::new(SessionNotification {
        session_id: info.id.clone(),
        update: XaiSessionUpdate::CompactionCheckpoint(Box::new(CompactionCheckpointInfo {
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            prompt_index_at_compaction: checkpoint.prompt_index_at_compaction,
            checkpoint_file: "compaction_checkpoints/recovery-checkpoint.json".into(),
            auto_continue: None,
            schema_version: 1,
            created_at: checkpoint.created_at.clone(),
        })),
        meta: None,
    }));
    (checkpoint, marker)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ordinary_and_durable_appends_keep_every_physical_line_parseable() {
    const N: usize = 100;
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let ordinary = adapter.clone();
    let durable = adapter.clone();
    let info_a = info.clone();
    let info_b = info.clone();
    let ordinary = tokio::spawn(async move {
        for index in 0..N {
            ordinary
                .append_update(&info_a, &update(&info_a, format!("ordinary-{index}")))
                .await
                .unwrap();
        }
    });
    let durable = tokio::spawn(async move {
        for index in 0..N {
            durable
                .append_update_durable_commit_aware(
                    &info_b,
                    &update(&info_b, format!("durable-{index}")),
                )
                .await
                .unwrap();
        }
    });
    ordinary.await.unwrap();
    durable.await.unwrap();

    let bytes = std::fs::read(dir.path().join("updates.jsonl")).unwrap();
    let parsed = bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(serde_json::from_slice::<SessionUpdateEnvelope>)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(parsed.len(), N * 2);
}

#[tokio::test]
async fn append_commit_is_reported_when_bookkeeping_fails() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let summary = dir.path().join("summary.json");
    std::fs::remove_file(&summary).unwrap();
    std::fs::create_dir(&summary).unwrap();

    assert!(matches!(
        adapter
            .append_update_durable_commit_aware(&info, &update(&info, "committed".into()))
            .await,
        Err(crate::session::storage::AppendUpdateError::Committed(_))
    ));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("updates.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
}

#[tokio::test]
async fn chat_replacement_commit_is_reported_when_bookkeeping_fails() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let summary = dir.path().join("summary.json");
    std::fs::remove_file(&summary).unwrap();
    std::fs::create_dir(&summary).unwrap();

    assert!(matches!(
        adapter
            .replace_chat_history_commit_aware(
                &info,
                &[ConversationItem::system("cache committed")],
            )
            .await,
        Err(crate::session::storage::ReplaceChatHistoryError::Committed(
            _
        ))
    ));
    let cache = adapter.load_chat_history_from_dir(dir.path()).unwrap();
    assert_eq!(cache[0].text_content(), "cache committed");
}

#[test]
fn directory_barrier_failure_is_retried_even_after_file_exists() {
    let mut attempts = 0;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("updates.jsonl");
    let mut flaky_parent = || {
        attempts += 1;
        if attempts == 1 {
            Err(io::Error::other("directory barrier failed"))
        } else {
            Ok(())
        }
    };
    assert!(
        JsonlStorageAdapter::append_jsonl_line_sync_with(
            &path,
            b"{\"record\":1}\n".to_vec(),
            AppendDurability::Durable,
            std::fs::File::sync_all,
            &mut flaky_parent,
        )
        .is_err()
    );
    JsonlStorageAdapter::append_jsonl_line_sync_with(
        &path,
        b"{\"record\":1}\n".to_vec(),
        AppendDurability::Durable,
        std::fs::File::sync_all,
        &mut flaky_parent,
    )
    .unwrap();
    assert_eq!(attempts, 2);
}

#[test]
fn file_barrier_error_propagates() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("updates.jsonl");
    let error = JsonlStorageAdapter::append_jsonl_line_sync_with(
        &path,
        b"{\"record\":1}\n".to_vec(),
        AppendDurability::Durable,
        |_| Err(io::Error::other("file barrier failed")),
        || Ok(()),
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "file barrier failed");
}

#[test]
fn cwd_switch_retry_after_post_append_barrier_failure_is_already_present() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chat_history.jsonl");
    let item = ConversationItem::working_directory_switch("moved", 3);
    let mut line = serde_json::to_vec(&item).unwrap();
    line.push(b'\n');

    assert!(matches!(
        JsonlStorageAdapter::append_cwd_switch_line_sync_with(
            &path,
            line.clone(),
            3,
            |_| Err(io::Error::other("file barrier failed")),
            || Ok(()),
        ),
        Err(crate::session::storage::AppendCwdSwitchError::Committed {
            acknowledgement: xai_chat_state::StrictAppendAck::Appended,
            ..
        })
    ));
    assert!(matches!(
        JsonlStorageAdapter::append_cwd_switch_line_sync_with(
            &path,
            line,
            3,
            |_| Ok(()),
            || Ok(()),
        )
        .unwrap(),
        xai_chat_state::StrictAppendAck::AlreadyPresent(item)
            if item.text_content() == "moved"
    ));
    assert_eq!(std::fs::read_to_string(path).unwrap().lines().count(), 1);
}

#[tokio::test]
async fn cwd_switch_retry_repairs_bookkeeping_without_duplicate() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let item = ConversationItem::working_directory_switch("moved", 4);

    let summary_path = dir.path().join("summary.json");
    let original_summary = std::fs::read(&summary_path).unwrap();
    std::fs::write(&summary_path, b"invalid summary").unwrap();
    assert!(matches!(
        adapter.append_cwd_switch_commit_aware(&info, &item).await,
        Err(crate::session::storage::AppendCwdSwitchError::Committed {
            acknowledgement: xai_chat_state::StrictAppendAck::Appended,
            ..
        })
    ));
    assert!(matches!(
        adapter
            .append_cwd_switch_commit_aware(
                &info,
                &ConversationItem::working_directory_switch("retry", 4),
            )
            .await,
        Err(crate::session::storage::AppendCwdSwitchError::Committed {
            acknowledgement: xai_chat_state::StrictAppendAck::AlreadyPresent(authoritative),
            ..
        }) if authoritative.text_content() == "moved"
    ));
    std::fs::write(&summary_path, original_summary).unwrap();
    assert_eq!(
        adapter.read_summary_sync(&info).unwrap().num_chat_messages,
        0
    );

    assert!(matches!(
        adapter
            .append_cwd_switch_commit_aware(&info, &item)
            .await
            .unwrap(),
        xai_chat_state::StrictAppendAck::AlreadyPresent(item)
            if item.text_content() == "moved"
    ));
    let summary = adapter.read_summary_sync(&info).unwrap();
    assert_eq!(summary.num_chat_messages, 1);
    assert_eq!(summary.cwd_switch_bookkeeping_generation, 4);

    assert!(matches!(
        adapter
            .append_cwd_switch_commit_aware(&info, &item)
            .await
            .unwrap(),
        xai_chat_state::StrictAppendAck::AlreadyPresent(item)
            if item.text_content() == "moved"
    ));
    let retried = adapter.read_summary_sync(&info).unwrap();
    assert_eq!(retried.num_chat_messages, 1);
    assert_eq!(retried.cwd_switch_bookkeeping_generation, 4);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("chat_history.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
}

#[tokio::test]
async fn cwd_switch_retained_by_history_replacement_is_not_recounted() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let item = ConversationItem::working_directory_switch("retained", 6);

    adapter
        .replace_chat_history(&info, std::slice::from_ref(&item))
        .await
        .unwrap();
    let replaced = adapter.read_summary_sync(&info).unwrap();
    assert_eq!(replaced.num_chat_messages, 1);
    assert_eq!(replaced.cwd_switch_bookkeeping_generation, 6);

    assert!(matches!(
        adapter
            .append_cwd_switch_commit_aware(
                &info,
                &ConversationItem::working_directory_switch("retry", 6),
            )
            .await
            .unwrap(),
        xai_chat_state::StrictAppendAck::AlreadyPresent(authoritative)
            if authoritative.text_content() == "retained"
    ));
    let summary = adapter.read_summary_sync(&info).unwrap();
    assert_eq!(summary.num_chat_messages, 1);
    assert_eq!(summary.cwd_switch_bookkeeping_generation, 6);
    assert_eq!(
        adapter
            .read_chat_history_sync(adapter.chat_file(&info), CHAT_FORMAT_VERSION)
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn cwd_switch_reappend_after_history_replacement_restores_message_count() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let item = ConversationItem::working_directory_switch("moved", 7);

    assert!(matches!(
        adapter
            .append_cwd_switch_commit_aware(&info, &item)
            .await
            .unwrap(),
        xai_chat_state::StrictAppendAck::Appended
    ));
    adapter.replace_chat_history(&info, &[]).await.unwrap();
    let replaced = adapter.read_summary_sync(&info).unwrap();
    assert_eq!(replaced.cwd_switch_bookkeeping_generation, 7);
    assert_eq!(replaced.num_chat_messages, 0);

    assert!(matches!(
        adapter
            .append_cwd_switch_commit_aware(&info, &item)
            .await
            .unwrap(),
        xai_chat_state::StrictAppendAck::Appended
    ));
    let summary = adapter.read_summary_sync(&info).unwrap();
    assert_eq!(summary.cwd_switch_bookkeeping_generation, 7);
    assert_eq!(summary.num_chat_messages, 1);
    assert_eq!(
        adapter
            .read_chat_history_sync(adapter.chat_file(&info), CHAT_FORMAT_VERSION)
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn resume_repairs_stale_cache_from_committed_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let (checkpoint, marker) = compaction_fixture(&info);
    adapter
        .write_compaction_checkpoint(&info, &checkpoint)
        .await
        .unwrap();
    adapter
        .append_update_durable_commit_aware(&info, &marker)
        .await
        .unwrap();
    adapter
        .replace_chat_history(&info, &[ConversationItem::system("stale old cache")])
        .await
        .unwrap();

    let loaded = adapter.load_session_without_updates(&info).await.unwrap();
    assert_eq!(loaded.chat_history.len(), 3);
    assert_eq!(
        loaded.chat_history[0].text_content(),
        "authoritative checkpoint"
    );
    let repaired = adapter.load_chat_history_from_dir(dir.path()).unwrap();
    assert_eq!(repaired[0].text_content(), "authoritative checkpoint");
}

#[tokio::test]
async fn resume_does_not_overwrite_transformed_legacy_local_compaction_cache() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let checkpoint = CompactionCheckpointFile {
        checkpoint_id: "legacy-local-checkpoint".into(),
        prompt_index_at_compaction: 2,
        compacted_history: vec![ConversationItem::system("legacy local checkpoint")],
        schema_version: LEGACY_COMPACTION_CHECKPOINT_SCHEMA_VERSION,
        created_at: "2026-01-01T00:00:00Z".into(),
        original_user_info: None,
        reread_file_paths: vec![],
    };
    let marker = SessionUpdate::Xai(Box::new(SessionNotification {
        session_id: info.id.clone(),
        update: XaiSessionUpdate::CompactionCheckpoint(Box::new(CompactionCheckpointInfo {
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            prompt_index_at_compaction: checkpoint.prompt_index_at_compaction,
            checkpoint_file: "compaction_checkpoints/legacy-local-checkpoint.json".into(),
            auto_continue: None,
            schema_version: LEGACY_COMPACTION_CHECKPOINT_SCHEMA_VERSION,
            created_at: checkpoint.created_at.clone(),
        })),
        meta: None,
    }));
    adapter
        .write_compaction_checkpoint(&info, &checkpoint)
        .await
        .unwrap();
    adapter
        .append_update_durable_commit_aware(&info, &marker)
        .await
        .unwrap();
    adapter
        .replace_chat_history(
            &info,
            &[ConversationItem::system("transformed legacy local cache")],
        )
        .await
        .unwrap();

    let loaded = adapter.load_session_without_updates(&info).await.unwrap();
    assert_eq!(
        loaded.chat_history[0].text_content(),
        "transformed legacy local cache"
    );
}

#[tokio::test]
async fn resume_repairs_stale_local_summary_cache_from_committed_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let checkpoint = CompactionCheckpointFile {
        checkpoint_id: "local-checkpoint".into(),
        prompt_index_at_compaction: 2,
        compacted_history: vec![
            ConversationItem::system("local checkpoint"),
            ConversationItem::user("summary of prior turns"),
        ],
        schema_version: FINALIZED_COMPACTION_CHECKPOINT_SCHEMA_VERSION,
        created_at: "2026-01-01T00:00:00Z".into(),
        original_user_info: None,
        reread_file_paths: vec![],
    };
    let marker = SessionUpdate::Xai(Box::new(SessionNotification {
        session_id: info.id.clone(),
        update: XaiSessionUpdate::CompactionCheckpoint(Box::new(CompactionCheckpointInfo {
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            prompt_index_at_compaction: checkpoint.prompt_index_at_compaction,
            checkpoint_file: "compaction_checkpoints/local-checkpoint.json".into(),
            auto_continue: None,
            schema_version: FINALIZED_COMPACTION_CHECKPOINT_SCHEMA_VERSION,
            created_at: checkpoint.created_at.clone(),
        })),
        meta: None,
    }));
    adapter
        .write_compaction_checkpoint(&info, &checkpoint)
        .await
        .unwrap();
    adapter
        .append_update_durable_commit_aware(&info, &marker)
        .await
        .unwrap();
    adapter
        .replace_chat_history(
            &info,
            &[ConversationItem::system("stale pre-compaction cache")],
        )
        .await
        .unwrap();

    let loaded = adapter.load_session_without_updates(&info).await.unwrap();
    assert_eq!(loaded.chat_history.len(), 2);
    assert_eq!(loaded.chat_history[0].text_content(), "local checkpoint");
    assert_eq!(
        loaded.chat_history[1].text_content(),
        "summary of prior turns"
    );
}

#[tokio::test]
async fn resume_preserves_local_summary_cache_entries_appended_after_checkpoint_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let checkpoint = CompactionCheckpointFile {
        checkpoint_id: "local-prefix".into(),
        prompt_index_at_compaction: 2,
        compacted_history: vec![ConversationItem::system("local checkpoint")],
        schema_version: FINALIZED_COMPACTION_CHECKPOINT_SCHEMA_VERSION,
        created_at: "2026-01-01T00:00:00Z".into(),
        original_user_info: None,
        reread_file_paths: vec![],
    };
    let marker = SessionUpdate::Xai(Box::new(SessionNotification {
        session_id: info.id.clone(),
        update: XaiSessionUpdate::CompactionCheckpoint(Box::new(CompactionCheckpointInfo {
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            prompt_index_at_compaction: checkpoint.prompt_index_at_compaction,
            checkpoint_file: "compaction_checkpoints/local-prefix.json".into(),
            auto_continue: None,
            schema_version: FINALIZED_COMPACTION_CHECKPOINT_SCHEMA_VERSION,
            created_at: checkpoint.created_at.clone(),
        })),
        meta: None,
    }));
    adapter
        .write_compaction_checkpoint(&info, &checkpoint)
        .await
        .unwrap();
    adapter
        .append_update_durable_commit_aware(&info, &marker)
        .await
        .unwrap();
    let mut cache = checkpoint.compacted_history.clone();
    cache.push(ConversationItem::user("later local turn"));
    adapter.replace_chat_history(&info, &cache).await.unwrap();

    let loaded = adapter.load_session_without_updates(&info).await.unwrap();
    assert_eq!(loaded.chat_history.len(), 2);
    assert_eq!(loaded.chat_history[1].text_content(), "later local turn");
}

#[tokio::test]
async fn resume_preserves_cache_entries_appended_after_checkpoint_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let (checkpoint, marker) = compaction_fixture(&info);
    adapter
        .write_compaction_checkpoint(&info, &checkpoint)
        .await
        .unwrap();
    adapter
        .append_update_durable_commit_aware(&info, &marker)
        .await
        .unwrap();
    let mut cache = checkpoint.compacted_history.clone();
    cache.push(ConversationItem::user("later turn"));
    adapter.replace_chat_history(&info, &cache).await.unwrap();

    let loaded = adapter.load_session_without_updates(&info).await.unwrap();
    assert_eq!(loaded.chat_history.len(), 4);
    assert_eq!(loaded.chat_history[3].text_content(), "later turn");
}

#[tokio::test]
async fn compact_metadata_survives_checkpoint_cold_load_and_responses_replay() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let response: xai_grok_sampling_types::CodexCompactResponse = serde_json::from_value(
        serde_json::json!({
            "output": [
                {
                    "type": "message",
                    "id": "msg_cold",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "retained"}],
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-cold-message"}
                },
                {
                    "type": "reasoning",
                    "id": "rs_cold",
                    "summary": [],
                    "encrypted_content": "cipher-reasoning-cold",
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-cold-reasoning"}
                },
                {
                    "type": "compaction",
                    "id": "cmp_cold",
                    "encrypted_content": "cipher-compaction-cold",
                    "internal_chat_message_metadata_passthrough": {"turn_id": "turn-cold-compaction"}
                }
            ]
        }),
    )
    .unwrap();
    let mut replacement = xai_grok_sampling_types::codex_compact_output_to_conversation(
        response.output,
        xai_grok_sampling_types::NativeCompactionCompatibility::codex(
            "gpt-cold",
            Some("acct-cold".into()),
        ),
    )
    .unwrap();
    replacement.insert(0, ConversationItem::system("system"));
    let checkpoint = CompactionCheckpointFile {
        checkpoint_id: "metadata-checkpoint".into(),
        prompt_index_at_compaction: 4,
        compacted_history: replacement.clone(),
        schema_version: 1,
        created_at: "2026-01-01T00:00:00Z".into(),
        original_user_info: None,
        reread_file_paths: vec![],
    };
    let marker = SessionUpdate::Xai(Box::new(SessionNotification {
        session_id: info.id.clone(),
        update: XaiSessionUpdate::CompactionCheckpoint(Box::new(CompactionCheckpointInfo {
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            prompt_index_at_compaction: checkpoint.prompt_index_at_compaction,
            checkpoint_file: "compaction_checkpoints/metadata-checkpoint.json".into(),
            auto_continue: None,
            schema_version: 1,
            created_at: checkpoint.created_at.clone(),
        })),
        meta: None,
    }));
    adapter
        .write_compaction_checkpoint(&info, &checkpoint)
        .await
        .unwrap();
    adapter
        .append_update_durable_commit_aware(&info, &marker)
        .await
        .unwrap();
    let _ = std::fs::remove_file(dir.path().join("chat_history.jsonl"));

    let cold = adapter.load_session_without_updates(&info).await.unwrap();
    assert_eq!(
        serde_json::to_value(&cold.chat_history).unwrap(),
        serde_json::to_value(&replacement).unwrap(),
        "checkpoint and repaired JSONL snapshot are semantically exact"
    );

    let ordinary_origin = xai_grok_sampling_types::ResponseMetadataOrigin::codex(
        xai_grok_sampling_types::CODEX_BACKEND_BASE_URL,
        "gpt-cold",
        Some("acct-cold".into()),
    )
    .unwrap();
    let ordinary_metadata = ConversationItem::ResponseOutputMetadata(
        xai_grok_sampling_types::ResponseOutputItemMetadata {
            response_id: "resp-after-compact".into(),
            output_items: 1,
            items: vec![xai_grok_sampling_types::ResponseOutputItemOrder {
                output_index: 0,
                kind: xai_grok_sampling_types::ResponseOutputItemKind::Message,
                item_id: Some("msg_after_compact".into()),
                call_id: None,
                internal_chat_message_metadata_passthrough: Some(
                    xai_grok_sampling_types::InternalChatMessageMetadataPassthrough {
                        turn_id: Some("turn-after-compact".into()),
                    },
                ),
            }],
            origin: Some(ordinary_origin.clone()),
        },
    );
    let ordinary_message = ConversationItem::Assistant(xai_grok_sampling_types::AssistantItem {
        content: "later ordinary response".into(),
        response_item_id: Some("msg_after_compact".into()),
        tool_calls: vec![],
        model_id: Some("gpt-cold".into()),
        model_fingerprint: None,
        reasoning_effort: None,
    });
    adapter
        .append_chat_message(&info, &ordinary_metadata)
        .await
        .unwrap();
    adapter
        .append_chat_message(&info, &ordinary_message)
        .await
        .unwrap();
    let cold_with_ordinary = adapter.load_session_without_updates(&info).await.unwrap();
    assert!(matches!(
        &cold_with_ordinary.chat_history[replacement.len()],
        ConversationItem::ResponseOutputMetadata(metadata)
            if metadata.items[0].item_id.as_deref() == Some("msg_after_compact")
                && metadata.items[0]
                    .internal_chat_message_metadata_passthrough
                    .as_ref()
                    .and_then(|value| value.turn_id.as_deref())
                    == Some("turn-after-compact")
    ));

    let request = xai_grok_sampling_types::ConversationRequest {
        items: cold_with_ordinary.chat_history,
        model: Some("gpt-cold".into()),
        ..Default::default()
    };
    let created = xai_grok_sampling_types::conversation_request_to_codex_create_response(&request);
    let mut wire = serde_json::to_value(created).unwrap();
    xai_grok_sampling_types::patch_response_message_item_ids(
        &mut wire,
        &xai_grok_sampling_types::response_message_item_ids(&request),
    );
    xai_grok_sampling_types::patch_response_item_metadata_passthrough(
        &mut wire,
        &xai_grok_sampling_types::response_item_metadata_passthrough_for_origin(
            &request,
            Some(&ordinary_origin),
        )
        .unwrap(),
    )
    .unwrap();
    let input = wire["input"].as_array().unwrap();
    assert_eq!(input[0]["id"], "msg_cold");
    assert_eq!(input[1]["id"], "rs_cold");
    assert_eq!(input[1]["encrypted_content"], "cipher-reasoning-cold");
    assert_eq!(input[2]["id"], "cmp_cold");
    assert_eq!(input[2]["encrypted_content"], "cipher-compaction-cold");
    assert_eq!(input[3]["id"], "msg_after_compact");
    assert_eq!(
        input
            .iter()
            .map(
                |item| item["internal_chat_message_metadata_passthrough"]["turn_id"]
                    .as_str()
                    .unwrap()
            )
            .collect::<Vec<_>>(),
        [
            "turn-cold-message",
            "turn-cold-reasoning",
            "turn-cold-compaction",
            "turn-after-compact"
        ]
    );

    let compact_wire = serde_json::to_value(
        xai_grok_sampling_types::conversation_request_to_codex_compact_request_for_origin(
            &request,
            Some(&ordinary_origin),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        compact_wire["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(
                |item| item["internal_chat_message_metadata_passthrough"]["turn_id"]
                    .as_str()
                    .unwrap()
            )
            .collect::<Vec<_>>(),
        [
            "turn-cold-message",
            "turn-cold-reasoning",
            "turn-cold-compaction",
            "turn-after-compact"
        ]
    );
}

#[tokio::test]
async fn cold_load_rejects_missing_or_mutated_native_manifest() {
    let response: xai_grok_sampling_types::CodexCompactResponse = serde_json::from_value(
        serde_json::json!({
            "output": [
                {"type":"message","id":"msg","role":"user","content":[{"type":"input_text","text":"retained"}]},
                {"type":"reasoning","id":"rs","summary":[],"encrypted_content":"reasoning"},
                {"type":"compaction","id":"cmp","encrypted_content":"compaction"}
            ]
        }),
    )
    .unwrap();
    let mut valid = xai_grok_sampling_types::codex_compact_output_to_conversation(
        response.output,
        xai_grok_sampling_types::NativeCompactionCompatibility::codex("test-model", None),
    )
    .unwrap();
    valid.insert(0, ConversationItem::system("system"));
    let valid_value = serde_json::to_value(&valid).unwrap();

    let mut invalid = Vec::new();
    let mutate_manifest =
        |value: &mut serde_json::Value,
         mutation: &dyn Fn(&mut serde_json::Map<String, serde_json::Value>)| {
            let descriptor = value
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|item| item["type"] == "native_compaction_metadata")
                .unwrap()
                .as_object_mut()
                .unwrap();
            mutation(descriptor);
        };

    let mut missing = valid_value.clone();
    mutate_manifest(&mut missing, &|manifest| {
        manifest.remove("item_metadata");
    });
    invalid.push(missing);

    let mut middle_removed = valid_value.clone();
    mutate_manifest(&mut middle_removed, &|manifest| {
        manifest["item_metadata"].as_array_mut().unwrap().remove(1);
    });
    invalid.push(middle_removed);

    let mut extra = valid_value.clone();
    mutate_manifest(&mut extra, &|manifest| {
        let entries = manifest["item_metadata"].as_array_mut().unwrap();
        entries.push(entries[1].clone());
    });
    invalid.push(extra);

    let mut duplicate = valid_value.clone();
    mutate_manifest(&mut duplicate, &|manifest| {
        let entries = manifest["item_metadata"].as_array_mut().unwrap();
        entries[1] = entries[0].clone();
    });
    invalid.push(duplicate);

    for (field, value) in [
        ("input_index", serde_json::json!(9)),
        ("kind", serde_json::json!("message")),
        ("item_id", serde_json::json!("wrong-id")),
    ] {
        let mut changed = valid_value.clone();
        mutate_manifest(&mut changed, &|manifest| {
            manifest["item_metadata"].as_array_mut().unwrap()[1][field] = value.clone();
        });
        invalid.push(changed);
    }

    let mut wrong_segment_length = valid_value.clone();
    mutate_manifest(&mut wrong_segment_length, &|manifest| {
        manifest["replacement_segment_items"] = serde_json::json!(2);
    });
    invalid.push(wrong_segment_length);

    let mut missing_descriptor = valid_value.clone();
    missing_descriptor
        .as_array_mut()
        .unwrap()
        .retain(|item| item["type"] != "native_compaction_metadata");
    invalid.push(missing_descriptor);

    let mut legacy = valid_value;
    mutate_manifest(&mut legacy, &|manifest| {
        manifest["schema_version"] = serde_json::json!(1);
    });
    invalid.push(legacy);

    for (index, items) in invalid.into_iter().enumerate() {
        let dir = tempfile::tempdir().unwrap();
        let info = info();
        let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
        adapter
            .init_session(&info, default_model_id())
            .await
            .unwrap();
        let mut jsonl = Vec::new();
        for item in items.as_array().unwrap() {
            serde_json::to_writer(&mut jsonl, item).unwrap();
            jsonl.push(b'\n');
        }
        std::fs::write(dir.path().join("chat_history.jsonl"), jsonl).unwrap();
        let error = adapter
            .load_session_without_updates(&info)
            .await
            .expect_err("invalid native binding must fail cold load");
        assert_eq!(
            error.kind(),
            std::io::ErrorKind::InvalidData,
            "case {index}"
        );
    }
}

#[tokio::test]
async fn later_rewind_marker_prevents_checkpoint_cache_repair() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let (checkpoint, marker) = compaction_fixture(&info);
    adapter
        .write_compaction_checkpoint(&info, &checkpoint)
        .await
        .unwrap();
    adapter
        .append_update_durable_commit_aware(&info, &marker)
        .await
        .unwrap();
    let rewind = SessionUpdate::Xai(Box::new(SessionNotification {
        session_id: info.id.clone(),
        update: XaiSessionUpdate::RewindMarker {
            target_prompt_index: 1,
            transaction_id: None,
            rewound_history_json: None,
            created_at: "2026-01-01T00:01:00Z".into(),
        },
        meta: None,
    }));
    adapter
        .append_update_durable_commit_aware(&info, &rewind)
        .await
        .unwrap();
    adapter
        .replace_chat_history(&info, &[ConversationItem::system("rewound cache")])
        .await
        .unwrap();

    let loaded = adapter.load_session_without_updates(&info).await.unwrap();
    assert_eq!(loaded.chat_history[0].text_content(), "rewound cache");
}

fn committed_rewind(
    info: &Info,
    history: Vec<ConversationItem>,
    transaction_id: &str,
) -> SessionUpdate {
    SessionUpdate::Xai(Box::new(SessionNotification {
        session_id: info.id.clone(),
        update: XaiSessionUpdate::RewindMarker {
            target_prompt_index: 1,
            transaction_id: Some(transaction_id.into()),
            rewound_history_json: Some(serde_json::to_string(&history).unwrap()),
            created_at: "2026-01-01T00:01:00Z".into(),
        },
        meta: None,
    }))
}

#[tokio::test]
async fn rewind_marker_repairs_cache_not_committed_on_resume() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let old = vec![
        ConversationItem::system("system"),
        ConversationItem::user("kept"),
        ConversationItem::assistant("dead branch"),
    ];
    adapter.replace_chat_history(&info, &old).await.unwrap();
    let rewound = old[..2].to_vec();
    adapter
        .append_update_durable_commit_aware(
            &info,
            &committed_rewind(&info, rewound.clone(), "rewind-cache-failed"),
        )
        .await
        .unwrap();

    let loaded = adapter.load_session_without_updates(&info).await.unwrap();
    assert_eq!(
        serde_json::to_value(&loaded.chat_history).unwrap(),
        serde_json::to_value(&rewound).unwrap()
    );
}

#[tokio::test]
async fn rewind_cache_commit_with_stale_bookkeeping_is_reconciled() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let rewound = vec![
        ConversationItem::system("system"),
        ConversationItem::user("kept"),
    ];
    adapter.replace_chat_history(&info, &rewound).await.unwrap();
    adapter
        .append_update_durable_commit_aware(
            &info,
            &committed_rewind(&info, rewound.clone(), "rewind-cache-current"),
        )
        .await
        .unwrap();
    let mut summary = adapter.read_summary_sync(&info).unwrap();
    summary.num_chat_messages = 999;
    adapter.write_summary_sync(&info, &summary).unwrap();

    let loaded = adapter.load_session_without_updates(&info).await.unwrap();
    assert_eq!(loaded.chat_history.len(), rewound.len());
    assert_eq!(
        adapter.read_summary_sync(&info).unwrap().num_chat_messages,
        rewound.len()
    );
}

#[tokio::test]
async fn rewind_crash_recovery_preserves_native_checkpoint_and_post_rewind_turns() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let (checkpoint, marker) = compaction_fixture(&info);
    adapter
        .write_compaction_checkpoint(&info, &checkpoint)
        .await
        .unwrap();
    adapter
        .append_update_durable_commit_aware(&info, &marker)
        .await
        .unwrap();
    let rewound = checkpoint.compacted_history.clone();
    adapter
        .append_update_durable_commit_aware(
            &info,
            &committed_rewind(&info, rewound.clone(), "rewind-after-native"),
        )
        .await
        .unwrap();
    let post_rewind = SessionUpdate::Acp(Box::new(acp::SessionNotification::new(
        info.id.clone(),
        acp::SessionUpdate::UserMessageChunk(
            acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new(
                "post-rewind".to_string(),
            )))
            .meta(serde_json::json!({"promptIndex": 1}).as_object().cloned()),
        ),
    )));
    adapter.append_update(&info, &post_rewind).await.unwrap();
    adapter
        .replace_chat_history(&info, &[ConversationItem::system("stale pre-rewind cache")])
        .await
        .unwrap();

    let loaded = adapter.load_session_without_updates(&info).await.unwrap();
    assert_eq!(loaded.chat_history.len(), rewound.len() + 1);
    assert_eq!(
        loaded.chat_history.last().unwrap().text_content(),
        "post-rewind"
    );
    assert!(
        xai_grok_sampling_types::native_compaction_compatibility(&loaded.chat_history)
            .unwrap()
            .is_some()
    );
}

#[tokio::test]
async fn crash_before_rewind_marker_commit_keeps_pre_rewind_cache() {
    let dir = tempfile::tempdir().unwrap();
    let info = info();
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    adapter
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let old = vec![ConversationItem::system("old authority")];
    adapter.replace_chat_history(&info, &old).await.unwrap();
    let loaded = adapter.load_session_without_updates(&info).await.unwrap();
    assert_eq!(loaded.chat_history[0].text_content(), "old authority");
}

#[cfg(target_os = "macos")]
#[test]
fn darwin_fullfsync_seam_reports_invalid_descriptor() {
    assert!(super::super::fullfsync_raw(-1).is_err());
}
