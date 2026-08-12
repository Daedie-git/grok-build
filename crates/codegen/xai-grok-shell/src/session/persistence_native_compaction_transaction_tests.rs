use super::*;
use crate::extensions::notification::{
    CompactionCheckpointFile, CompactionCheckpointInfo, SessionNotification,
    SessionUpdate as XaiSessionUpdate,
};

fn fixture(
    info: &Info,
) -> (
    CompactionCheckpointFile,
    SessionUpdate,
    Vec<ConversationItem>,
) {
    let checkpoint = CompactionCheckpointFile {
        checkpoint_id: "checkpoint-test".to_string(),
        prompt_index_at_compaction: 3,
        compacted_history: vec![ConversationItem::system("replacement")],
        schema_version: 1,
        created_at: "2026-01-01T00:00:00Z".to_string(),
        original_user_info: None,
        reread_file_paths: vec![],
    };
    let marker = SessionUpdate::Xai(Box::new(SessionNotification {
        session_id: info.id.clone(),
        update: XaiSessionUpdate::CompactionCheckpoint(Box::new(CompactionCheckpointInfo {
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            prompt_index_at_compaction: checkpoint.prompt_index_at_compaction,
            checkpoint_file: "compaction_checkpoints/checkpoint-test.json".to_string(),
            auto_continue: None,
            schema_version: 1,
            created_at: checkpoint.created_at.clone(),
        })),
        meta: None,
    }));
    let replacement = checkpoint.compacted_history.clone();
    (checkpoint, marker, replacement)
}

#[tokio::test]
async fn native_transaction_ack_follows_all_durable_writes() {
    let dir = tempfile::tempdir().unwrap();
    let info = Info {
        id: acp::SessionId::new("native-success"),
        cwd: "/tmp".to_string(),
    };
    let storage = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    storage
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let (checkpoint, marker, replacement) = fixture(&info);

    assert!(matches!(
        persist_native_compaction_transaction(&storage, &info, &checkpoint, &marker, &replacement,)
            .await,
        TimelineTransactionOutcome::Committed {
            marker_bookkeeping_error: None,
            cache_status: TimelineCacheStatus::Current,
        }
    ));

    assert!(
        dir.path()
            .join("compaction_checkpoints/checkpoint-test.json")
            .is_file()
    );
    assert!(
        !std::fs::read(dir.path().join("updates.jsonl"))
            .unwrap()
            .is_empty()
    );
    let installed = storage.load_chat_history_from_dir(dir.path()).unwrap();
    assert_eq!(installed.len(), replacement.len());
    assert_eq!(installed[0].text_content(), "replacement");
}

#[tokio::test]
async fn native_transaction_marker_and_cache_bookkeeping_failures_stay_committed() {
    let dir = tempfile::tempdir().unwrap();
    let info = Info {
        id: acp::SessionId::new("native-bookkeeping-failure"),
        cwd: "/tmp".to_string(),
    };
    let storage = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    storage
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let summary_path = dir.path().join("summary.json");
    std::fs::remove_file(&summary_path).unwrap();
    std::fs::create_dir(&summary_path).unwrap();
    let (checkpoint, marker, replacement) = fixture(&info);

    assert!(matches!(
        persist_native_compaction_transaction(&storage, &info, &checkpoint, &marker, &replacement,)
            .await,
        TimelineTransactionOutcome::Committed {
            marker_bookkeeping_error: Some(_),
            cache_status: TimelineCacheStatus::CurrentWithBookkeepingError(_),
        }
    ));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("updates.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    let installed = storage.load_chat_history_from_dir(dir.path()).unwrap();
    assert_eq!(installed[0].text_content(), "replacement");
}

#[tokio::test]
async fn native_transaction_returns_write_error_without_replacing_history() {
    let dir = tempfile::tempdir().unwrap();
    let info = Info {
        id: acp::SessionId::new("native-failure"),
        cwd: "/tmp".to_string(),
    };
    let storage = JsonlStorageAdapter::with_explicit_session_dir(dir.path().to_path_buf());
    storage
        .init_session(&info, default_model_id())
        .await
        .unwrap();
    let original = vec![ConversationItem::system("original")];
    storage
        .replace_chat_history(&info, &original)
        .await
        .unwrap();
    std::fs::write(
        dir.path().join("compaction_checkpoints"),
        b"not a directory",
    )
    .unwrap();
    let (checkpoint, marker, replacement) = fixture(&info);

    assert!(matches!(
        persist_native_compaction_transaction(&storage, &info, &checkpoint, &marker, &replacement,)
            .await,
        TimelineTransactionOutcome::NotCommitted(_)
    ));

    let still_live = storage.load_chat_history_from_dir(dir.path()).unwrap();
    assert_eq!(still_live.len(), original.len());
    assert_eq!(still_live[0].text_content(), "original");
}

#[derive(Clone, Copy)]
enum Fault {
    None,
    Checkpoint,
    MarkerNotCommitted,
    MarkerBookkeeping,
    CacheNotCommitted,
    CacheBookkeeping,
}

#[derive(Default)]
struct FakeState {
    checkpoint: bool,
    marker: bool,
    cache: bool,
    writes: Vec<&'static str>,
}

struct FakeStorage {
    fault: Fault,
    state: std::sync::Mutex<FakeState>,
}

impl FakeStorage {
    fn new(fault: Fault) -> Self {
        Self {
            fault,
            state: std::sync::Mutex::new(FakeState::default()),
        }
    }
}

#[async_trait::async_trait]
impl NativeCompactionStorage for FakeStorage {
    async fn stage_checkpoint(
        &self,
        _info: &Info,
        _checkpoint: &CompactionCheckpointFile,
    ) -> io::Result<()> {
        if matches!(self.fault, Fault::Checkpoint) {
            return Err(io::Error::other("checkpoint write failed"));
        }
        let mut state = self.state.lock().unwrap();
        state.checkpoint = true;
        state.writes.push("checkpoint");
        Ok(())
    }

    async fn commit_marker(
        &self,
        _info: &Info,
        _marker: &SessionUpdate,
    ) -> Result<(), crate::session::storage::AppendUpdateError> {
        if matches!(self.fault, Fault::MarkerNotCommitted) {
            return Err(crate::session::storage::AppendUpdateError::NotCommitted(
                io::Error::other("marker append failed"),
            ));
        }
        {
            let mut state = self.state.lock().unwrap();
            state.marker = true;
            state.writes.push("marker");
        }
        if matches!(self.fault, Fault::MarkerBookkeeping) {
            return Err(crate::session::storage::AppendUpdateError::Committed(
                io::Error::other("marker bookkeeping failed"),
            ));
        }
        Ok(())
    }

    async fn replace_cache(
        &self,
        _info: &Info,
        _replacement: &[ConversationItem],
    ) -> Result<(), crate::session::storage::ReplaceChatHistoryError> {
        if matches!(self.fault, Fault::CacheNotCommitted) {
            return Err(
                crate::session::storage::ReplaceChatHistoryError::NotCommitted(io::Error::other(
                    "cache replacement failed",
                )),
            );
        }
        {
            let mut state = self.state.lock().unwrap();
            state.cache = true;
            state.writes.push("cache");
        }
        if matches!(self.fault, Fault::CacheBookkeeping) {
            return Err(crate::session::storage::ReplaceChatHistoryError::Committed(
                io::Error::other("cache bookkeeping failed"),
            ));
        }
        Ok(())
    }
}

async fn run_fault(fault: Fault) -> (TimelineTransactionOutcome, FakeStorage) {
    let info = Info {
        id: acp::SessionId::new("native-fault"),
        cwd: "/tmp".to_string(),
    };
    let (checkpoint, marker, replacement) = fixture(&info);
    let storage = FakeStorage::new(fault);
    let outcome =
        persist_native_compaction_transaction(&storage, &info, &checkpoint, &marker, &replacement)
            .await;
    (outcome, storage)
}

#[tokio::test]
async fn marker_not_committed_keeps_old_state_authoritative() {
    let (outcome, storage) = run_fault(Fault::MarkerNotCommitted).await;
    assert!(matches!(
        outcome,
        TimelineTransactionOutcome::NotCommitted(_)
    ));
    let state = storage.state.lock().unwrap();
    assert!(state.checkpoint, "orphan staged checkpoint is harmless");
    assert!(!state.marker);
    assert!(!state.cache);
}

#[tokio::test]
async fn marker_committed_bookkeeping_failure_still_installs_cache() {
    let (outcome, storage) = run_fault(Fault::MarkerBookkeeping).await;
    assert!(matches!(
        outcome,
        TimelineTransactionOutcome::Committed {
            marker_bookkeeping_error: Some(_),
            cache_status: TimelineCacheStatus::Current,
        }
    ));
    let state = storage.state.lock().unwrap();
    assert!(state.checkpoint && state.marker && state.cache);
}

#[tokio::test]
async fn cache_replacement_failure_after_marker_is_committed() {
    let (outcome, storage) = run_fault(Fault::CacheNotCommitted).await;
    assert!(matches!(
        outcome,
        TimelineTransactionOutcome::Committed {
            cache_status: TimelineCacheStatus::RepairRequired(_),
            ..
        }
    ));
    let state = storage.state.lock().unwrap();
    assert!(state.checkpoint && state.marker);
    assert!(!state.cache);
}

#[tokio::test]
async fn cache_summary_failure_after_file_commit_is_committed() {
    let (outcome, storage) = run_fault(Fault::CacheBookkeeping).await;
    assert!(matches!(
        outcome,
        TimelineTransactionOutcome::Committed {
            cache_status: TimelineCacheStatus::CurrentWithBookkeepingError(_),
            ..
        }
    ));
    assert!(storage.state.lock().unwrap().cache);
}

async fn run_rewind_fault(fault: Fault) -> (TimelineTransactionOutcome, FakeStorage) {
    let info = Info {
        id: acp::SessionId::new("rewind-fault"),
        cwd: "/tmp".to_string(),
    };
    let replacement = vec![ConversationItem::system("rewound")];
    let marker = SessionUpdate::Xai(Box::new(SessionNotification {
        session_id: info.id.clone(),
        update: XaiSessionUpdate::RewindMarker {
            target_prompt_index: 1,
            transaction_id: Some("rewind-transaction".into()),
            rewound_history_json: Some(serde_json::to_string(&replacement).unwrap()),
            created_at: "2026-01-01T00:00:00Z".into(),
        },
        meta: None,
    }));
    let storage = FakeStorage::new(fault);
    let outcome = persist_marker_first_transaction(&storage, &info, &marker, &replacement).await;
    (outcome, storage)
}

#[tokio::test]
async fn rewind_cache_write_is_strictly_after_marker_commit() {
    let (outcome, storage) = run_rewind_fault(Fault::None).await;
    assert!(matches!(
        outcome,
        TimelineTransactionOutcome::Committed { .. }
    ));
    assert_eq!(storage.state.lock().unwrap().writes, ["marker", "cache"]);
}

#[tokio::test]
async fn rewind_marker_not_committed_never_touches_cache() {
    let (outcome, storage) = run_rewind_fault(Fault::MarkerNotCommitted).await;
    assert!(matches!(
        outcome,
        TimelineTransactionOutcome::NotCommitted(_)
    ));
    let state = storage.state.lock().unwrap();
    assert!(!state.marker && !state.cache);
    assert!(state.writes.is_empty());
}

#[tokio::test]
async fn rewind_post_commit_cache_failures_are_committed_outcomes() {
    let (not_replaced, storage) = run_rewind_fault(Fault::CacheNotCommitted).await;
    assert!(matches!(
        not_replaced,
        TimelineTransactionOutcome::Committed {
            cache_status: TimelineCacheStatus::RepairRequired(_),
            ..
        }
    ));
    assert_eq!(storage.state.lock().unwrap().writes, ["marker"]);

    let (bookkeeping, storage) = run_rewind_fault(Fault::CacheBookkeeping).await;
    assert!(matches!(
        bookkeeping,
        TimelineTransactionOutcome::Committed {
            cache_status: TimelineCacheStatus::CurrentWithBookkeepingError(_),
            ..
        }
    ));
    assert_eq!(storage.state.lock().unwrap().writes, ["marker", "cache"]);
}

#[tokio::test]
async fn lost_ack_after_full_commit_does_not_undo_storage() {
    let (outcome, storage) = run_fault(Fault::None).await;
    let (respond_to, response) = tokio::sync::oneshot::channel();
    drop(response);
    assert!(respond_to.send(outcome).is_err(), "acknowledgement is lost");
    let state = storage.state.lock().unwrap();
    assert!(state.checkpoint && state.marker && state.cache);
}

#[tokio::test]
async fn injected_checkpoint_failure_changes_nothing() {
    let (outcome, storage) = run_fault(Fault::Checkpoint).await;
    assert!(matches!(
        outcome,
        TimelineTransactionOutcome::NotCommitted(_)
    ));
    let state = storage.state.lock().unwrap();
    assert!(!state.checkpoint && !state.marker && !state.cache);
}
