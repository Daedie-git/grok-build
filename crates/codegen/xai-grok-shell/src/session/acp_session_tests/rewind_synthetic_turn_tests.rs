//! Regression tests: rewind must remove the rewound turn even when the
//! session contains synthetic-origin turns (auto-wake task/subagent
//! completions, notification drains, scheduler fires).
//!
//! Those turns increment `prompt_index` but push a *synthetic* `User` item;
//! truncation that counts only non-synthetic `User` items therefore leaves
//! the "rewound" turn in the model's context.

use super::support::{create_test_actor, successful_rewind_persistence};

use crate::sampling::ConversationItem;
use crate::session::{RewindMode, RewindRequest};

#[tokio::test(flavor = "current_thread")]
async fn rewind_not_committed_keeps_prepared_snapshot_out_of_live_state() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, _gateway_rx) = tokio::sync::mpsc::unbounded_channel();
            let (persistence_tx, mut persistence_rx) = tokio::sync::mpsc::unbounded_channel();
            let actor = create_test_actor(0, 200_000, 80, gateway_tx, persistence_tx).await;
            let mut snapshot = actor.chat_state_handle.snapshot().await.unwrap();
            snapshot.conversation = vec![
                ConversationItem::system("system"),
                ConversationItem::user("user info"),
                ConversationItem::user("prompt zero"),
                ConversationItem::assistant("answer zero"),
                ConversationItem::user("dead prompt"),
                ConversationItem::assistant("dead answer"),
            ];
            snapshot.prompt_index = 2;
            snapshot.prompt_texts = vec!["prompt zero".into(), "dead prompt".into()];
            actor.chat_state_handle.restore_snapshot(snapshot.clone());

            let rewind = actor.handle_rewind(RewindRequest {
                target_prompt_index: 1,
                force: true,
                mode: RewindMode::ConversationOnly,
            });
            tokio::pin!(rewind);
            let message = tokio::select! {
                message = persistence_rx.recv() => message.unwrap(),
                result = &mut rewind => panic!("rewind completed before marker outcome: {result:?}"),
            };
            let crate::session::persistence::PersistenceMsg::InstallRewindAndAck {
                replacement,
                respond_to,
                ..
            } = message
            else {
                panic!("expected rewind transaction")
            };
            assert_eq!(replacement.len(), 4);
            // `snapshot()` queues behind the rewind gate. Use the live
            // queries that stay allowed so this check cannot deadlock the
            // pending NotCommitted ack.
            assert_eq!(actor.chat_state_handle.get_prompt_index().await, 2);
            assert_eq!(actor.chat_state_handle.get_conversation_len().await, 6);
            respond_to
                .send(crate::session::persistence::TimelineTransactionOutcome::NotCommitted(
                    std::io::Error::other("injected marker failure"),
                ))
                .unwrap();
            let error = rewind.await.expect_err("NotCommitted must fail the API call");
            assert!(error.to_string().contains("not committed"));
            let after = actor.chat_state_handle.snapshot().await.unwrap();
            assert_eq!(after.prompt_index, 2);
            assert_eq!(after.conversation.len(), 6);
        })
        .await;
}

/// Conversation+files rewind must not touch the working tree when the durable
/// marker fails to commit — same fail-closed authority as chat history.
#[tokio::test(flavor = "current_thread")]
async fn rewind_all_not_committed_leaves_chat_and_files_untouched() {
    use std::path::Path;
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, _gateway_rx) = tokio::sync::mpsc::unbounded_channel();
            let (persistence_tx, mut persistence_rx) = tokio::sync::mpsc::unbounded_channel();
            let actor = create_test_actor(0, 200_000, 80, gateway_tx, persistence_tx).await;

            let cwd = Path::new("/tmp");
            let edited = Path::new("/tmp/edited.rs");
            // Live disk is the post-prompt-1 edit; rewind would restore BEFORE.
            actor
                .tool_context
                .fs
                .write_file(edited, b"AFTER_EDIT")
                .await
                .expect("seed live file");
            actor
                .file_state_tracker
                .add_before_snapshot_for_prompt(
                    1,
                    edited,
                    cwd,
                    Some("BEFORE_EDIT".into()),
                )
                .await;

            let mut snapshot = actor.chat_state_handle.snapshot().await.unwrap();
            snapshot.conversation = vec![
                ConversationItem::system("system"),
                ConversationItem::user("user info"),
                ConversationItem::user("prompt zero"),
                ConversationItem::assistant("answer zero"),
                ConversationItem::user("prompt one edited the file"),
                ConversationItem::assistant("done"),
            ];
            snapshot.prompt_index = 2;
            snapshot.prompt_texts = vec!["prompt zero".into(), "prompt one edited the file".into()];
            actor.chat_state_handle.restore_snapshot(snapshot);

            let rewind = actor.handle_rewind(RewindRequest {
                target_prompt_index: 1,
                force: true,
                mode: RewindMode::All,
            });
            tokio::pin!(rewind);
            let message = tokio::select! {
                message = persistence_rx.recv() => message.unwrap(),
                result = &mut rewind => panic!("rewind completed before marker outcome: {result:?}"),
            };
            let crate::session::persistence::PersistenceMsg::InstallRewindAndAck {
                respond_to,
                ..
            } = message
            else {
                panic!("expected rewind transaction before any file mutation")
            };

            // Marker not yet acked: live chat and disk must still be the pre-rewind world.
            // `snapshot()` queues behind the rewind gate; these queries do not.
            assert_eq!(actor.chat_state_handle.get_prompt_index().await, 2);
            assert_eq!(actor.chat_state_handle.get_conversation_len().await, 6);
            let mid_file = actor
                .tool_context
                .fs
                .try_read_to_string(edited)
                .await
                .expect("read")
                .expect("file present");
            assert_eq!(mid_file, "AFTER_EDIT");

            respond_to
                .send(crate::session::persistence::TimelineTransactionOutcome::NotCommitted(
                    std::io::Error::other("injected marker failure"),
                ))
                .unwrap();
            let error = rewind
                .await
                .expect_err("NotCommitted All-mode rewind must fail the API call");
            assert!(
                error.to_string().contains("not committed"),
                "unexpected error: {error}"
            );

            let after = actor.chat_state_handle.snapshot().await.unwrap();
            assert_eq!(after.prompt_index, 2, "chat must stay unrewound");
            assert_eq!(after.conversation.len(), 6);
            let after_file = actor
                .tool_context
                .fs
                .try_read_to_string(edited)
                .await
                .expect("read")
                .expect("file present");
            assert_eq!(
                after_file, "AFTER_EDIT",
                "working tree must not revert when the rewind marker was NotCommitted"
            );
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn committed_rewind_installs_full_snapshot_without_losing_actor_state() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, _gateway_rx) = tokio::sync::mpsc::unbounded_channel();
            let persistence_tx = successful_rewind_persistence();
            let actor = create_test_actor(0, 200_000, 80, gateway_tx, persistence_tx).await;
            let mut snapshot = actor.chat_state_handle.snapshot().await.unwrap();
            snapshot.conversation = vec![
                ConversationItem::system("system"),
                ConversationItem::user("user info"),
                ConversationItem::user("kept"),
                ConversationItem::assistant("kept answer"),
                ConversationItem::user("removed"),
                ConversationItem::assistant("removed answer"),
            ];
            snapshot.prompt_index = 2;
            snapshot.prompt_texts = vec!["kept".into(), "removed".into()];
            snapshot.total_tokens = 4_321;
            snapshot.estimate_at_last_response = 4_000;
            snapshot.agent_edited_paths.insert("src/lib.rs".into());
            snapshot.credentials.api_key = Some("secret".into());
            let live_model = snapshot.sampling_config.model.clone();
            actor.chat_state_handle.restore_snapshot(snapshot);
            actor.chat_state_handle.begin_turn_capture();

            let response = actor
                .handle_rewind(RewindRequest {
                    target_prompt_index: 1,
                    force: true,
                    mode: RewindMode::ConversationOnly,
                })
                .await
                .unwrap();
            assert!(response.success);
            let installed = actor.chat_state_handle.snapshot().await.unwrap();
            assert_eq!(installed.prompt_index, 1);
            assert_eq!(installed.conversation.len(), 4);
            assert_eq!(
                installed.total_tokens,
                xai_chat_state::estimate_conversation_tokens(&installed.conversation),
                "rewind must re-estimate tokens from the truncated conversation"
            );
            assert_eq!(installed.estimate_at_last_response, installed.total_tokens);
            assert!(installed.agent_edited_paths.contains("src/lib.rs"));
            assert_eq!(installed.credentials.api_key.as_deref(), Some("secret"));
            assert_eq!(installed.sampling_config.model, live_model);

            actor
                .chat_state_handle
                .push_assistant_response(ConversationItem::assistant("after rewind"));
            let capture = actor
                .chat_state_handle
                .take_turn_messages()
                .await
                .expect("turn capture must survive rewind install");
            assert_eq!(
                capture.messages.last().map(ConversationItem::text_content),
                Some("after rewind".into())
            );
        })
        .await;
}

/// Build the canonical bugged-session shape:
///
/// ```text
/// [Sys, User(user_info), U0(real), A0, U1(auto-wake, synthetic), A1, U2(real), A2]
/// prompt_index = 3, prompt_texts = [P0, TASK_WAKE, P2]
/// ```
///
/// Turn 1 is a background-task auto-wake (`PromptOrigin::TaskCompleted`):
/// it consumed a prompt index but its user item is synthetic.
fn seed_conversation(mark_turn_starts: bool) -> Vec<ConversationItem> {
    let turn_user = |text: &str, idx: usize| {
        let mut item = ConversationItem::user(text);
        if mark_turn_starts {
            item.set_prompt_index(idx);
        }
        item
    };
    let auto_wake = |text: &str, idx: usize| {
        let mut item = ConversationItem::task_completed(text);
        if mark_turn_starts {
            item.set_prompt_index(idx);
        }
        item
    };
    vec![
        ConversationItem::system("SYS"),
        ConversationItem::user("<user_info>OS: test</user_info>"),
        turn_user("P0", 0),
        ConversationItem::assistant("A0"),
        auto_wake("Background task abc completed", 1),
        ConversationItem::assistant("A1"),
        turn_user("P2", 2),
        ConversationItem::assistant("A2"),
    ]
}

async fn run_rewind_over_synthetic_turn(mark_turn_starts: bool) {
    let (gateway_tx, _gateway_rx) = tokio::sync::mpsc::unbounded_channel();
    let persistence_tx = successful_rewind_persistence();
    let actor = create_test_actor(0, 200_000, 80, gateway_tx, persistence_tx).await;

    let mut snap = actor
        .chat_state_handle
        .snapshot()
        .await
        .expect("snapshot available");
    snap.conversation = seed_conversation(mark_turn_starts);
    snap.prompt_index = 3;
    snap.prompt_texts = vec![
        "P0".into(),
        "Background task abc completed".into(),
        "P2".into(),
    ];
    snap.last_compaction_prompt_index = None;
    actor.chat_state_handle.restore_snapshot(snap);

    // Rewind to prompt #2 — "restore state before P2 ran".
    let resp = actor
        .handle_rewind(RewindRequest {
            target_prompt_index: 2,
            force: true,
            mode: RewindMode::ConversationOnly,
        })
        .await
        .expect("handle_rewind ok");
    assert!(resp.success, "rewind should succeed: {resp:?}");
    assert_eq!(resp.prompt_text.as_deref(), Some("P2"));

    let conv = actor.chat_state_handle.get_conversation().await;
    let texts: Vec<String> = conv.iter().map(|c| c.text_content()).collect();

    assert!(
        !texts.iter().any(|t| t == "P2" || t == "A2"),
        "rewound turn must not stay in the model's context \
         (mark_turn_starts={mark_turn_starts}): {texts:?}"
    );
    assert_eq!(
        texts,
        vec![
            "SYS",
            "<user_info>OS: test</user_info>",
            "P0",
            "A0",
            "Background task abc completed",
            "A1",
        ],
        "conversation must keep prompts 0..=1 only"
    );
    assert_eq!(actor.chat_state_handle.get_prompt_index().await, 2);
}

/// Marker-less items (sessions persisted before `UserItem.prompt_index`
/// existed): the counting fallback must classify the synthetic auto-wake
/// item as a turn start.
#[tokio::test(flavor = "current_thread")]
async fn rewind_removes_turn_after_synthetic_auto_wake_unmarked() {
    let local = tokio::task::LocalSet::new();
    local.run_until(run_rewind_over_synthetic_turn(false)).await;
}

/// Marked items (what `turn.rs` stamps on every turn start): the explicit
/// per-item prompt index takes priority.
#[tokio::test(flavor = "current_thread")]
async fn rewind_removes_turn_after_synthetic_auto_wake_marked() {
    let local = tokio::task::LocalSet::new();
    local.run_until(run_rewind_over_synthetic_turn(true)).await;
}

/// Rewind on a session with no prompts: the picker has nothing to offer and
/// an execute request is rejected (no silent no-op "success").
#[tokio::test(flavor = "current_thread")]
async fn rewind_with_no_prompts_lists_no_points_and_rejects_execute() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, _gateway_rx) = tokio::sync::mpsc::unbounded_channel();
            let persistence_tx = successful_rewind_persistence();
            let actor = create_test_actor(0, 200_000, 80, gateway_tx, persistence_tx).await;

            let points = actor.get_rewind_points().await;
            assert!(
                points.rewind_points.is_empty(),
                "fresh session must expose zero rewind points: {points:?}"
            );

            let resp = actor
                .handle_rewind(RewindRequest {
                    target_prompt_index: 0,
                    force: true,
                    mode: RewindMode::ConversationOnly,
                })
                .await
                .expect("handle_rewind ok");
            assert!(!resp.success, "rewind with no prompts must be rejected");
            assert!(
                resp.error
                    .as_deref()
                    .unwrap_or("")
                    .contains("Cannot rewind"),
                "rejection must carry a clear error: {resp:?}"
            );
        })
        .await;
}

/// Rewind to the start of the conversation (target = 0) keeps only the
/// session preamble — System + user_info + pre-turn synthetic reminders —
/// even when turn 0 exists alongside synthetic auto-wake turns.
#[tokio::test(flavor = "current_thread")]
async fn rewind_to_start_keeps_only_preamble() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, _gateway_rx) = tokio::sync::mpsc::unbounded_channel();
            let persistence_tx = successful_rewind_persistence();
            let actor = create_test_actor(0, 200_000, 80, gateway_tx, persistence_tx).await;

            let mut conversation = seed_conversation(true);
            // Pre-turn reminder in the preamble prefix must survive target=0.
            conversation.insert(2, ConversationItem::system_reminder("skills"));
            let mut snap = actor
                .chat_state_handle
                .snapshot()
                .await
                .expect("snapshot available");
            snap.conversation = conversation;
            snap.prompt_index = 3;
            snap.prompt_texts = vec![
                "P0".into(),
                "Background task abc completed".into(),
                "P2".into(),
            ];
            actor.chat_state_handle.restore_snapshot(snap);

            let resp = actor
                .handle_rewind(RewindRequest {
                    target_prompt_index: 0,
                    force: true,
                    mode: RewindMode::ConversationOnly,
                })
                .await
                .expect("handle_rewind ok");
            assert!(resp.success, "rewind to start should succeed: {resp:?}");
            assert_eq!(resp.prompt_text.as_deref(), Some("P0"));

            let conv = actor.chat_state_handle.get_conversation().await;
            let texts: Vec<String> = conv.iter().map(|c| c.text_content()).collect();
            assert_eq!(
                texts,
                vec!["SYS", "<user_info>OS: test</user_info>", "skills"],
                "target 0 must keep only the preamble prefix"
            );
            assert_eq!(actor.chat_state_handle.get_prompt_index().await, 0);

            // With prompt_index back at 0 the session behaves like a fresh
            // one: no points, further rewinds rejected.
            assert!(actor.get_rewind_points().await.rewind_points.is_empty());
            let again = actor
                .handle_rewind(RewindRequest {
                    target_prompt_index: 0,
                    force: true,
                    mode: RewindMode::ConversationOnly,
                })
                .await
                .expect("handle_rewind ok");
            assert!(!again.success, "no prompts left to rewind: {again:?}");
        })
        .await;
}

/// Two sequential rewinds narrow the history correctly each time — the
/// second rewind operates on the already-truncated conversation (markers
/// still present on the surviving items).
#[tokio::test(flavor = "current_thread")]
async fn rewind_twice_narrows_history_each_time() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, _gateway_rx) = tokio::sync::mpsc::unbounded_channel();
            let persistence_tx = successful_rewind_persistence();
            let actor = create_test_actor(0, 200_000, 80, gateway_tx, persistence_tx).await;

            // 5 turns: real, wake, real, wake, real.
            let marked = |text: &str, idx: usize| {
                let mut item = ConversationItem::user(text);
                item.set_prompt_index(idx);
                item
            };
            let marked_wake = |text: &str, idx: usize| {
                let mut item = ConversationItem::task_completed(text);
                item.set_prompt_index(idx);
                item
            };
            let mut snap = actor
                .chat_state_handle
                .snapshot()
                .await
                .expect("snapshot available");
            snap.conversation = vec![
                ConversationItem::system("SYS"),
                ConversationItem::user("<user_info>OS: test</user_info>"),
                marked("P0", 0),
                ConversationItem::assistant("A0"),
                marked_wake("W1", 1),
                ConversationItem::assistant("A1"),
                marked("P2", 2),
                ConversationItem::assistant("A2"),
                marked_wake("W3", 3),
                ConversationItem::assistant("A3"),
                marked("P4", 4),
                ConversationItem::assistant("A4"),
            ];
            snap.prompt_index = 5;
            snap.prompt_texts = vec![
                "P0".into(),
                "W1".into(),
                "P2".into(),
                "W3".into(),
                "P4".into(),
            ];
            actor.chat_state_handle.restore_snapshot(snap);

            // First rewind: to turn 3 (drops W3, A3, P4, A4).
            let first = actor
                .handle_rewind(RewindRequest {
                    target_prompt_index: 3,
                    force: true,
                    mode: RewindMode::ConversationOnly,
                })
                .await
                .expect("handle_rewind ok");
            assert!(first.success, "{first:?}");
            assert_eq!(first.prompt_text.as_deref(), Some("W3"));
            let conv = actor.chat_state_handle.get_conversation().await;
            let texts: Vec<String> = conv.iter().map(|c| c.text_content()).collect();
            assert_eq!(
                texts,
                vec![
                    "SYS",
                    "<user_info>OS: test</user_info>",
                    "P0",
                    "A0",
                    "W1",
                    "A1",
                    "P2",
                    "A2",
                ],
                "first rewind keeps turns 0..=2"
            );
            assert_eq!(actor.chat_state_handle.get_prompt_index().await, 3);

            // Second rewind: to turn 1 (drops W1, A1, P2, A2).
            let second = actor
                .handle_rewind(RewindRequest {
                    target_prompt_index: 1,
                    force: true,
                    mode: RewindMode::ConversationOnly,
                })
                .await
                .expect("handle_rewind ok");
            assert!(second.success, "{second:?}");
            assert_eq!(second.prompt_text.as_deref(), Some("W1"));
            let conv = actor.chat_state_handle.get_conversation().await;
            let texts: Vec<String> = conv.iter().map(|c| c.text_content()).collect();
            assert_eq!(
                texts,
                vec!["SYS", "<user_info>OS: test</user_info>", "P0", "A0"],
                "second rewind keeps only turn 0"
            );
            assert_eq!(actor.chat_state_handle.get_prompt_index().await, 1);

            // Picker after two rewinds offers exactly turn 0.
            let points = actor.get_rewind_points().await;
            let indices: Vec<usize> = points
                .rewind_points
                .iter()
                .map(|p| p.prompt_index)
                .collect();
            assert_eq!(indices, vec![0]);
        })
        .await;
}

/// Midpoint rewind with synthetic turns on BOTH sides of the cut, in both
/// marker and counting-fallback modes.
#[tokio::test(flavor = "current_thread")]
async fn rewind_to_midpoint_with_synthetic_turns_on_both_sides() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            for mark_turn_starts in [false, true] {
                let (gateway_tx, _gateway_rx) = tokio::sync::mpsc::unbounded_channel();
                let persistence_tx = successful_rewind_persistence();
                let actor = create_test_actor(0, 200_000, 80, gateway_tx, persistence_tx).await;

                let user = |text: &str, idx: usize| {
                    let mut item = ConversationItem::user(text);
                    if mark_turn_starts {
                        item.set_prompt_index(idx);
                    }
                    item
                };
                let wake = |text: &str, idx: usize| {
                    let mut item = ConversationItem::task_completed(text);
                    if mark_turn_starts {
                        item.set_prompt_index(idx);
                    }
                    item
                };
                let mut snap = actor
                    .chat_state_handle
                    .snapshot()
                    .await
                    .expect("snapshot available");
                snap.conversation = vec![
                    ConversationItem::system("SYS"),
                    ConversationItem::user("<user_info>OS: test</user_info>"),
                    user("P0", 0),
                    ConversationItem::assistant("A0"),
                    wake("W1", 1),
                    ConversationItem::assistant("A1"),
                    user("P2", 2),
                    ConversationItem::assistant("A2"),
                    wake("W3", 3),
                    ConversationItem::assistant("A3"),
                    user("P4", 4),
                    ConversationItem::assistant("A4"),
                ];
                snap.prompt_index = 5;
                snap.prompt_texts = vec![
                    "P0".into(),
                    "W1".into(),
                    "P2".into(),
                    "W3".into(),
                    "P4".into(),
                ];
                actor.chat_state_handle.restore_snapshot(snap);

                let resp = actor
                    .handle_rewind(RewindRequest {
                        target_prompt_index: 2,
                        force: true,
                        mode: RewindMode::ConversationOnly,
                    })
                    .await
                    .expect("handle_rewind ok");
                assert!(resp.success, "mark={mark_turn_starts}: {resp:?}");
                assert_eq!(resp.prompt_text.as_deref(), Some("P2"));

                let conv = actor.chat_state_handle.get_conversation().await;
                let texts: Vec<String> = conv.iter().map(|c| c.text_content()).collect();
                assert_eq!(
                    texts,
                    vec![
                        "SYS",
                        "<user_info>OS: test</user_info>",
                        "P0",
                        "A0",
                        "W1",
                        "A1",
                    ],
                    "midpoint rewind keeps turns 0..=1 (mark={mark_turn_starts})"
                );
                assert_eq!(actor.chat_state_handle.get_prompt_index().await, 2);
            }
        })
        .await;
}

/// Rewind to the auto-wake turn itself (target = 1) must cut the auto-wake
/// item and everything after it.
#[tokio::test(flavor = "current_thread")]
async fn rewind_to_synthetic_auto_wake_turn_cuts_at_the_wake() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, _gateway_rx) = tokio::sync::mpsc::unbounded_channel();
            let persistence_tx = successful_rewind_persistence();
            let actor = create_test_actor(0, 200_000, 80, gateway_tx, persistence_tx).await;

            let mut snap = actor
                .chat_state_handle
                .snapshot()
                .await
                .expect("snapshot available");
            snap.conversation = seed_conversation(true);
            snap.prompt_index = 3;
            snap.prompt_texts = vec![
                "P0".into(),
                "Background task abc completed".into(),
                "P2".into(),
            ];
            actor.chat_state_handle.restore_snapshot(snap);

            let resp = actor
                .handle_rewind(RewindRequest {
                    target_prompt_index: 1,
                    force: true,
                    mode: RewindMode::ConversationOnly,
                })
                .await
                .expect("handle_rewind ok");
            assert!(resp.success, "rewind should succeed: {resp:?}");

            let conv = actor.chat_state_handle.get_conversation().await;
            let texts: Vec<String> = conv.iter().map(|c| c.text_content()).collect();
            assert_eq!(
                texts,
                vec!["SYS", "<user_info>OS: test</user_info>", "P0", "A0"],
                "auto-wake turn and everything after it must be removed"
            );
            assert_eq!(actor.chat_state_handle.get_prompt_index().await, 1);
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn review_regression_rewind_to_post_compaction_prompt_preserves_rich_live_items() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, _gateway_rx) = tokio::sync::mpsc::unbounded_channel();
            let persistence_tx = successful_rewind_persistence();
            let actor = create_test_actor(0, 200_000, 80, gateway_tx, persistence_tx).await;

            fn user_at(text: &str, prompt_index: usize) -> ConversationItem {
                let mut item = ConversationItem::user(text);
                item.set_prompt_index(prompt_index);
                item
            }

            let signed = ConversationItem::Reasoning(xai_grok_sampling_types::rs::ReasoningItem {
                id: "rs-post-compact".into(),
                summary: Vec::new(),
                content: None,
                encrypted_content: Some("signed-post-checkpoint".into()),
                status: None,
            });
            let mut snap = actor
                .chat_state_handle
                .snapshot()
                .await
                .expect("snapshot available");
            snap.conversation = vec![
                ConversationItem::system("SYS"),
                user_at("P0", 0),
                ConversationItem::assistant("A0"),
                user_at("P1", 1),
                ConversationItem::assistant("A1"),
                user_at("P2", 2),
                ConversationItem::assistant("A2"),
                user_at("P3", 3),
                ConversationItem::assistant("A3"),
                user_at("P4", 4),
                ConversationItem::assistant("A4"),
                user_at("P5", 5),
                signed,
                ConversationItem::assistant("A5"),
                user_at("P6", 6),
                ConversationItem::assistant("A6"),
            ];
            snap.prompt_index = 7;
            snap.prompt_texts = (0..7).map(|i| format!("P{i}")).collect();
            snap.last_compaction_prompt_index = Some(5);
            actor.chat_state_handle.restore_snapshot(snap);

            let resp = actor
                .handle_rewind(RewindRequest {
                    target_prompt_index: 6,
                    force: true,
                    mode: RewindMode::ConversationOnly,
                })
                .await
                .expect("handle_rewind ok");
            assert!(resp.success, "post-compaction rewind should succeed: {resp:?}");

            let conv = actor.chat_state_handle.get_conversation().await;
            assert!(
                conv.iter().any(|item| matches!(
                    item,
                    ConversationItem::Reasoning(r)
                        if r.encrypted_content.as_deref() == Some("signed-post-checkpoint")
                )),
                "post-compaction rewind must truncate the rich live snapshot, not rebuild text-only ACP chunks: {conv:?}"
            );
            assert!(
                !conv.iter().any(|item| item.text_content() == "P6"),
                "target 6 must drop prompt 6: {conv:?}"
            );
        })
        .await;
}
