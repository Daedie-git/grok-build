//! Session-level proof that a concurrent ACP prompt cannot reach the sampler
//! while a rewind transaction is still pending.

use super::support::*;
use super::*;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use xai_grok_test_support::sse::responses_api_script_exact;
use xai_grok_test_support::{MockInferenceServer, ScriptedResponse};

use crate::session::{RewindMode, RewindRequest};

/// `SessionActor` turn futures overflow the default test thread stack.
fn block_on_session(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(f)
        .expect("spawn large-stack test thread")
        .join()
        .expect("test thread");
}

fn current_thread_local<F>(f: F)
where
    F: Future<Output = ()> + 'static,
{
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    tokio::task::LocalSet::new().block_on(&rt, f);
}

fn drain_gateway(mut rx: tokio::sync::mpsc::UnboundedReceiver<xai_acp_lib::AcpClientMessage>) {
    tokio::task::spawn_local(async move {
        while let Some(msg) = rx.recv().await {
            if let xai_acp_lib::AcpClientMessage::SessionNotification(args) = msg {
                let _ = args.response_tx.send(Ok(()));
            }
        }
    });
}

async fn actor_with_held_rewind_ack(
    server: &MockInferenceServer,
    persistence_tx: tokio::sync::mpsc::UnboundedSender<PersistenceMsg>,
    gateway_tx: tokio::sync::mpsc::UnboundedSender<xai_acp_lib::AcpClientMessage>,
) -> Arc<SessionActor> {
    let sampling_cfg = xai_grok_sampler::SamplerConfig {
        api_key: Some("test-key".to_string()),
        base_url: server.url(),
        model: "test".to_string(),
        api_backend: xai_grok_sampler::ApiBackend::Responses,
        context_window: 256_000,
        max_retries: Some(0),
        idle_timeout_secs: Some(30),
        ..Default::default()
    };
    let (sampler_event_tx, sampler_event_rx) =
        tokio::sync::mpsc::unbounded_channel::<xai_grok_sampler::SamplingEvent>();
    let sampler_handle = xai_grok_sampler::SamplerActor::spawn(
        sampling_cfg,
        xai_grok_sampler::RetryPolicy {
            max_retries: 0,
            rate_limit_retry_threshold: 0,
            ..Default::default()
        },
        sampler_event_tx,
    );

    let mut actor = create_test_actor(0, 256_000, 85, gateway_tx, persistence_tx).await;
    actor.sampler_handle = sampler_handle;
    *actor.agent.borrow_mut() = test_grok_build_agent_with_todo().await;

    let mut cfg = actor
        .chat_state_handle
        .get_sampling_config()
        .await
        .expect("test actor has sampling config");
    cfg.base_url = server.url();
    cfg.api_backend = xai_grok_sampling_types::ApiBackend::Responses;
    cfg.model = "test".to_string();
    actor.chat_state_handle.update_sampling_config(cfg);
    let mut creds = actor.chat_state_handle.get_credentials().await;
    creds.api_key = Some("test-key".to_string());
    actor.chat_state_handle.update_credentials(creds);

    actor
        .workspace_ops
        .bind_local_session(
            &actor.session_id_string(),
            actor.tool_context.cwd.as_path().to_path_buf(),
            actor.tool_context.hunk_tracker_handle.clone(),
            actor.agent.borrow().tool_bridge().toolset(),
            None,
        )
        .expect("bind_local_session");

    let actor = Arc::new(actor);
    {
        let drainer = actor.clone();
        let mut sampler_event_rx = sampler_event_rx;
        tokio::task::spawn_local(async move {
            while let Some(event) = sampler_event_rx.recv().await {
                drainer.handle_sampling_event(event).await;
            }
        });
    }
    actor
}

#[test]
fn concurrent_prompt_does_not_sample_until_rewind_installs() {
    block_on_session(|| {
        current_thread_local(async {
            let server = MockInferenceServer::start()
                .await
                .expect("mock inference server");
            server.enqueue_response(
                "/v1/responses",
                ScriptedResponse::sse(responses_api_script_exact("done", "test")),
            );

            let (gateway_tx, gateway_rx) =
                tokio::sync::mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
            drain_gateway(gateway_rx);
            let (persistence_tx, mut persistence_rx) =
                tokio::sync::mpsc::unbounded_channel::<PersistenceMsg>();
            let (rewind_ack_tx, rewind_ack_rx) = tokio::sync::oneshot::channel();
            tokio::task::spawn_local(async move {
                let mut rewind_ack_tx = Some(rewind_ack_tx);
                while let Some(message) = persistence_rx.recv().await {
                    match message {
                        PersistenceMsg::InstallRewindAndAck { respond_to, .. }
                        | PersistenceMsg::InstallCompactionAndAck { respond_to, .. } => {
                            if let Some(tx) = rewind_ack_tx.take() {
                                let _ = tx.send(respond_to);
                            } else {
                                let _ = respond_to.send(
                                    crate::session::persistence::TimelineTransactionOutcome::Committed {
                                        marker_bookkeeping_error: None,
                                        cache_status:
                                            crate::session::persistence::TimelineCacheStatus::Current,
                                    },
                                );
                            }
                        }
                        PersistenceMsg::FlushAndAck { respond_to } => {
                            let _ = respond_to.send(Ok(()));
                        }
                        _ => {}
                    }
                }
            });

            let actor = actor_with_held_rewind_ack(&server, persistence_tx, gateway_tx).await;
            let mut snapshot = actor.chat_state_handle.snapshot().await.unwrap();
            snapshot.conversation = vec![
                ConversationItem::system("system"),
                ConversationItem::user("kept"),
                ConversationItem::assistant("kept answer"),
                ConversationItem::user("removed"),
                ConversationItem::assistant("removed answer"),
            ];
            snapshot.prompt_index = 2;
            snapshot.prompt_texts = vec!["kept".into(), "removed".into()];
            actor.chat_state_handle.restore_snapshot(snapshot);

            let rewind = actor.handle_rewind(RewindRequest {
                target_prompt_index: 1,
                force: true,
                mode: RewindMode::ConversationOnly,
            });
            tokio::pin!(rewind);
            let rewind_respond_to = tokio::select! {
                ack = rewind_ack_rx => ack.expect("rewind persist ack sender"),
                result = &mut rewind => panic!("rewind completed before marker outcome: {result:?}"),
            };

            let prompt_actor = actor.clone();
            let prompt = tokio::task::spawn_local(async move {
                prompt_actor
                    .handle_prompt(
                        "concurrent-during-rewind",
                        vec![acp::ContentBlock::Text(acp::TextContent::new(
                            "late prompt".to_string(),
                        ))],
                        PromptMode::Agent,
                        None,
                        None,
                        None,
                        None,
                        true,
                        false,
                        None,
                        None,
                        None,
                    )
                    .await
            });

            tokio::time::sleep(Duration::from_millis(150)).await;
            assert_eq!(
                server.request_count(),
                0,
                "sampler must not be invoked while rewind persistence is still pending"
            );
            assert!(
                !prompt.is_finished(),
                "ACP prompt must wait for rewind install or abort"
            );

            rewind_respond_to
                .send(
                    crate::session::persistence::TimelineTransactionOutcome::Committed {
                        marker_bookkeeping_error: None,
                        cache_status: crate::session::persistence::TimelineCacheStatus::Current,
                    },
                )
                .expect("rewind ack still live");
            let rewind_response = rewind.await.expect("rewind result");
            assert!(
                rewind_response.success,
                "rewind should succeed: {rewind_response:?}"
            );

            let prompt_result = tokio::time::timeout(Duration::from_secs(15), prompt)
                .await
                .expect("prompt must finish after rewind install")
                .expect("prompt task");
            assert!(
                prompt_result.is_ok(),
                "concurrent prompt should complete after rewind: {prompt_result:?}"
            );
            assert!(
                server.request_count() >= 1,
                "sampler should run only after rewind installed"
            );
        });
    });
}
