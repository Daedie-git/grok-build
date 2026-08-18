//! Mid-turn `SetSessionModel` must not abort the process.
//!
//! The turn task and the session command loop share one `LocalSet`. If the
//! turn holds `RefCell<Agent>` across `.await`, a concurrent
//! `handle_set_session_model` `borrow`/`borrow_mut` used to panic
//! (`already borrowed`) and `abort()`.
use super::support::*;
use super::*;

fn switch_sampling(
    model: &str,
) -> (
    xai_grok_sampler::SamplerConfig,
    xai_grok_sampling_types::SamplingIdentity,
) {
    let cfg = xai_grok_sampler::SamplerConfig {
        api_key: Some("test-key".into()),
        base_url: "https://api.x.ai/v1".into(),
        model: model.to_string(),
        context_window: 256_000,
        ..Default::default()
    };
    let identity = xai_grok_sampler::resolve_runtime_sampling_identity(
        cfg.api_backend.clone(),
        &cfg.base_url,
        &cfg.model,
        &cfg.extra_headers,
        &cfg.env_http_headers,
    )
    .expect("test sampling identity");
    (cfg, identity)
}

async fn switch_to(actor: &SessionActor, model: &str) -> Result<acp::ModelId, acp::Error> {
    let (cfg, identity) = switch_sampling(model);
    actor
        .handle_set_session_model(
            cfg,
            identity,
            None,
            false,
            true,
            false,
            85,
            xai_grok_agent::SystemPromptIdentity::default(),
        )
        .await
}

/// The 1.0.5 abort: turn holds `agent` across a yield while `SetSessionModel`
/// applies a compatible model change (prompt rewrite requested).
#[tokio::test(flavor = "current_thread")]
async fn set_session_model_does_not_abort_when_turn_holds_agent() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (actor, _gateway_rx) = build_actor().await;
            actor.state.lock().await.running_task = Some(running_task_stub("loop-3"));

            let (held_tx, held_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = tokio::sync::oneshot::channel();
            let holder = actor.clone();
            tokio::task::spawn_local(async move {
                let _guard = holder.agent.borrow();
                let _ = held_tx.send(());
                let _ = release_rx.await;
            });
            held_rx.await.expect("turn task must hold agent");

            let updated = switch_to(&actor, "grok-4.6")
                .await
                .expect("mid-turn model switch must not panic or fail");
            assert_eq!(updated.0.as_ref(), "grok-4.6");
            let model = actor
                .chat_state_handle
                .get_sampling_config()
                .await
                .expect("sampling config")
                .model;
            assert_eq!(model, "grok-4.6");

            let _ = release_tx.send(());
        })
        .await;
}

/// Same abort if the command loop cannot see `running_task` but `agent` is
/// still borrowed (the turn's named `Ref` across `.await`).
#[tokio::test(flavor = "current_thread")]
async fn set_session_model_does_not_abort_when_agent_is_borrowed() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (actor, _gateway_rx) = build_actor().await;

            let (held_tx, held_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = tokio::sync::oneshot::channel();
            let holder = actor.clone();
            tokio::task::spawn_local(async move {
                let _guard = holder.agent.borrow();
                let _ = held_tx.send(());
                let _ = release_rx.await;
            });
            held_rx.await.expect("holder must borrow agent");

            let updated = switch_to(&actor, "grok-4.6")
                .await
                .expect("switch must survive an outstanding Agent Ref");
            assert_eq!(updated.0.as_ref(), "grok-4.6");

            let _ = release_tx.send(());
        })
        .await;
}
