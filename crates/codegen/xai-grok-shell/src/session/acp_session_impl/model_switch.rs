use super::*;
use crate::remote::DEFAULT_CONTEXT_WINDOW;
use xai_chat_state::conversation_util::replace_or_insert_system_head;

pub(super) struct PreparedAgentRebuild {
    agent: xai_grok_agent::Agent,
    agent_name: String,
    system_prompt: String,
    prompt_context: xai_grok_agent::PromptContext,
}

impl SessionActor {
    pub(super) async fn handle_set_session_model(
        &self,
        sampling_config: xai_grok_sampler::SamplerConfig,
        sampling_identity: xai_grok_sampling_types::SamplingIdentity,
        rebuild_definition: Option<Box<xai_grok_agent::AgentDefinition>>,
        use_concise: bool,
        apply_prompt_override: bool,
        skip_prompt_rewrite: bool,
        auto_compact_threshold_percent: u8,
    ) -> Result<acp::ModelId, acp::Error> {
        // Build a candidate harness first, but do not touch live agent state,
        // tool bindings, conversation, prompt artifacts, or persistence until
        // the authoritative sampling transition has committed.
        let prepared_rebuild = match rebuild_definition {
            Some(definition) => Some(self.prepare_agent_rebuild(*definition).await?),
            None => None,
        };

        // Build the complete proposal before asking the chat-state actor to
        // transition its three authoritative fields. The actor revalidates
        // history with the explicitly resolved runtime identity; rejection
        // leaves conversation/config/credentials inert.
        let current = self
            .chat_state_handle
            .get_sampling_state()
            .await
            .ok_or_else(|| acp::Error::internal_error().data("chat-state actor unavailable"))?;
        let model_id = acp::ModelId::new(sampling_config.model.clone());
        let new_context_window = self.compaction.context_window_override.unwrap_or_else(|| {
            std::num::NonZeroU64::new(sampling_config.context_window).unwrap_or_else(|| {
                std::num::NonZeroU64::new(DEFAULT_CONTEXT_WINDOW)
                    .expect("DEFAULT_CONTEXT_WINDOW is non-zero")
            })
        });
        let proposed_config = xai_grok_sampling_types::SamplingConfig {
            base_url: sampling_config.base_url.clone(),
            model: sampling_config.model.clone(),
            max_completion_tokens: sampling_config.max_completion_tokens,
            temperature: sampling_config.temperature,
            top_p: sampling_config.top_p,
            api_backend: sampling_config.api_backend.clone(),
            extra_headers: sampling_config.extra_headers.clone(),
            query_params: sampling_config.query_params.clone(),
            env_http_headers: sampling_config.env_http_headers.clone(),
            context_window: new_context_window,
            reasoning_effort: sampling_config.reasoning_effort,
            stream_tool_calls: Some(sampling_config.stream_tool_calls),
        };
        let session_key = self
            .auth_manager
            .as_ref()
            .and_then(|am| am.current_or_expired().map(|a| a.key));
        let proposed_credentials = xai_chat_state::Credentials {
            api_key: sampling_config.api_key.clone(),
            auth_type: crate::agent::config::resolve_chat_state_auth_type(
                sampling_config.model.as_str(),
                session_key.as_deref(),
                current.credentials.auth_type,
            ),
            alpha_test_key: current.credentials.alpha_test_key,
            client_version: sampling_config.client_version.clone(),
        };
        let transition = self
            .chat_state_handle
            .transition_sampling_state(proposed_config, proposed_credentials, sampling_identity)
            .await
            .ok_or_else(|| acp::Error::internal_error().data("chat-state actor unavailable"))?;
        match transition {
            Ok(applied) => {
                if let xai_chat_state::SamplingTransitionPersistence::CommittedWithBookkeepingWarning(error) = applied.persistence {
                    tracing::warn!(%error, "sampling identity history committed with bookkeeping warning");
                }
            }
            Err(xai_chat_state::SamplingTransitionError::History(error)) => {
                return Err(acp::Error::invalid_params().data(error.to_string()));
            }
            Err(error) => {
                return Err(acp::Error::internal_error().data(error.to_string()));
            }
        }

        if let Some(prepared) = prepared_rebuild {
            self.install_prepared_agent_rebuild(prepared).await;
        }

        let prev_threshold = self.compaction.threshold_percent.get();
        if prev_threshold != auto_compact_threshold_percent {
            tracing::info!(
                session_id = %self.session_info.id.0,
                new_model = %sampling_config.model,
                old_threshold = prev_threshold,
                new_threshold = auto_compact_threshold_percent,
                "auto_compact_threshold_percent updated for model switch"
            );
        }
        self.compaction
            .threshold_percent
            .set(auto_compact_threshold_percent);
        self.supports_backend_search
            .set(sampling_config.supports_backend_search);
        self.compactions_remaining
            .set(sampling_config.compactions_remaining);
        self.compaction_at_tokens
            .set(sampling_config.compaction_at_tokens);
        xai_grok_telemetry::unified_log::info(
            "backend_search: model switch",
            Some(self.session_info.id.0.as_ref()),
            Some(serde_json::json!({
                "new_model": &sampling_config.model,
                "api_backend": format!("{:?}", sampling_config.api_backend),
                "supports_backend_search": sampling_config.supports_backend_search,
            })),
        );
        self.invalidate_model_auth_memo();
        self.signals_handle()
            .record_model_usage(&sampling_config.model);
        if apply_prompt_override && !skip_prompt_rewrite {
            let mut conversation = self.chat_state_handle.get_conversation().await;
            for item in conversation.iter_mut() {
                if let ConversationItem::System(sys) = item {
                    if use_concise {
                        sys.content = std::sync::Arc::<str>::from(
                            xai_grok_agent::prompt::template::COMPACT_SYSTEM_PROMPT,
                        );
                    } else {
                        sys.content =
                            std::sync::Arc::<str>::from(self.agent.borrow().system_prompt());
                    }
                    break;
                }
            }
            self.chat_state_handle.replace_conversation(conversation);
        } else if !apply_prompt_override {
            tracing::info!(
                session_id = %self.session_info.id.0,
                model_id = %model_id.0,
                "handle_set_session_model: skipping prompt override (apply_prompt_override=false)"
            );
        } else {
            tracing::info!(
                session_id = %self.session_info.id.0,
                model_id = %model_id.0,
                "handle_set_session_model: skipping prompt rewrite (just rebuilt harness)"
            );
        }
        let agent_name = self.agent.borrow().definition().name.clone();
        let _ = self
            .notifications
            .persistence_tx
            .send(PersistenceMsg::CurrentModel {
                model_id: model_id.clone(),
                agent_name: Some(agent_name),
                reasoning_effort: Some(sampling_config.reasoning_effort),
            });
        Ok(model_id)
    }

    pub(super) async fn handle_override_model_name(
        &self,
        model_name: String,
        extra_headers: indexmap::IndexMap<String, String>,
        context_window: Option<std::num::NonZeroU64>,
    ) -> Result<(), acp::Error> {
        let current = self
            .chat_state_handle
            .get_sampling_state()
            .await
            .ok_or_else(|| acp::Error::internal_error().data("chat-state actor unavailable"))?;
        let mut proposed_config = current.config;
        let old_model = proposed_config.model.clone();
        let old_context_window = proposed_config.context_window.get();
        proposed_config.model = model_name.clone();
        proposed_config.extra_headers.extend(extra_headers.clone());
        if let Some(cw) = context_window
            && self.compaction.context_window_override.is_none()
        {
            proposed_config.context_window = cw;
        }

        let mut proposed_credentials = current.credentials;
        if let Some(resolved) = crate::agent::config::try_resolve_model_credentials(
            model_name.as_str(),
            proposed_credentials.api_key.as_deref(),
        ) {
            proposed_credentials.api_key = resolved.api_key;
            proposed_credentials.auth_type = resolved.auth_type;
        }

        let sampling_identity = xai_grok_sampler::resolve_runtime_sampling_identity(
            proposed_config.api_backend.clone(),
            &proposed_config.base_url,
            &proposed_config.model,
            &proposed_config.extra_headers,
            &proposed_config.env_http_headers,
        )
        .map_err(|error| acp::Error::invalid_params().data(error.to_string()))?;
        let transition = self
            .chat_state_handle
            .transition_sampling_state(proposed_config, proposed_credentials, sampling_identity)
            .await
            .ok_or_else(|| acp::Error::internal_error().data("chat-state actor unavailable"))?;
        match transition {
            Ok(applied) => {
                if let xai_chat_state::SamplingTransitionPersistence::CommittedWithBookkeepingWarning(error) = applied.persistence {
                    tracing::warn!(%error, "override history committed with bookkeeping warning");
                }
            }
            Err(xai_chat_state::SamplingTransitionError::History(error)) => {
                return Err(acp::Error::invalid_params().data(error.to_string()));
            }
            Err(error) => {
                return Err(acp::Error::internal_error().data(error.to_string()));
            }
        }

        tracing::info!(
            target: SESSION_LOG,
            session_id = %self.session_info.id,
            old_model = %old_model,
            new_model = %model_name,
            extra_header_count = extra_headers.len(),
            old_context_window,
            new_context_window = ?context_window.map(|cw| cw.get()),
            "OVERRIDE_MODEL: changing model name in sampling config"
        );
        // Shell-owned secondary state moves only after the actor's
        // authoritative transition succeeds.
        self.signals_handle().set_primary_model(&model_name);
        self.invalidate_model_auth_memo();
        Ok(())
    }

    /// Build and render a candidate zero-turn harness without mutating any
    /// live session, conversation, tool binding, or persistence state.
    pub(super) async fn prepare_agent_rebuild(
        &self,
        definition: xai_grok_agent::AgentDefinition,
    ) -> Result<PreparedAgentRebuild, acp::Error> {
        {
            let state = self.state.lock().await;
            if state.running_task.is_some() {
                tracing::warn!(
                    session_id = %self.session_info.id.0,
                    new_agent_type = %definition.name,
                    "prepare_agent_rebuild: turn in flight, rejecting rebuild"
                );
                return Err(acp::Error::internal_error()
                    .data("rebuild_agent: turn in flight, refusing to rebuild harness"));
            }
        }
        let new_agent_name = definition.name.clone();
        tracing::info!(
            session_id = %self.session_info.id.0,
            new_agent_type = %new_agent_name,
            "prepare_agent_rebuild: rebuilding harness"
        );
        let new_agent = self
            .rebuild_spec
            .build_agent(definition)
            .await
            .map_err(|e| {
                tracing::error!(
                    session_id = %self.session_info.id.0,
                    new_agent_type = %new_agent_name,
                    error = %e,
                    "prepare_agent_rebuild: AgentBuilder::build failed"
                );
                acp::Error::internal_error().data(format!(
                    "rebuild_agent: build failed for agent_type={new_agent_name}: {e}"
                ))
            })?;
        let system_prompt = new_agent.system_prompt().to_string();
        let mut prompt_context = new_agent.prompt_context().clone();
        prompt_context.normalize_for_persistence();
        Ok(PreparedAgentRebuild {
            agent: new_agent,
            agent_name: new_agent_name,
            system_prompt,
            prompt_context,
        })
    }

    /// Install a fully prepared harness after the authoritative sampling
    /// transition succeeds. This retains the existing tool/resource rebinding,
    /// prompt rewrite, persistence, and notification behavior.
    pub(super) async fn install_prepared_agent_rebuild(&self, prepared: PreparedAgentRebuild) {
        let PreparedAgentRebuild {
            agent: new_agent,
            agent_name: new_agent_name,
            system_prompt: new_system_prompt,
            prompt_context: new_prompt_context,
        } = prepared;
        if let Some(handle) = self.compaction.prefire.take_handle() {
            handle.abort();
            let _ = handle.await;
            self.compaction.prefire.finish();
        }
        self.compaction.prefire.clear();
        *self.agent.borrow_mut() = new_agent;
        *self.active_agent_type.lock() = Some(new_agent_name.clone());
        self.emit_resolved_tool_overrides();
        self.queue_exit_reminder_on_approved_exit.store(
            self.is_cursor_harness(),
            std::sync::atomic::Ordering::Relaxed,
        );
        if let Err(e) = self.workspace_ops.bind_local_session(
            &self.session_id_string(),
            self.tool_context.cwd.as_path().to_path_buf(),
            self.tool_context.hunk_tracker_handle.clone(),
            self.agent.borrow().tool_bridge().toolset(),
            None,
        ) {
            tracing::warn!(error = %e, "failed to rebind local session toolset after agent rebuild");
        }
        {
            let bridge = self.agent.borrow().tool_bridge().clone();
            let snapshot = self.tool_metadata_snapshot.clone();
            let tool_index = crate::session::tool_index::Bm25ToolSearchIndex::new(snapshot);
            bridge
                .update_resource(xai_grok_tools::types::tool_index::ToolIndex(
                    std::sync::Arc::new(tool_index),
                ))
                .await;
            if let Some(client) = self.rebuild_spec.managed_gateway_tool_client.clone() {
                bridge.update_resource(client).await;
            }
            let plan_path = self.plan_mode.lock().plan_file_path().to_path_buf();
            bridge
                .update_resource(xai_grok_tools::types::resources::PlanFilePath(plan_path))
                .await;
            if let Some(display_cwd) = self.display_cwd.get() {
                bridge
                    .set_display_cwd(std::path::PathBuf::from(display_cwd))
                    .await;
            }
            bridge
                .update_resource(
                    xai_grok_tools::implementations::grok_build::workflow::WorkflowLaunchHandle(
                        self.workflow_launch_tx.clone(),
                    ),
                )
                .await;
            if !self.goal_runs_on_workflow_engine() {
                bridge
                    .update_resource(
                        xai_grok_tools::implementations::grok_build::update_goal::GoalUpdateHandle(
                            self.goal_update_tx.clone(),
                        ),
                    )
                    .await;
            }
            if let Some(reservations) = self.tool_context.task_completion_reservations.clone() {
                bridge.update_resource(reservations).await;
            }
            if let Some(gate) = self.tool_context.task_wake_suppressed.clone() {
                bridge.update_resource(gate).await;
            }
            self.inject_deny_read_globs().await;
        }
        {
            let notified = self.mcp_handshakes_done.notified();
            tokio::pin!(notified);
            let needs_wait = {
                let s = self.mcp_state.lock().await;
                !s.configs.is_empty() && !s.is_initialized()
            };
            if needs_wait {
                const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
                tokio::select! {
                    () = &mut notified => {}
                    () = tokio::time::sleep(TIMEOUT) => {
                        tracing::warn!(
                            session_id = %self.session_info.id.0,
                            "install_prepared_agent_rebuild: timed out waiting for MCP handshakes"
                        );
                    }
                }
            }
        }
        self.re_register_mcp_tools_on_rebuilt_bridge().await;
        if let Some(old_handle) = self.deferred_prefix.take() {
            old_handle.abort();
        }
        let new_user_prefix = self.build_user_message_prefix().await;
        {
            let mut conversation = self.chat_state_handle.get_conversation().await;
            let _ = replace_or_insert_system_head(&mut conversation, &new_system_prompt);
            let drop_startup_skill_reminder = false;
            Self::rewrite_zero_turn_prefix(
                &mut conversation,
                new_user_prefix,
                drop_startup_skill_reminder,
            );
            if !conversation_has_project_instructions(&conversation)
                && let Some(agents_md_reminder) = self.agent.borrow().agents_md_user_reminder()
            {
                let agents_md_at = conversation.len().min(2);
                conversation.insert(
                    agents_md_at,
                    ConversationItem::project_instructions(agents_md_reminder),
                );
            }
            self.inject_baseline_skill_reminder(&mut conversation).await;
            self.chat_state_handle.replace_conversation(conversation);
        }
        save_prompt_context(&self.session_info, &new_prompt_context);
        save_system_prompt(&self.session_info, &new_system_prompt);
        let snapshot = self.chat_state_handle.get_conversation().await;
        persist_chat_history_jsonl_sync(&self.session_info, &snapshot);
        self.mcp_reminder_dirty
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.send_available_commands_update().await;
        tracing::info!(
            session_id = %self.session_info.id.0,
            new_agent_type = %new_agent_name,
            "install_prepared_agent_rebuild: harness rebuild complete"
        );
    }
    /// Apply a client-supplied `systemPromptOverride` on session attach without
    /// wiping user/assistant history: swap only the leading `System` message,
    /// atomically inside the `ChatStateActor` (see
    /// `ChatStateCommand::ReplaceSystemHead` for the serialization guarantees).
    /// `system_prompt.txt` (not owned by the persistence actor) is saved
    /// directly, even on a head no-op, so a previously-diverged secondary
    /// artifact self-heals. Skipped entirely on a verbatim mirror-fork
    /// (`preserve_inherited_system`).
    pub(super) async fn handle_replace_system_prompt(&self, system_prompt: String) {
        if self.startup_hints.preserve_inherited_system {
            tracing::debug!(
                session_id = %self.session_info.id.0,
                "handle_replace_system_prompt: skipped (preserve_inherited_system)"
            );
            return;
        }
        let Some(changed) = self
            .chat_state_handle
            .replace_system_head(&system_prompt)
            .await
        else {
            tracing::error!(
                session_id = %self.session_info.id.0,
                "handle_replace_system_prompt: chat-state actor unavailable; override not applied"
            );
            return;
        };
        save_system_prompt(&self.session_info, &system_prompt);
        if changed {
            tracing::info!(
                session_id = %self.session_info.id.0,
                prompt_len = system_prompt.len(),
                "handle_replace_system_prompt: client override applied"
            );
        } else {
            tracing::debug!(
                session_id = %self.session_info.id.0,
                "handle_replace_system_prompt: head already matches, no-op"
            );
        }
    }
}
