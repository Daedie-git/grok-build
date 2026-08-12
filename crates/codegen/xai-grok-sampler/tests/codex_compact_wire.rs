mod support;

use std::sync::{Arc, Mutex};

use axum::Json;
use axum::Router;
use axum::extract::Request;
use axum::http::{HeaderMap, StatusCode, Uri};
use axum::routing::post;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use xai_grok_sampler::SamplingClient;
use xai_grok_sampling_types::{
    CHATGPT_ACCOUNT_ID_HEADER, CodexCompactResponse, ConversationItem, ConversationRequest,
    NativeCompactionCompatibility, ReasoningEffort, SamplingError, ToolSpec, TurnRoutingState,
    codex_compact_output_to_conversation,
};

#[derive(Debug)]
struct CapturedRequest {
    uri: Uri,
    headers: HeaderMap,
    body: Value,
}

fn codex_test_config(base_url: &str, api_key: &str) -> xai_grok_sampler::SamplerConfig {
    let mut config = support::test_config(base_url, api_key);
    config.api_backend = xai_grok_sampling_types::ApiBackend::Responses;
    config.provider_id = Some(xai_grok_sampling_types::ProviderId::Codex);
    config
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_compact_uses_exact_path_headers_body_and_decodes_replacement() {
    let captured: Arc<Mutex<Option<CapturedRequest>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&captured);
    let app = Router::new().route(
        "/v1/responses/compact",
        post(move |request: Request| {
            let sink = Arc::clone(&sink);
            async move {
                let (parts, body) = request.into_parts();
                let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
                let body = serde_json::from_slice(&bytes).unwrap();
                *sink.lock().unwrap() = Some(CapturedRequest {
                    uri: parts.uri,
                    headers: parts.headers,
                    body,
                });
                Json(json!({
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_native_retained",
                            "status": "completed",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "original objective"}],
                            "internal_chat_message_metadata_passthrough": {"turn_id": "turn-live-message"}
                        },
                        {
                            "type": "reasoning",
                            "id": "rs_native",
                            "summary": [],
                            "encrypted_content": "encrypted-reasoning",
                            "status": "completed",
                            "internal_chat_message_metadata_passthrough": {"turn_id": "turn-live-reasoning"}
                        },
                        {
                            "type": "compaction",
                            "id": "cmp_native",
                            "encrypted_content": "encrypted-replacement",
                            "internal_chat_message_metadata_passthrough": {"turn_id": "turn-live-compaction"}
                        }
                    ]
                }))
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let mut cfg = codex_test_config(&format!("http://{addr}/v1"), "test-codex-token");
    cfg.extra_headers
        .insert(CHATGPT_ACCOUNT_ID_HEADER.into(), "acct_test_native".into());
    let client = SamplingClient::new(cfg).expect("client builds");
    let response = client
        .conversation_compact_responses(ConversationRequest {
            items: vec![
                ConversationItem::system("Follow repository instructions."),
                ConversationItem::user("original objective"),
            ],
            tools: vec![ToolSpec {
                name: "read_file".into(),
                description: Some("Read a file".into()),
                parameters: json!({
                    "type": "object",
                    "properties": {"target_file": {"type": "string"}},
                    "required": ["target_file"]
                }),
            }],
            model: Some("gpt-5.6-sol".into()),
            temperature: Some(1.0),
            top_p: Some(0.95),
            max_output_tokens: Some(16_384),
            reasoning_effort: Some(ReasoningEffort::Medium),
            prompt_cache_key: Some("cache-native".into()),
            x_grok_conv_id: Some("conv-native".into()),
            x_grok_req_id: Some("req-native".into()),
            x_grok_session_id: Some("session-native".into()),
            x_grok_turn_idx: Some("7".into()),
            x_grok_agent_id: Some("agent-native".into()),
            ..Default::default()
        })
        .await
        .expect("compact succeeds");

    assert_eq!(response.output.len(), 3);
    assert!(matches!(
        &response.output[0],
        xai_grok_sampling_types::CodexCompactOutputItem::Message { id, .. }
            if id.as_deref() == Some("msg_native_retained")
    ));
    assert!(matches!(
        &response.output[2],
        xai_grok_sampling_types::CodexCompactOutputItem::Compaction {
            id,
            internal_chat_message_metadata_passthrough: Some(metadata),
            ..
        } if id.as_deref() == Some("cmp_native")
            && metadata.turn_id.as_deref() == Some("turn-live-compaction")
    ));
    let captured = captured.lock().unwrap().take().expect("request captured");
    assert_eq!(captured.uri.path(), "/v1/responses/compact");
    assert_eq!(
        captured.headers.get("authorization").unwrap(),
        "Bearer test-codex-token"
    );
    assert_eq!(
        captured.headers.get(CHATGPT_ACCOUNT_ID_HEADER).unwrap(),
        "acct_test_native"
    );
    assert_eq!(
        captured.headers.get("x-grok-conv-id").unwrap(),
        "conv-native"
    );
    assert_eq!(captured.headers.get("x-grok-req-id").unwrap(), "req-native");
    assert_eq!(
        captured.headers.get("x-grok-session-id").unwrap(),
        "session-native"
    );
    assert_eq!(captured.headers.get("x-grok-turn-idx").unwrap(), "7");
    assert_eq!(
        captured.headers.get("x-grok-agent-id").unwrap(),
        "agent-native"
    );
    assert_eq!(captured.body["model"], "gpt-5.6-sol");
    assert_eq!(
        captured.body["instructions"],
        "Follow repository instructions."
    );
    assert_eq!(captured.body["prompt_cache_key"], "cache-native");
    assert_eq!(captured.body["parallel_tool_calls"], true);
    assert!(captured.body["tools"].is_array());
    assert!(captured.body["reasoning"].is_object());
    for unsupported in [
        "temperature",
        "top_p",
        "max_output_tokens",
        "stream",
        "store",
        "include",
    ] {
        assert!(
            captured.body.get(unsupported).is_none(),
            "native compact must omit {unsupported}: {:#}",
            captured.body
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unsupported_provider_neither_sends_nor_captures_turn_routing_state() {
    let ordinary_headers = Arc::new(Mutex::new(Vec::<Option<String>>::new()));
    let compact_headers = Arc::new(Mutex::new(Vec::<Option<String>>::new()));
    let ordinary_sink = Arc::clone(&ordinary_headers);
    let compact_sink = Arc::clone(&compact_headers);
    let app = Router::new()
        .route(
            "/v1/responses",
            post(move |request: Request| {
                let ordinary_sink = Arc::clone(&ordinary_sink);
                async move {
                    ordinary_sink.lock().unwrap().push(
                        request
                            .headers()
                            .get("x-codex-turn-state")
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned),
                    );
                    axum::response::Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "text/event-stream")
                        .header("x-codex-turn-state", "server-ordinary")
                        .body(axum::body::Body::from("data: [DONE]\n\n"))
                        .unwrap()
                }
            }),
        )
        .route(
            "/v1/responses/compact",
            post(move |request: Request| {
                let compact_sink = Arc::clone(&compact_sink);
                async move {
                    compact_sink.lock().unwrap().push(
                        request
                            .headers()
                            .get("x-codex-turn-state")
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned),
                    );
                    let mut response = axum::response::Response::new(axum::body::Body::from(
                        serde_json::to_vec(&json!({
                            "output": [{"type": "compaction", "encrypted_content": "opaque"}]
                        }))
                        .unwrap(),
                    ));
                    response
                        .headers_mut()
                        .insert("content-type", "application/json".parse().unwrap());
                    response
                        .headers_mut()
                        .insert("x-codex-turn-state", "server-compact".parse().unwrap());
                    response
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    // Explicit non-Codex identity must win even though these test routes emit
    // Codex-specific headers.
    let mut config = support::test_config(&format!("http://{addr}/v1"), "test-token");
    config.api_backend = xai_grok_sampling_types::ApiBackend::Responses;
    config.provider_id = Some(xai_grok_sampling_types::ProviderId::Custom);
    let client = SamplingClient::new(config).unwrap();
    let populated = TurnRoutingState::fresh();
    assert!(populated.capture_first("client-state".to_string()));
    let request = ConversationRequest {
        items: vec![ConversationItem::user("hello")],
        turn_routing_state: Some(populated.clone()),
        ..Default::default()
    };
    let (stream, _, _) = client
        .conversation_stream_responses(request.clone())
        .await
        .expect("ordinary request succeeds");
    drop(stream);
    assert!(matches!(
        client.conversation_compact_responses(request).await,
        Err(SamplingError::InvalidConfiguration(_))
    ));
    assert_eq!(populated.value(), Some("client-state"));

    let fresh = TurnRoutingState::fresh();
    let (stream, _, _) = client
        .conversation_stream_responses(ConversationRequest {
            items: vec![ConversationItem::user("next")],
            turn_routing_state: Some(fresh.clone()),
            ..Default::default()
        })
        .await
        .expect("second ordinary request succeeds");
    drop(stream);

    assert!(
        fresh.value().is_none(),
        "unsupported response must not capture"
    );
    assert_eq!(*ordinary_headers.lock().unwrap(), vec![None, None]);
    assert!(
        compact_headers.lock().unwrap().is_empty(),
        "unsupported provider must reject native compact before HTTP"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_native_manifest_is_rejected_by_both_endpoints_before_http() {
    let requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let request_count = Arc::clone(&requests);
    let app = Router::new()
        .route(
            "/v1/responses",
            post(move || {
                request_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { StatusCode::NO_CONTENT }
            }),
        )
        .route(
            "/v1/responses/compact",
            post({
                let requests = Arc::clone(&requests);
                move || {
                    requests.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    async { StatusCode::NO_CONTENT }
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    let client = SamplingClient::new(codex_test_config(
        &format!("http://{addr}/v1"),
        "test-token",
    ))
    .unwrap();

    let response: CodexCompactResponse = serde_json::from_value(json!({
        "output": [
            {
                "type": "message",
                "id": "msg-retained",
                "role": "user",
                "content": [{"type": "input_text", "text": "retained"}]
            },
            {
                "type": "reasoning",
                "id": "rs-retained",
                "summary": [],
                "encrypted_content": "reasoning-opaque"
            },
            {"type": "compaction", "id": "cmp-retained", "encrypted_content": "opaque"}
        ]
    }))
    .unwrap();
    let valid = codex_compact_output_to_conversation(
        response.output,
        NativeCompactionCompatibility::codex("test-model", None),
    )
    .unwrap();
    let mut invalid_histories = Vec::new();

    fn manifest_mut(items: &mut [ConversationItem]) -> &mut NativeCompactionCompatibility {
        items
            .iter_mut()
            .find_map(|item| match item {
                ConversationItem::Provider(provider) => {
                    provider.as_native_compaction_metadata_mut()
                }
                _ => None,
            })
            .unwrap()
    }

    let mut missing_manifest = valid.clone();
    manifest_mut(&mut missing_manifest).item_metadata.clear();
    invalid_histories.push(missing_manifest);

    let mut missing_middle = valid.clone();
    manifest_mut(&mut missing_middle).item_metadata.remove(1);
    invalid_histories.push(missing_middle);

    let mut extra = valid.clone();
    let extra_entry = manifest_mut(&mut extra).item_metadata[1].clone();
    manifest_mut(&mut extra).item_metadata.push(extra_entry);
    invalid_histories.push(extra);

    let mut duplicate = valid.clone();
    manifest_mut(&mut duplicate).item_metadata[1].input_index = 0;
    invalid_histories.push(duplicate);

    let mut wrong_index = valid.clone();
    manifest_mut(&mut wrong_index).item_metadata[1].input_index = 9;
    invalid_histories.push(wrong_index);

    let mut wrong_kind = valid.clone();
    manifest_mut(&mut wrong_kind).item_metadata[1].kind =
        xai_grok_sampling_types::NativeCompactionItemKind::Message;
    invalid_histories.push(wrong_kind);

    let mut wrong_id = valid.clone();
    manifest_mut(&mut wrong_id).item_metadata[1].item_id = Some("rs-other".into());
    invalid_histories.push(wrong_id);

    let mut wrong_segment_length = valid.clone();
    manifest_mut(&mut wrong_segment_length).replacement_segment_items -= 1;
    invalid_histories.push(wrong_segment_length);

    let mut missing_descriptor = valid.clone();
    missing_descriptor.retain(|item| {
        !matches!(item, ConversationItem::Provider(provider) if provider.is_native_compaction_metadata())
    });
    invalid_histories.push(missing_descriptor);

    let mut legacy = valid;
    manifest_mut(&mut legacy).schema_version = 1;
    invalid_histories.push(legacy);

    for items in invalid_histories {
        let request = ConversationRequest {
            items,
            model: Some("test-model".into()),
            ..Default::default()
        };
        assert!(matches!(
            client.conversation_stream_responses(request.clone()).await,
            Err(SamplingError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            client.conversation_compact_responses(request).await,
            Err(SamplingError::InvalidConfiguration(_))
        ));
    }
    assert_eq!(
        requests.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "invalid bindings must fail before either HTTP endpoint"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_compact_preserves_actionable_detail_error() {
    let app = Router::new().route(
        "/v1/responses/compact",
        post(|| async {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "detail": "input[3] has unsupported type function_call_output"
                })),
            )
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let client = SamplingClient::new(codex_test_config(
        &format!("http://{addr}/v1"),
        "test-token",
    ))
    .unwrap();
    let error = client
        .conversation_compact_responses(ConversationRequest {
            items: vec![ConversationItem::user("hello")],
            ..Default::default()
        })
        .await
        .expect_err("400 must fail");
    match error {
        SamplingError::Api {
            status, message, ..
        } => {
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                message,
                "input[3] has unsupported type function_call_output"
            );
        }
        other => panic!("expected API error, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_compact_classifies_unauthorized_as_auth() {
    let app = Router::new().route(
        "/v1/responses/compact",
        post(|| async {
            (
                StatusCode::UNAUTHORIZED,
                Json(json!({"detail": "access token expired"})),
            )
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let client = SamplingClient::new(codex_test_config(
        &format!("http://{addr}/v1"),
        "test-token",
    ))
    .unwrap();
    let error = client
        .conversation_compact_responses(ConversationRequest {
            items: vec![ConversationItem::user("hello")],
            ..Default::default()
        })
        .await
        .expect_err("401 must fail");
    match error {
        SamplingError::Auth {
            message,
            credential,
        } => {
            assert!(message.contains("access token expired"), "{message}");
            assert!(message.contains("responses/compact"), "{message}");
            assert_eq!(credential, xai_grok_sampling_types::SentCredential::Sent);
        }
        other => panic!("expected auth error, got {other:?}"),
    }
}
