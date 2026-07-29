//! `AcpTerminalAdapter`: implements `xai-grok-tools::TerminalBackend` over ACP
//! gateway calls, for bash execution when the terminal is served by the client.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::exit_watcher::{poll_for_terminal_exit, release_terminal, watch_for_exit};
use super::output_recorder::{OutputRecorder, read_log_tail};
use agent_client_protocol as acp;
use xai_acp_lib::AcpAgentGatewaySender as GatewaySender;
use xai_grok_tools::computer::types::{
    BackgroundHandle, ComputerError, KillOutcome, TaskKind, TaskSnapshot, TerminalBackend,
    TerminalRunRequest, TerminalRunResult,
};

/// A snapshot's per-completion fields, grouped to avoid transposed positional args.
#[derive(Clone)]
pub(super) struct SnapshotOutput {
    pub(super) output: String,
    pub(super) truncated: bool,
    pub(super) exit_code: Option<i32>,
    pub(super) signal: Option<String>,
}

pub(super) struct TrackedTask {
    command: String,
    display_command: Option<String>,
    cwd: String,
    output_file: PathBuf,
    start_time: std::time::SystemTime,
    /// Stamped once when the task completes, so repeated snapshots agree.
    end_time: Option<std::time::SystemTime>,
    completed: bool,
    exit_code: Option<i32>,
    signal: Option<String>,
    last_output: String,
    last_truncated: bool,
    /// Persisted once a blocking waiter receives the completed snapshot.
    block_waited: bool,
    /// Blocking waits that can still receive this task's completion.
    active_block_waiters: usize,
    explicitly_killed: bool,
    kind: TaskKind,
    owner_session_id: Option<String>,
    description: Option<String>,
    output_byte_limit: usize,
}

/// Hand-written (`SystemTime` has no `Default`); call sites spread from it.
impl Default for TrackedTask {
    fn default() -> Self {
        Self {
            command: String::new(),
            display_command: None,
            cwd: String::new(),
            output_file: PathBuf::new(),
            start_time: std::time::SystemTime::now(),
            end_time: None,
            completed: false,
            exit_code: None,
            signal: None,
            last_output: String::new(),
            last_truncated: false,
            block_waited: false,
            active_block_waiters: 0,
            explicitly_killed: false,
            kind: TaskKind::Bash,
            owner_session_id: None,
            description: None,
            output_byte_limit: crate::terminal::DEFAULT_OUTPUT_BYTE_LIMIT,
        }
    }
}

impl TrackedTask {
    pub(super) fn mark_completed(&mut self, out: SnapshotOutput) {
        self.completed = true;
        self.end_time = Some(std::time::SystemTime::now());
        self.exit_code = out.exit_code;
        self.signal = out.signal;
        self.last_output = out.output;
        self.last_truncated = out.truncated;
    }

    pub(super) fn to_snapshot(&self, task_id: &str, out: SnapshotOutput) -> TaskSnapshot {
        let completed = self.completed || out.exit_code.is_some() || out.signal.is_some();
        TaskSnapshot {
            task_id: task_id.to_string(),
            command: self.command.clone(),
            display_command: self.display_command.clone(),
            cwd: self.cwd.clone(),
            start_time: self.start_time,
            end_time: self
                .end_time
                .or_else(|| completed.then(std::time::SystemTime::now)),
            output: out.output,
            output_file: self.output_file.clone(),
            truncated: out.truncated,
            exit_code: out.exit_code,
            signal: out.signal,
            completed,
            // An active waiter has not delivered anything yet. Publishing that
            // speculative state can suppress the only completion wake if the
            // wait future is cancelled before `finish_with_snapshot` commits.
            block_waited: self.block_waited,
            explicitly_killed: self.explicitly_killed,
            kind: self.kind,
            owner_session_id: self.owner_session_id.clone(),
            description: self.description.clone(),
            // ACP tracked tasks are only registered via run_background.
            is_backgrounded: true,
        }
    }
}

pub(super) type TaskMap = Arc<Mutex<HashMap<String, TrackedTask>>>;

/// Registers one blocking wait for exactly as long as its future can still
/// deliver the task's completed snapshot.
///
/// The guard borrows the task map and task id so its lifetime is scoped to one
/// `wait_for_completion` future. Dropping that future (including cancellation)
/// removes only its registration, leaving concurrent waiters visible.
struct BlockWaitGuard<'a> {
    tasks: &'a TaskMap,
    task_id: &'a str,
    registered: bool,
}

impl<'a> BlockWaitGuard<'a> {
    fn new(tasks: &'a TaskMap, task_id: &'a str) -> Self {
        let mut tracked_tasks = tasks.lock().unwrap();
        let registered = tracked_tasks.get_mut(task_id).is_some_and(|task| {
            if let Some(next) = task.active_block_waiters.checked_add(1) {
                task.active_block_waiters = next;
                true
            } else {
                tracing::error!(task_id, "too many concurrent blocking task waiters");
                false
            }
        });
        drop(tracked_tasks);

        Self {
            tasks,
            task_id,
            registered,
        }
    }

    /// Reconciles the queried snapshot with tracked completion state and ends
    /// this registration in the same critical section. If the exit watcher
    /// completed the task after `get_task` returned a running snapshot, this
    /// upgrades the stale result before persisting `block_waited`.
    fn finish_with_snapshot(&mut self, mut snapshot: Option<TaskSnapshot>) -> Option<TaskSnapshot> {
        if !self.registered {
            if let Some(snapshot) = snapshot.as_mut()
                && snapshot.completed
            {
                snapshot.block_waited = true;
            }
            return snapshot;
        }

        let mut tasks = self
            .tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let block_waited = tasks.get_mut(self.task_id).is_some_and(|task| {
            if !snapshot.as_ref().is_some_and(|snapshot| snapshot.completed) && task.completed {
                snapshot = Some(task.to_snapshot(
                    self.task_id,
                    SnapshotOutput {
                        output: task.last_output.clone(),
                        truncated: task.last_truncated,
                        exit_code: task.exit_code,
                        signal: task.signal.clone(),
                    },
                ));
            }

            debug_assert!(task.active_block_waiters > 0);
            task.active_block_waiters = task.active_block_waiters.saturating_sub(1);
            if snapshot.as_ref().is_some_and(|snapshot| snapshot.completed) {
                task.block_waited = true;
            }
            task.block_waited || task.active_block_waiters > 0
        });
        self.registered = false;
        drop(tasks);

        if let Some(snapshot) = snapshot.as_mut() {
            snapshot.block_waited = block_waited;
        }
        snapshot
    }

    fn cancel(&mut self) {
        if !self.registered {
            return;
        }

        let mut tasks = self
            .tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(task) = tasks.get_mut(self.task_id) {
            debug_assert!(task.active_block_waiters > 0);
            task.active_block_waiters = task.active_block_waiters.saturating_sub(1);
        }
        self.registered = false;
    }
}

impl Drop for BlockWaitGuard<'_> {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn wrap_command(command: &str) -> Result<String, ComputerError> {
    #[cfg(not(unix))]
    {
        let _ = command;
        Ok(command.to_string())
    }
    #[cfg(unix)]
    {
        let quoted = shlex::try_quote(command).map_err(|_| ComputerError::CommandNotQuoted)?;
        Ok(format!(
            "{} -lc {quoted}",
            crate::terminal::default_shell_path()
        ))
    }
}

fn to_env(env: HashMap<String, String>) -> Vec<acp::EnvVariable> {
    env.into_iter()
        .map(|(name, value)| acp::EnvVariable::new(name, value))
        .collect()
}

pub(super) fn parse_exit(
    status: &Option<acp::TerminalExitStatus>,
) -> (Option<i32>, Option<String>) {
    match status {
        Some(e) => (e.exit_code.map(|v| v as i32), e.signal.clone()),
        None => (None, None),
    }
}

/// Wraps xai-grok-shell's ACP gateway to satisfy xai-grok-tools' TerminalBackend.
pub struct AcpTerminalAdapter {
    gateway: GatewaySender,
    session_id: acp::SessionId,
    tasks: TaskMap,
}

impl AcpTerminalAdapter {
    pub fn new(gateway: GatewaySender, session_id: acp::SessionId) -> Self {
        Self {
            gateway,
            session_id,
            tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    async fn create_terminal(
        &self,
        command: String,
        request: &TerminalRunRequest,
    ) -> Result<acp::CreateTerminalResponse, ComputerError> {
        self.gateway
            .send(
                acp::CreateTerminalRequest::new(self.session_id.clone(), command)
                    .args(vec![])
                    .env(to_env(request.env.clone()))
                    .cwd(Some(request.working_directory.clone()))
                    .output_byte_limit(Some(request.output_byte_limit as u64)),
            )
            .await
            .map_err(|e| ComputerError::io(e.to_string()))
    }

    fn terminal_id(&self, task_id: &str) -> acp::TerminalId {
        acp::TerminalId::new(task_id)
    }
}

#[async_trait::async_trait]
impl TerminalBackend for AcpTerminalAdapter {
    async fn run(&self, request: TerminalRunRequest) -> Result<TerminalRunResult, ComputerError> {
        let command = wrap_command(&request.command)?;
        let create_res = self.create_terminal(command, &request).await?;

        let timed_out = match tokio::time::timeout(
            request.timeout,
            self.gateway.send(acp::WaitForTerminalExitRequest::new(
                self.session_id.clone(),
                create_res.terminal_id.clone(),
            )),
        )
        .await
        {
            Ok(Ok(_)) => false,
            Ok(Err(e)) => {
                release_terminal(&self.gateway, &self.session_id, &create_res.terminal_id).await;
                return Err(ComputerError::io(e.to_string()));
            }
            Err(_) => {
                let _ = self
                    .gateway
                    .send(acp::KillTerminalRequest::new(
                        self.session_id.clone(),
                        create_res.terminal_id.clone(),
                    ))
                    .await;
                true
            }
        };

        let output = match self
            .gateway
            .send(acp::TerminalOutputRequest::new(
                self.session_id.clone(),
                create_res.terminal_id.clone(),
            ))
            .await
        {
            Ok(output) => output,
            Err(e) => {
                release_terminal(&self.gateway, &self.session_id, &create_res.terminal_id).await;
                return Err(ComputerError::io(e.to_string()));
            }
        };

        release_terminal(&self.gateway, &self.session_id, &create_res.terminal_id).await;

        let (exit_code, signal) = parse_exit(&output.exit_status);
        let total_bytes = output.output.len();

        let mut recorder =
            OutputRecorder::new(request.output_file.clone(), request.output_byte_limit);
        recorder.initialize().await;
        if let Err(e) = recorder.append(&output.output).await {
            tracing::warn!(error = %e, "output recorder failed to write foreground output");
        }

        Ok(TerminalRunResult {
            combined_output: output.output,
            exit_code,
            truncated: output.truncated,
            signal,
            timed_out,
            output_file: request.output_file,
            total_bytes,
            pid: None,
        })
    }

    async fn run_background(
        &self,
        request: TerminalRunRequest,
    ) -> Result<BackgroundHandle, ComputerError> {
        let command = wrap_command(&request.command)?;
        let notification_handle = request.notification_handle.clone();
        let display_command = request.display_command.clone();
        let cwd = request.working_directory.to_string_lossy().to_string();
        let output_file = request.output_file.clone();

        let create_res = self.create_terminal(command.clone(), &request).await?;
        let task_id = create_res.terminal_id.0.to_string();
        let description = request.description;

        {
            let mut tasks = self.tasks.lock().unwrap();
            tasks.insert(
                task_id.clone(),
                TrackedTask {
                    command,
                    display_command,
                    cwd,
                    output_file: output_file.clone(),
                    kind: request.kind,
                    owner_session_id: request.owner_session_id.clone(),
                    description,
                    output_byte_limit: request.output_byte_limit,
                    ..Default::default()
                },
            );
        }

        let recorder = OutputRecorder::new(output_file.clone(), request.output_byte_limit);
        recorder.initialize().await;
        tokio::spawn(watch_for_exit(
            self.gateway.clone(),
            self.session_id.clone(),
            task_id.clone(),
            Arc::clone(&self.tasks),
            notification_handle,
            recorder,
        ));

        Ok(BackgroundHandle {
            task_id,
            output_file,
            pid: None,
        })
    }

    async fn get_task(&self, task_id: &str) -> Option<TaskSnapshot> {
        let live = self
            .gateway
            .send(acp::TerminalOutputRequest::new(
                self.session_id.clone(),
                self.terminal_id(task_id),
            ))
            .await
            .ok();

        // The std Mutex guard cannot be held across the await below, so resolve
        // under the lock and read the log file after releasing it.
        enum Resolved {
            Ready(TaskSnapshot),
            FromLog {
                snapshot: TaskSnapshot,
                output_file: PathBuf,
                limit: usize,
            },
            Missing,
        }
        let resolved = {
            let tasks = self.tasks.lock().unwrap();
            match (live, tasks.get(task_id)) {
                (Some(output), Some(tracked)) => {
                    let (exit_code, signal) = parse_exit(&output.exit_status);
                    Resolved::Ready(tracked.to_snapshot(
                        task_id,
                        SnapshotOutput {
                            output: output.output,
                            truncated: output.truncated,
                            exit_code,
                            signal,
                        },
                    ))
                }
                (Some(output), None) => {
                    let (exit_code, signal) = parse_exit(&output.exit_status);
                    Resolved::Ready(TrackedTask::default().to_snapshot(
                        task_id,
                        SnapshotOutput {
                            output: output.output,
                            truncated: output.truncated,
                            exit_code,
                            signal,
                        },
                    ))
                }
                // Live poll failed: a completed task keeps its authoritative
                // last_output; only a still-running task falls back to the log.
                (None, Some(tracked)) => {
                    let snapshot = tracked.to_snapshot(
                        task_id,
                        SnapshotOutput {
                            output: tracked.last_output.clone(),
                            truncated: tracked.last_truncated,
                            exit_code: tracked.exit_code,
                            signal: tracked.signal.clone(),
                        },
                    );
                    if snapshot.completed {
                        Resolved::Ready(snapshot)
                    } else {
                        Resolved::FromLog {
                            snapshot,
                            output_file: tracked.output_file.clone(),
                            limit: tracked.output_byte_limit,
                        }
                    }
                }
                (None, None) => Resolved::Missing,
            }
        };

        match resolved {
            Resolved::Ready(snapshot) => Some(snapshot),
            Resolved::Missing => None,
            Resolved::FromLog {
                mut snapshot,
                output_file,
                limit,
            } => {
                if let Some(tail) = read_log_tail(&output_file, limit).await {
                    snapshot.output = tail.text;
                    snapshot.truncated = tail.truncated;
                }
                Some(snapshot)
            }
        }
    }

    async fn kill_task(&self, task_id: &str) -> KillOutcome {
        {
            let mut tasks = self.tasks.lock().unwrap();
            if let Some(task) = tasks.get_mut(task_id) {
                task.explicitly_killed = true;
            }
        }

        match self
            .gateway
            .send(acp::KillTerminalRequest::new(
                self.session_id.clone(),
                self.terminal_id(task_id),
            ))
            .await
        {
            Ok(_) => KillOutcome::Killed,
            Err(_) => KillOutcome::NotFound,
        }
    }

    async fn wait_for_completion(
        &self,
        task_id: &str,
        timeout: Option<Duration>,
    ) -> Option<TaskSnapshot> {
        let timeout = timeout.unwrap_or(Duration::from_secs(30));
        let mut block_wait_guard = BlockWaitGuard::new(&self.tasks, task_id);

        let gateway_result = tokio::time::timeout(
            timeout,
            self.gateway.send(acp::WaitForTerminalExitRequest::new(
                self.session_id.clone(),
                self.terminal_id(task_id),
            )),
        )
        .await;

        match &gateway_result {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                tracing::warn!(task_id, error = %e, "gateway error waiting for terminal exit, falling back to polling");
                let deadline = tokio::time::Instant::now() + timeout;
                poll_for_terminal_exit(
                    &self.gateway,
                    &self.session_id,
                    &self.terminal_id(task_id),
                    Some(deadline),
                )
                .await;
            }
            Err(_) => {
                tracing::debug!(task_id, "timeout waiting for terminal exit");
            }
        }

        let snapshot = self.get_task(task_id).await;
        block_wait_guard.finish_with_snapshot(snapshot)
    }

    async fn list_tasks(&self) -> Vec<TaskSnapshot> {
        let task_ids: Vec<String> = {
            let tasks = self.tasks.lock().unwrap();
            tasks.keys().cloned().collect()
        };
        let mut snapshots = Vec::new();
        for task_id in task_ids {
            if let Some(snapshot) = self.get_task(&task_id).await {
                snapshots.push(snapshot);
            }
        }
        snapshots
    }

    async fn kill_all_background_tasks(&self) {
        let task_ids: Vec<String> = {
            let tasks = self.tasks.lock().unwrap();
            tasks
                .iter()
                .filter(|(_, t)| !t.completed)
                .map(|(id, _)| id.clone())
                .collect()
        };
        for task_id in task_ids {
            self.kill_task(&task_id).await;
        }
    }

    async fn kill_foreground_commands(&self) {
        let session_id = self.session_id.0.to_string();
        crate::terminal::kill_and_release_all_for_session(&session_id).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_grok_tools::notification::types::ToolNotificationHandle;

    fn make_tracked_task(command: &str) -> TrackedTask {
        TrackedTask {
            command: command.to_string(),
            cwd: "/tmp".to_string(),
            output_file: PathBuf::from("/tmp/out.log"),
            ..Default::default()
        }
    }

    fn out(output: &str, exit_code: Option<i32>, signal: Option<String>) -> SnapshotOutput {
        SnapshotOutput {
            output: output.into(),
            truncated: false,
            exit_code,
            signal,
        }
    }

    #[test]
    fn wrap_command_quotes_shell_metacharacters() {
        let cmd = wrap_command("echo 'hello world' && ls").unwrap();
        #[cfg(unix)]
        {
            let shell = crate::terminal::default_shell_path();
            assert!(
                cmd.starts_with(&format!("{shell} -lc")),
                "expected wrapped cmd to begin with `{shell} -lc`, got: {cmd}"
            );
        }
        #[cfg(not(unix))]
        assert_eq!(cmd, "echo 'hello world' && ls");
        assert!(cmd.contains("echo"));
    }

    #[test]
    fn parse_exit_maps_code_signal_and_none() {
        let code = Some(acp::TerminalExitStatus::new().exit_code(Some(42)));
        assert_eq!(parse_exit(&code), (Some(42), None));
        let signal = Some(acp::TerminalExitStatus::new().signal(Some("SIGKILL".into())));
        assert_eq!(parse_exit(&signal), (None, Some("SIGKILL".into())));
        assert_eq!(parse_exit(&None), (None, None));
    }

    #[test]
    fn to_snapshot_derives_completed_and_end_time() {
        let running = make_tracked_task("ls -la").to_snapshot("t-1", out("partial", None, None));
        assert!(!running.completed);
        assert!(running.end_time.is_none());

        // An exit code or a signal marks the snapshot complete and stamps end_time.
        let exited = make_tracked_task("fast").to_snapshot("t-2", out("", Some(1), None));
        assert!(exited.completed);
        assert!(exited.end_time.is_some());
        assert_eq!(exited.exit_code, Some(1));

        let signaled =
            make_tracked_task("killed").to_snapshot("t-3", out("", None, Some("SIGTERM".into())));
        assert!(signaled.completed);
        assert!(signaled.end_time.is_some());
    }

    /// Scripted client side of the terminal protocol: each `terminal/output`
    /// serves the next snapshot; `wait_for_exit` resolves after the last one.
    fn scripted_gateway(outputs: Vec<(String, bool)>) -> GatewaySender {
        use xai_acp_lib::AcpClientMessage;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            let mut next = 0usize;
            let mut wait_reply: Option<
                tokio::sync::oneshot::Sender<
                    xai_acp_lib::AcpResult<acp::WaitForTerminalExitResponse>,
                >,
            > = None;
            let mut exited = false;
            while let Some(msg) = rx.recv().await {
                match msg {
                    AcpClientMessage::CreateTerminal(args) => {
                        let _ = args
                            .response_tx
                            .send(Ok(acp::CreateTerminalResponse::new("term-1")));
                    }
                    AcpClientMessage::WaitForTerminalExit(args) => {
                        wait_reply = Some(args.response_tx);
                    }
                    AcpClientMessage::TerminalOutput(args) => {
                        let idx = next.min(outputs.len() - 1);
                        let (text, truncated) = outputs[idx].clone();
                        let mut response = acp::TerminalOutputResponse::new(text, truncated);
                        if exited {
                            response = response.exit_status(Some(
                                acp::TerminalExitStatus::new().exit_code(Some(0)),
                            ));
                        }
                        next += 1;
                        let _ = args.response_tx.send(Ok(response));
                        if next >= outputs.len()
                            && let Some(reply) = wait_reply.take()
                        {
                            exited = true;
                            let _ = reply.send(Ok(acp::WaitForTerminalExitResponse::new(
                                acp::TerminalExitStatus::new().exit_code(Some(0)),
                            )));
                        }
                    }
                    AcpClientMessage::ReleaseTerminal(args) => {
                        let _ = args
                            .response_tx
                            .send(Ok(acp::ReleaseTerminalResponse::new()));
                        break;
                    }
                    AcpClientMessage::KillTerminalCommand(args) => {
                        let _ = args.response_tx.send(Ok(acp::KillTerminalResponse::new()));
                    }
                    _ => {}
                }
            }
        });
        GatewaySender::new(tx)
    }

    fn background_request(output_file: PathBuf) -> TerminalRunRequest {
        TerminalRunRequest {
            command: "watch-something".into(),
            working_directory: PathBuf::from("/tmp"),
            env: HashMap::new(),
            timeout: Duration::from_secs(60),
            output_byte_limit: 1024 * 1024,
            output_file,
            notification_handle: ToolNotificationHandle::noop(),
            tool_call_id: "call-1".into(),
            display_command: Some("[monitor] watch".into()),
            auto_background_on_timeout: false,
            foreground_block_budget: None,
            kind: TaskKind::Monitor,
            owner_session_id: Some("owner-1".into()),
            description: None,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn run_background_records_snapshots_and_threads_task_kind() {
        use xai_grok_tools::notification::types::ToolNotification;

        let dir = tempfile::tempdir().unwrap();
        let output_file = dir.path().join("terminal").join("monitor-call-1.log");

        let gateway = scripted_gateway(vec![
            ("line1\n".into(), false),
            ("line1\nline2\n".into(), false),
            ("line1\nline2\nline3\n".into(), false),
        ]);
        let adapter = AcpTerminalAdapter::new(gateway, acp::SessionId::new("sess-1"));

        let (handle, mut notifications) = ToolNotificationHandle::channel();
        let mut request = background_request(output_file.clone());
        request.notification_handle = handle;

        let bg = adapter.run_background(request).await.unwrap();
        assert_eq!(bg.task_id, "term-1");
        assert!(output_file.exists());

        let snapshot = adapter.get_task(&bg.task_id).await.unwrap();
        assert_eq!(snapshot.kind, TaskKind::Monitor);
        assert_eq!(snapshot.owner_session_id.as_deref(), Some("owner-1"));

        let completed = loop {
            match notifications.recv().await.expect("completion notification") {
                ToolNotification::TaskCompleted(snapshot) => break snapshot,
                _ => continue,
            }
        };
        assert_eq!(completed.kind, TaskKind::Monitor);
        assert_eq!(completed.owner_session_id.as_deref(), Some("owner-1"));
        assert_eq!(completed.exit_code, Some(0));

        assert_eq!(
            std::fs::read_to_string(&output_file).unwrap(),
            "line1\nline2\nline3\n"
        );
    }

    /// A gateway whose `terminal/output` never replies, so live polls fail and
    /// `get_task` exercises its offline fallback.
    fn output_unavailable_gateway() -> GatewaySender {
        use xai_acp_lib::AcpClientMessage;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if let AcpClientMessage::ReleaseTerminal(args) = msg {
                    let _ = args
                        .response_tx
                        .send(Ok(acp::ReleaseTerminalResponse::new()));
                }
            }
        });
        GatewaySender::new(tx)
    }

    fn insert_task(adapter: &AcpTerminalAdapter, task_id: &str, task: TrackedTask) {
        adapter
            .tasks
            .lock()
            .unwrap()
            .insert(task_id.to_string(), task);
    }

    #[tokio::test]
    async fn get_task_completed_keeps_completion_buffer_over_log() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("done.log");
        tokio::fs::write(&log, "stale mirrored bytes")
            .await
            .unwrap();

        let adapter =
            AcpTerminalAdapter::new(output_unavailable_gateway(), acp::SessionId::new("s"));
        let mut task = TrackedTask {
            output_file: log,
            ..Default::default()
        };
        task.mark_completed(out("authoritative output", Some(0), None));
        insert_task(&adapter, "t-done", task);

        let snap = adapter.get_task("t-done").await.unwrap();
        assert_eq!(snap.output, "authoritative output");
        assert!(snap.completed);
    }

    #[tokio::test]
    async fn get_task_running_fills_output_from_log() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("run.log");
        tokio::fs::write(&log, "live streamed bytes").await.unwrap();

        let adapter =
            AcpTerminalAdapter::new(output_unavailable_gateway(), acp::SessionId::new("s"));
        insert_task(
            &adapter,
            "t-run",
            TrackedTask {
                output_file: log,
                output_byte_limit: 1024,
                ..Default::default()
            },
        );

        let snap = adapter.get_task("t-run").await.unwrap();
        assert_eq!(snap.output, "live streamed bytes");
        assert!(!snap.completed);
        assert!(!snap.truncated);
    }

    #[tokio::test]
    async fn cancelled_wait_after_completion_keeps_emitted_wake_eligible() {
        use xai_acp_lib::AcpClientMessage;

        let (gateway_tx, mut gateway_rx) = tokio::sync::mpsc::unbounded_channel();
        let adapter = AcpTerminalAdapter::new(
            GatewaySender::new(gateway_tx),
            acp::SessionId::new("session"),
        );
        insert_task(&adapter, "pending", make_tracked_task("sleep 60"));
        let tasks = Arc::clone(&adapter.tasks);

        let mut wait =
            Box::pin(adapter.wait_for_completion("pending", Some(Duration::from_secs(30))));
        let pending_reply = loop {
            tokio::select! {
                snapshot = &mut wait => {
                    panic!("wait unexpectedly completed: {snapshot:?}");
                }
                message = gateway_rx.recv() => {
                    match message.expect("wait request") {
                        AcpClientMessage::WaitForTerminalExit(args) => {
                            break args.response_tx;
                        }
                        _ => panic!("unexpected gateway request"),
                    }
                }
            }
        };

        let emitted_completion = {
            let mut tracked = tasks.lock().unwrap();
            assert_eq!(tracked["pending"].active_block_waiters, 1);
            assert!(!tracked["pending"].block_waited);
            let task = tracked.get_mut("pending").unwrap();
            task.mark_completed(out("done", Some(0), None));
            task.to_snapshot("pending", out("done", Some(0), None))
        };
        assert!(emitted_completion.completed);
        assert!(
            !emitted_completion.block_waited,
            "an active waiter must not speculatively suppress the completion wake"
        );

        drop(wait);
        drop(pending_reply);

        let snapshot = {
            let tracked = tasks.lock().unwrap();
            assert_eq!(tracked["pending"].active_block_waiters, 0);
            tracked["pending"].to_snapshot("pending", out("", None, None))
        };
        assert!(
            !snapshot.block_waited,
            "cancelling after completion must leave auto-wake eligible"
        );
    }

    #[tokio::test]
    async fn cancelling_one_of_two_waiters_keeps_only_live_wait_visible() {
        use xai_acp_lib::AcpClientMessage;

        let (gateway_tx, mut gateway_rx) = tokio::sync::mpsc::unbounded_channel();
        let adapter = AcpTerminalAdapter::new(
            GatewaySender::new(gateway_tx),
            acp::SessionId::new("session"),
        );
        insert_task(&adapter, "pending", make_tracked_task("sleep 60"));
        let tasks = Arc::clone(&adapter.tasks);

        let mut wait_one =
            Box::pin(adapter.wait_for_completion("pending", Some(Duration::from_secs(30))));
        let mut wait_two =
            Box::pin(adapter.wait_for_completion("pending", Some(Duration::from_secs(30))));
        let mut pending_replies = Vec::new();
        while pending_replies.len() < 2 {
            tokio::select! {
                snapshot = &mut wait_one => {
                    panic!("first wait unexpectedly completed: {snapshot:?}");
                }
                snapshot = &mut wait_two => {
                    panic!("second wait unexpectedly completed: {snapshot:?}");
                }
                message = gateway_rx.recv() => {
                    match message.expect("wait request") {
                        AcpClientMessage::WaitForTerminalExit(args) => {
                            pending_replies.push(args.response_tx);
                        }
                        _ => panic!("unexpected gateway request"),
                    }
                }
            }
        }

        {
            let tracked = tasks.lock().unwrap();
            assert_eq!(tracked["pending"].active_block_waiters, 2);
        }

        drop(wait_one);
        let one_waiter_snapshot = {
            let tracked = tasks.lock().unwrap();
            assert_eq!(tracked["pending"].active_block_waiters, 1);
            tracked["pending"].to_snapshot("pending", out("", None, None))
        };
        assert!(
            !one_waiter_snapshot.block_waited,
            "a surviving waiter must remain speculative until it delivers completion"
        );

        drop(wait_two);
        drop(pending_replies);
        let no_waiter_snapshot = {
            let tracked = tasks.lock().unwrap();
            assert_eq!(tracked["pending"].active_block_waiters, 0);
            tracked["pending"].to_snapshot("pending", out("", None, None))
        };
        assert!(
            !no_waiter_snapshot.block_waited,
            "dropping the last waiter must restore auto-wake eligibility"
        );
    }

    #[test]
    fn wait_finalization_upgrades_snapshot_completed_by_exit_watcher() {
        let tasks = Arc::new(Mutex::new(HashMap::from([(
            "racing".to_string(),
            make_tracked_task("fast command"),
        )])));
        let mut guard = BlockWaitGuard::new(&tasks, "racing");

        let stale_running_snapshot = {
            let tracked = tasks.lock().unwrap();
            tracked["racing"].to_snapshot("racing", out("partial", None, None))
        };
        assert!(!stale_running_snapshot.completed);

        let emitted_completion = {
            let mut tracked = tasks.lock().unwrap();
            let task = tracked.get_mut("racing").unwrap();
            task.mark_completed(out("complete", Some(0), None));
            task.to_snapshot("racing", out("complete", Some(0), None))
        };
        assert!(
            !emitted_completion.block_waited,
            "completion notification stays wake-eligible until the wait delivers"
        );

        let delivered = guard
            .finish_with_snapshot(Some(stale_running_snapshot))
            .expect("tracked completion should replace stale snapshot");
        assert!(delivered.completed);
        assert_eq!(delivered.output, "complete");
        assert_eq!(delivered.exit_code, Some(0));
        assert!(delivered.block_waited);

        let tracked = tasks.lock().unwrap();
        assert_eq!(tracked["racing"].active_block_waiters, 0);
        assert!(tracked["racing"].block_waited);
        assert!(
            tracked["racing"]
                .to_snapshot("racing", out("complete", Some(0), None))
                .block_waited,
            "snapshots after successful delivery must suppress redundant wakes"
        );
    }
}
