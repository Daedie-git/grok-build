//! Shared marker → ack / lost-ack verify → outcome mapping for history rewrites.
//!
//! Rewind and compaction (local + native) all use the same durable commit
//! protocol at the caller layer. Persistence still owns marker-first writes
//! ([`crate::session::persistence`]); this module owns the oneshot await and
//! lost-ack resolution so those paths cannot drift.

use std::io;

use crate::session::persistence::{TimelineCacheStatus, TimelineTransactionOutcome};

/// Durable resolution of a lost timeline-transaction acknowledgement.
#[derive(Debug)]
pub enum TimelineCommitVerification {
    /// The exact marker is durable and no later timeline transaction superseded it.
    Committed,
    /// The exact transaction marker is absent.
    NotCommitted,
    /// Durable state could not establish one unambiguous outcome.
    Indeterminate(io::Error),
}

/// Failure from [`await_timeline_transaction`] after a lost or negative ack.
#[derive(Debug)]
pub enum TimelineTransactionResolveError {
    /// Marker never landed; live history must stay untouched.
    NotCommitted(String),
    /// Durable state is ambiguous; caller must set `reconciliation_required`.
    Indeterminate(String),
}

impl TimelineTransactionResolveError {
    pub fn message(&self) -> &str {
        match self {
            Self::NotCommitted(message) | Self::Indeterminate(message) => message,
        }
    }

    pub fn requires_reconciliation(&self) -> bool {
        matches!(self, Self::Indeterminate(_))
    }
}

/// Await a persistence oneshot. If the ack is lost, run `verify` on a blocking
/// pool and map Committed / NotCommitted / Indeterminate the same way for every
/// history-rewriting op.
pub async fn await_timeline_transaction(
    response: tokio::sync::oneshot::Receiver<TimelineTransactionOutcome>,
    op_name: &'static str,
    verify: impl FnOnce() -> TimelineCommitVerification + Send + 'static,
) -> Result<TimelineTransactionOutcome, TimelineTransactionResolveError> {
    match response.await {
        Ok(outcome) => Ok(outcome),
        Err(_) => {
            let verification = tokio::task::spawn_blocking(verify).await.map_err(|error| {
                TimelineTransactionResolveError::Indeterminate(format!(
                    "{op_name} acknowledgement was lost and durable verification failed; \
                         reload required: {error}"
                ))
            })?;
            match verification {
                TimelineCommitVerification::Committed => {
                    tracing::warn!(
                        "{op_name} acknowledgement lost; exact durable transaction verified committed"
                    );
                    Ok(TimelineTransactionOutcome::Committed {
                        marker_bookkeeping_error: None,
                        cache_status: TimelineCacheStatus::RepairRequired(io::Error::other(
                            format!("{op_name} cache status unknown after lost acknowledgement"),
                        )),
                    })
                }
                TimelineCommitVerification::NotCommitted => {
                    Err(TimelineTransactionResolveError::NotCommitted(format!(
                        "{op_name} acknowledgement was lost before the exact transaction \
                         committed; original history remains live"
                    )))
                }
                TimelineCommitVerification::Indeterminate(error) => {
                    Err(TimelineTransactionResolveError::Indeterminate(format!(
                        "{op_name} acknowledgement was lost and durable state is ambiguous; \
                         reload required before further sampling: {error}"
                    )))
                }
            }
        }
    }
}

/// Map a committed / not-committed persistence outcome and emit shared warnings.
/// Returns `Err(message)` when the marker was not committed.
///
/// [`TimelineCacheStatus::RepairRequired`] is a committed marker whose derived
/// cache did not land. Callers must not resume ordinary chat-cache appends:
/// gate the session (set `reconciliation_required`) and force a reload so
/// recovery can rebuild the authoritative cache.
pub fn ensure_timeline_committed(
    outcome: TimelineTransactionOutcome,
    op_name: &'static str,
    session_id: &str,
) -> Result<TimelineCacheStatus, String> {
    match outcome {
        TimelineTransactionOutcome::NotCommitted(error) => Err(format!(
            "{op_name} was not committed; original history remains live: {error}"
        )),
        TimelineTransactionOutcome::Committed {
            marker_bookkeeping_error,
            cache_status,
        } => {
            if let Some(error) = marker_bookkeeping_error {
                tracing::warn!(
                    session_id = %session_id,
                    %error,
                    "{op_name} committed; marker summary bookkeeping needs repair"
                );
            }
            match &cache_status {
                TimelineCacheStatus::Current => {}
                TimelineCacheStatus::CurrentWithBookkeepingError(error) => {
                    tracing::warn!(
                        session_id = %session_id,
                        %error,
                        "{op_name} committed and chat cache replaced; summary bookkeeping needs repair"
                    );
                }
                TimelineCacheStatus::RepairRequired(error) => {
                    tracing::warn!(
                        session_id = %session_id,
                        %error,
                        "{op_name} committed; chat cache was not replaced — reload required before further sampling"
                    );
                }
            }
            Ok(cache_status)
        }
    }
}
