//! Session-private native compaction adapters.

mod codex;

pub(super) use codex::{
    CodexCompactionInput, CompactionStrategy, CompactionStrategyOverride, NativeCompactionCounters,
    NativeCompactionFallbackReason, NativeCompactionOutcome, run_native_compaction,
    select_compaction_strategy,
};
