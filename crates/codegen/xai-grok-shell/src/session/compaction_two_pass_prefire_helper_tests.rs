use super::{
    CompactionStrategyOverride, ConfiguredCodexCompactionStrategy, fingerprint_prefix,
    prefire_lead_percent,
};
use xai_grok_sampling_types::ConversationItem;

#[test]
fn strategy_environment_snapshot_distinguishes_absent_invalid_and_text_values() {
    let unset =
        ConfiguredCodexCompactionStrategy::from_env_result(Err(std::env::VarError::NotPresent));
    assert_eq!(unset, ConfiguredCodexCompactionStrategy::Unset);
    assert_eq!(unset.as_override(), CompactionStrategyOverride::Unset);

    for value in ["", "local", " native_codex "] {
        let configured = ConfiguredCodexCompactionStrategy::from_env_result(Ok(value.to_owned()));
        assert_eq!(
            configured.as_override(),
            CompactionStrategyOverride::Value(value)
        );
    }

    let invalid =
        ConfiguredCodexCompactionStrategy::from_env_result(Err(std::env::VarError::NotUnicode(
            std::ffi::OsString::from("synthetic non-Unicode environment value"),
        )));
    assert_eq!(invalid, ConfiguredCodexCompactionStrategy::InvalidEncoding);
    assert_eq!(
        invalid.as_override(),
        CompactionStrategyOverride::InvalidEncoding
    );
}

#[test]
fn fingerprint_stable_for_same_prefix() {
    let items = vec![
        ConversationItem::system("sys"),
        ConversationItem::user("hello"),
        ConversationItem::assistant("hi"),
    ];
    assert_eq!(fingerprint_prefix(&items), fingerprint_prefix(&items));
}

#[test]
fn fingerprint_changes_when_prefix_content_changes() {
    let base = vec![
        ConversationItem::system("sys"),
        ConversationItem::user("hello"),
    ];
    let edited = vec![
        ConversationItem::system("sys"),
        ConversationItem::user("HELLO there"), // a real edit/rewind of the prefix
    ];
    assert_ne!(
        fingerprint_prefix(&base),
        fingerprint_prefix(&edited),
        "a changed prefix must invalidate the cached NOTE1 fingerprint"
    );
}

#[test]
fn fingerprint_changes_with_length() {
    let short = vec![ConversationItem::user("a")];
    let long = vec![
        ConversationItem::user("a"),
        ConversationItem::assistant("b"),
    ];
    assert_ne!(fingerprint_prefix(&short), fingerprint_prefix(&long));
}

#[test]
fn prefire_lead_percent_defaults_to_10() {
    // SAFETY: single-threaded test mutation of our own env var.
    unsafe { std::env::remove_var("GROK_PREFIRE_LEAD_PERCENT") };
    assert_eq!(prefire_lead_percent(), 10);
}
