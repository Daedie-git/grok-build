//! Presentation helpers for ChatGPT/Codex account rate limits.

use xai_grok_shell::extensions::codex_usage::{CodexRateLimitWindow, CodexRateLimits};

fn window_label(window: &CodexRateLimitWindow, fallback: &'static str) -> &'static str {
    match window.window_duration_mins {
        Some(300) => "5-hour limit",
        Some(10_080) => "Weekly limit",
        _ => fallback,
    }
}

fn format_reset(unix_seconds: i64) -> Option<String> {
    chrono::DateTime::from_timestamp(unix_seconds, 0).map(|timestamp| {
        timestamp
            .with_timezone(&chrono::Local)
            .format("%B %-d, %H:%M")
            .to_string()
    })
}

fn append_window(
    lines: &mut Vec<String>,
    window: &CodexRateLimitWindow,
    fallback_label: &'static str,
) {
    let label = window_label(window, fallback_label);
    lines.push(format!(
        "{label}: {}% used",
        window.used_percent.floor() as i64
    ));
    if let Some(reset) = window.resets_at.and_then(format_reset) {
        lines.push(format!("  Resets: {reset}"));
    }
}

/// `/usage` account section for a Codex-backed model.
pub fn format_usage_summary(limits: &CodexRateLimits) -> String {
    let mut lines = vec!["Codex account usage:".to_string()];
    if let Some(primary) = &limits.primary {
        append_window(&mut lines, primary, "Primary limit");
    }
    if let Some(secondary) = &limits.secondary {
        append_window(&mut lines, secondary, "Secondary limit");
    }
    if limits.primary.is_none() && limits.secondary.is_none() {
        lines.push("No rate-limit windows were reported.".to_string());
    }
    if let Some(credits) = &limits.credits
        && credits.has_credits
    {
        let balance = if credits.unlimited {
            "unlimited".to_string()
        } else {
            credits
                .balance
                .as_deref()
                .map(|balance| {
                    if balance.starts_with('$') {
                        balance.to_string()
                    } else {
                        format!("${balance}")
                    }
                })
                .unwrap_or_else(|| "available".to_string())
        };
        lines.push(format!("Credits: {balance}"));
    }
    lines.join("\n")
}

/// High-usage prompt warning. The more-consumed window wins.
pub fn usage_warning(limits: &CodexRateLimits) -> Option<(String, bool)> {
    let windows = [
        limits
            .primary
            .as_ref()
            .map(|window| (window, "Primary limit")),
        limits
            .secondary
            .as_ref()
            .map(|window| (window, "Secondary limit")),
    ];
    let (window, fallback) = windows
        .into_iter()
        .flatten()
        .filter(|(window, _)| window.used_percent > 90.0)
        .max_by(|(a, _), (b, _)| a.used_percent.total_cmp(&b.used_percent))?;
    let remaining = (100 - window.used_percent.floor() as i64).max(0);
    Some((
        format!(
            "Codex {} left: {remaining}%",
            window_label(window, fallback).to_ascii_lowercase()
        ),
        window.used_percent > 95.0,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(used_percent: f64, minutes: i64, resets_at: Option<i64>) -> CodexRateLimitWindow {
        CodexRateLimitWindow {
            used_percent,
            window_duration_mins: Some(minutes),
            resets_at,
        }
    }

    #[test]
    fn summary_labels_five_hour_and_weekly_windows() {
        let limits = CodexRateLimits {
            plan_type: Some("plus".into()),
            primary: Some(window(12.0, 300, None)),
            secondary: Some(window(34.0, 10_080, None)),
            credits: None,
        };
        assert_eq!(
            format_usage_summary(&limits),
            "Codex account usage:\n5-hour limit: 12% used\nWeekly limit: 34% used"
        );
    }

    #[test]
    fn warning_uses_most_consumed_window() {
        let limits = CodexRateLimits {
            plan_type: None,
            primary: Some(window(96.0, 300, None)),
            secondary: Some(window(92.0, 10_080, None)),
            credits: None,
        };
        assert_eq!(
            usage_warning(&limits),
            Some(("Codex 5-hour limit left: 4%".to_string(), true))
        );
    }
}
