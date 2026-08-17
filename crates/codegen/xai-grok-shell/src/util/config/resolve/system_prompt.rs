pub const ENV_SYSTEM_PROMPT_LABEL: &str = "GROK_SYSTEM_PROMPT_LABEL";

pub const DEFAULT_SYSTEM_PROMPT_LABEL: &str = xai_grok_agent::DEFAULT_SYSTEM_PROMPT_LABEL;
pub const CODEX_SYSTEM_PROMPT_LABEL: &str = xai_grok_agent::CODEX_SYSTEM_PROMPT_LABEL;

/// Resolve the system-prompt identity for a model.
/// Label precedence: env → config per-model → `[agent]` → GB per-model (or
/// Codex provider default) → GB global → `"Grok"`. Empty values fall through.
/// Vendor attribution follows the provider independently of the chosen label.
///
/// Per-model TOML is looked up by session catalog id, then routing slug
/// (`ModelInfo.model`). Do not use CLI `-m` alone — it may outlive a mid-session
/// model switch.
pub(crate) fn resolve_system_prompt_identity(
    cfg: &crate::agent::config::Config,
    model_id: &str,
    model: Option<&crate::agent::config::ModelInfo>,
) -> xai_grok_agent::SystemPromptIdentity {
    let label_for = |key: &str| {
        cfg.config_models
            .get(key)
            .and_then(|m| m.system_prompt_label.clone())
    };
    let user_per_model =
        label_for(model_id).or_else(|| model.map(|m| m.model.as_str()).and_then(label_for));

    let label = resolve_system_prompt_label_from_tiers(
        user_per_model,
        cfg.agent.system_prompt_label.clone(),
        catalog_system_prompt_label(model),
        cfg.remote_settings
            .as_ref()
            .and_then(|r| r.system_prompt_label.clone()),
    );
    system_prompt_identity(label, model)
}

fn system_prompt_identity(
    label: String,
    model: Option<&crate::agent::config::ModelInfo>,
) -> xai_grok_agent::SystemPromptIdentity {
    let vendor = if model.is_some_and(model_uses_codex_identity) {
        String::new()
    } else {
        xai_grok_agent::DEFAULT_SYSTEM_PROMPT_VENDOR.to_string()
    };
    xai_grok_agent::SystemPromptIdentity { label, vendor }
}

/// GB per-model label, or the Codex provider default when the catalog left it
/// unset. Empty/whitespace is treated as unset.
pub(crate) fn catalog_system_prompt_label(
    model: Option<&crate::agent::config::ModelInfo>,
) -> Option<String> {
    let model = model?;
    if let Some(label) = model
        .system_prompt_label
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return Some(label.to_string());
    }
    model_uses_codex_identity(model).then(|| CODEX_SYSTEM_PROMPT_LABEL.to_string())
}

fn model_uses_codex_identity(model: &crate::agent::config::ModelInfo) -> bool {
    model.provider_id == Some(xai_grok_sampling_types::ProviderId::Codex)
        || xai_grok_sampling_types::is_codex_backend_url(&model.base_url)
}

pub(crate) fn resolve_system_prompt_label_from_tiers(
    user_per_model: Option<String>,
    user_global: Option<String>,
    gb_per_model: Option<String>,
    gb_global: Option<String>,
) -> String {
    let non_empty = |s: Option<String>| {
        s.and_then(|v| {
            let t = v.trim();
            (!t.is_empty()).then(|| t.to_string())
        })
    };
    std::env::var(ENV_SYSTEM_PROMPT_LABEL)
        .ok()
        .and_then(|s| non_empty(Some(s)))
        .or_else(|| non_empty(user_per_model))
        .or_else(|| non_empty(user_global))
        .or_else(|| non_empty(gb_per_model))
        .or_else(|| non_empty(gb_global))
        .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT_LABEL.to_string())
}

#[cfg(test)]
mod system_prompt_label_tests {
    use super::{
        CODEX_SYSTEM_PROMPT_LABEL, DEFAULT_SYSTEM_PROMPT_LABEL, ENV_SYSTEM_PROMPT_LABEL,
        resolve_system_prompt_label_from_tiers,
    };

    /// Serialize access to `GROK_SYSTEM_PROMPT_LABEL` and clear it for tier tests.
    /// `env_wins_over_all_tiers` mutates the env; without this lock, parallel tests
    /// that expect the var unset (e.g. `gb_per_model_beats_gb_global`) flake.
    fn with_env_cleared<R>(f: impl FnOnce() -> R) -> R {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev = std::env::var(ENV_SYSTEM_PROMPT_LABEL).ok();
        // Safety: test-only, locked.
        unsafe { std::env::remove_var(ENV_SYSTEM_PROMPT_LABEL) };
        let r = f();
        match prev {
            Some(v) => unsafe { std::env::set_var(ENV_SYSTEM_PROMPT_LABEL, v) },
            None => unsafe { std::env::remove_var(ENV_SYSTEM_PROMPT_LABEL) },
        }
        r
    }

    #[test]
    fn default_when_all_unset() {
        with_env_cleared(|| {
            assert_eq!(
                resolve_system_prompt_label_from_tiers(None, None, None, None),
                DEFAULT_SYSTEM_PROMPT_LABEL
            );
        });
    }

    #[test]
    fn per_model_beats_global_and_gb() {
        with_env_cleared(|| {
            assert_eq!(
                resolve_system_prompt_label_from_tiers(
                    Some("PerModel".into()),
                    Some("Global".into()),
                    Some("GbPer".into()),
                    Some("GbGlobal".into()),
                ),
                "PerModel"
            );
        });
    }

    #[test]
    fn global_beats_gb() {
        with_env_cleared(|| {
            assert_eq!(
                resolve_system_prompt_label_from_tiers(
                    None,
                    Some("Global".into()),
                    Some("GbPer".into()),
                    Some("GbGlobal".into()),
                ),
                "Global"
            );
        });
    }

    #[test]
    fn gb_per_model_beats_gb_global() {
        with_env_cleared(|| {
            assert_eq!(
                resolve_system_prompt_label_from_tiers(
                    None,
                    None,
                    Some("GbPer".into()),
                    Some("GbGlobal".into()),
                ),
                "GbPer"
            );
        });
    }

    #[test]
    fn empty_and_whitespace_fall_through() {
        with_env_cleared(|| {
            assert_eq!(
                resolve_system_prompt_label_from_tiers(
                    Some("  ".into()),
                    Some("".into()),
                    None,
                    Some("GbGlobal".into()),
                ),
                "GbGlobal"
            );
        });
    }

    #[test]
    fn env_wins_over_all_tiers() {
        let _guard = ENV_LOCK.lock().unwrap();
        // Safety: test-only, locked.
        unsafe { std::env::set_var(ENV_SYSTEM_PROMPT_LABEL, "FromEnv") };
        let got = resolve_system_prompt_label_from_tiers(
            Some("PerModel".into()),
            Some("Global".into()),
            Some("GbPer".into()),
            Some("GbGlobal".into()),
        );
        unsafe { std::env::remove_var(ENV_SYSTEM_PROMPT_LABEL) };
        assert_eq!(got, "FromEnv");
    }

    #[test]
    fn catalog_codex_provider_defaults_to_codex_not_global_grok() {
        let mut model = crate::agent::config::ModelInfo::fallback("gpt-5.6-sol");
        model.provider_id = Some(xai_grok_sampling_types::ProviderId::Codex);
        assert_eq!(
            super::catalog_system_prompt_label(Some(&model)).as_deref(),
            Some(CODEX_SYSTEM_PROMPT_LABEL)
        );
        assert_eq!(
            resolve_system_prompt_label_from_tiers(
                None,
                None,
                super::catalog_system_prompt_label(Some(&model)),
                Some("Grok 4.6".into()),
            ),
            CODEX_SYSTEM_PROMPT_LABEL
        );
    }

    #[test]
    fn catalog_codex_url_without_provider_id_defaults_to_codex() {
        let mut model = crate::agent::config::ModelInfo::fallback("gpt-5.6-sol");
        model.base_url = xai_grok_sampling_types::CODEX_BACKEND_BASE_URL.to_string();
        assert_eq!(
            super::catalog_system_prompt_label(Some(&model)).as_deref(),
            Some(CODEX_SYSTEM_PROMPT_LABEL)
        );
    }

    #[test]
    fn catalog_explicit_label_wins_over_codex_provider_default() {
        let mut model = crate::agent::config::ModelInfo::fallback("gpt-5.6-sol");
        model.provider_id = Some(xai_grok_sampling_types::ProviderId::Codex);
        model.system_prompt_label = Some("Custom Codex".into());
        assert_eq!(
            super::catalog_system_prompt_label(Some(&model)).as_deref(),
            Some("Custom Codex")
        );
        assert_eq!(
            super::system_prompt_identity("Custom Codex".into(), Some(&model)),
            xai_grok_agent::SystemPromptIdentity {
                label: "Custom Codex".into(),
                vendor: String::new(),
            }
        );
    }

    #[test]
    fn catalog_non_codex_without_label_does_not_invent_identity() {
        let model = crate::agent::config::ModelInfo::fallback("grok-4.6");
        assert_eq!(super::catalog_system_prompt_label(Some(&model)), None);
    }

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
}
