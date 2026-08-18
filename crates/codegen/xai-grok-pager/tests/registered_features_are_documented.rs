//! The registry is the source of truth and the operator tables are
//! hand-maintained mirrors with no compile-time tripwire of their own. This test
//! is theirs.
//!
//! Public checkouts strip `docs/internal/`; `include_str!` would fail the
//! whole pager test crate there. Load at runtime and skip when the files
//! are absent so the tripwire still fires in the monorepo.

use std::path::PathBuf;

use xai_grok_shell::agent::config::FEATURES;

fn internal_doc(name: &str) -> std::io::Result<Option<String>> {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "docs", "internal", name]
        .iter()
        .collect();
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err),
    }
}

#[test]
fn every_registered_feature_reaches_the_operator() {
    let Some(enterprise) = internal_doc("25-enterprise.md").expect("read 25-enterprise.md")
    else {
        eprintln!("skipping: docs/internal not shipped in this checkout");
        return;
    };
    let Some(env_vars) =
        internal_doc("22-environment-variables.md").expect("read 22-environment-variables.md")
    else {
        eprintln!("skipping: docs/internal not shipped in this checkout");
        return;
    };

    for spec in FEATURES {
        assert!(
            enterprise.contains(&format!("`{}`", spec.key)),
            "{} has no row in the 25-enterprise.md pinning table",
            spec.key,
        );
        assert!(
            env_vars.contains(&format!("`{}`", spec.env)),
            "{} is undocumented in 22-environment-variables.md",
            spec.env,
        );
    }
}
