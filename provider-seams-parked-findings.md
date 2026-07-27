# Provider-Seams Parked Findings

These findings were identified during the Provider-Seams V2 audit but are intentionally **not part of the current refactor**. They are either pre-existing, optional hardening, documentation/test improvements, or would require a broader design change than the immediate provider-identity and Codex retry fixes.

## Sampling transition hardening

### `ResolvedSamplingTarget` cannot independently recompute environment-backed account identity

The target constructor verifies provider/API/route/model consistency, but accepts the runtime-resolved ChatGPT account identity supplied by the caller. Current transition callers use the sampler's effective-header resolver, but the type system does not make bypass impossible.

Why parked: there is no known failing caller in this change. Enforcing this structurally would require moving target construction behind a sampler-owned interface or introducing an unforgeable resolved-identity value across crates.

## Codex response-history edge cases

### End-to-end no-retry regression coverage for multiple messages

The sampling-types tests prove that multiple exact Codex message items survive conversion, persistence, ordinary replay, and compact replay. There is not yet an actor/stream-level test asserting that the original retry path emits no retry event.

Why parked: this is additional integration coverage, not a remaining production failure in the fixed conversion.

### Multi-part `output_text` preservation

One Codex message containing multiple `output_text` parts is currently joined with newline separators. This predates the multi-message fix and does not preserve original content-part boundaries, annotations, or logprobs.

Why parked: fixing it requires a durable representation decision or a new fail-closed restriction and is separate from supporting multiple message output items.

### Duplicate non-message owner identities

Exact capture rejects duplicate message IDs. Duplicate reasoning IDs or function-call IDs may instead fail later during manifest-owner validation.

Why parked: current replay validation fails closed; moving the error earlier is hardening rather than a demonstrated retry regression.

### Additional ordering variants

Potential tests include adjacent messages, three messages, calls before the first message, calls after the last message, and message/output/message combinations.

Why parked: the complete output manifest already validates and reorders all owners, and current tests cover interleaved reasoning, calls, two messages, cold persistence, ordinary replay, and compact replay.

## Documentation and cleanup

### Stale single-assistant wording

A conversation response accessor comment still implies conversion always appends exactly one assistant, while exact Codex responses may now retain multiple assistants and return the last one through the convenience accessor.

Why parked: documentation-only and not behaviorally significant.

### Capability-interface simplification

Some semantic Boolean methods on `ProviderCapabilities` are derivable from typed policies and could be removed to make the interface smaller.

Why parked: fields are already private and callers do not infer identity from capabilities. Further cleanup is not required for correctness.
