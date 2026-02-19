

## CSS-LLM deterministic metadata bundle

Additional CSS-LLM format control files are included in this folder:

- `codex.json`: package manifest tying core model files to deterministic replay rules.
- `meta.json`: model identity/capabilities/governance references.
- `policy.json`: canonicalization, evaluation, and integrity policy.
- `tools.json`: deterministic tool contracts for math/geometry execution.
- `semantics.json`: operator/algebra/calculus/matrix semantics map.
- `geometry.json`: core geometry constants and formulas.
- `math-schema.json`: machine-validated canonical JSON schema for math AST payloads.

These files are metadata/config surfaces; they keep the adapter file lightweight while making deterministic behavior and replay requirements explicit.


## Phase checklist

- [x] Deterministic metadata manifests are present.
- [x] Governance and replay policy surfaces are documented.
- [ ] Add examples for consuming metadata in client bootstrap.
