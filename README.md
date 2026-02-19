# CSS-LLM

Formalized CSS governance grammar plus a minimal viable 1B browser tensor shell.

## What this repo now contains

- **Formal governance grammar:** `specs/governance-grammar-v1.css`
- **Minimal 1B profile:** `models/model-1b-governance.css`
- **WebGPU shell implementation:** `src/tensor-shell-minimal.js`
- **Runnable usage example:** `examples/index.html`
- **Architecture notes:** `docs/architecture.md`

## Quick start

Serve the repository root with any static file server and open `examples/index.html` in a browser with WebGPU support.

> Example:
>
> ```bash
> python3 -m http.server 8080
> ```

Then visit: `http://localhost:8080/examples/index.html`
