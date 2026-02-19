# CSS-LLM

Formalized CSS governance contracts for both browser execution and enterprise server-required deployment.

## What this repo contains

### Browser profile

- **Formal governance grammar:** `specs/governance-grammar-v1.css`
- **Minimal 1B profile:** `models/model-1b-governance.css`
- **WebGPU shell implementation:** `src/tensor-shell-minimal.js`
- **Runnable usage example:** `examples/index.html`

### Enterprise profile (server-required)

- **Enterprise contract format:** `specs/enterprise-model-format-v1.css`
- **Enterprise model profile:** `models/model-enterprise-server-governance.css`
- **Server transport client:** `src/enterprise-model-client.js`

## Quick start (browser demo)

Serve the repository root with any static file server and open `examples/index.html` in a browser with WebGPU support.

```bash
python3 -m http.server 8080
```

Then visit: `http://localhost:8080/examples/index.html`
