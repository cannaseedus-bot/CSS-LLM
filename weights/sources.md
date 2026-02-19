# External Weight Sources

Real model weights are stored outside GitHub.

## Supported sources

- Google Drive (shared download link resolved to direct HTTPS URLs)
- Hugging Face model/file URLs (public or authenticated via pre-signed URL/proxy)
- Any HTTPS object-store URL (S3/GCS/Azure presigned)

## Sharded format

Large models should be split into base64 shard artifacts, e.g.:

- `css_llm_001.bin.base64.json`
- `css_llm_002.bin.base64.json`
- `css_llm_003.bin.base64.json`
- `css_llm_index.json`

Each shard stores only a chunk of binary payload as Base64 text.

## Generate local shards

```bash
node tools/pack-weights.js weights/shards css_llm 3
```

## Fetch remote shards

```bash
node tools/fetch-weights.js "https://host/path/css_llm_index.json" weights/shards
```

## Assemble to binary

```bash
node tools/assemble-shards.js weights/shards/css_llm_index.json weights/model.bin
```

## Notes

- Do **not** commit downloaded/assembled model binaries to git.
- Keep downloaded artifacts local or in private storage.
- In-repo files are development fixtures only.
