# External Weight Sources

Real model weights are stored outside GitHub.

## Supported sources

- Google Drive (shared download link resolved to a direct HTTPS file URL)
- Hugging Face model/file URLs (public or authenticated via pre-signed URL/proxy)
- Any HTTPS object-store URL (S3/GCS/Azure presigned)

## Example

```bash
node tools/fetch-weights.js "https://huggingface.co/<org>/<model>/resolve/main/model.bin" weights/model.bin
```

```bash
node tools/fetch-weights.js "https://drive.google.com/uc?export=download&id=<FILE_ID>" weights/model.bin
```

## Notes

- Do **not** commit downloaded model binaries to git.
- Keep downloaded artifacts local or in private storage.
- For in-repo fixtures use `weights/model-1b-int4.base64.json` only.
