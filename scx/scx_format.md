# SCX Mini Binary Format

## Header

- magic: `SCXM` (4 bytes)
- version: `u32`
- tensor_count: `u32`

## Tensor record (repeated)

- name_length: `u32`
- name bytes (`utf-8`)
- packed_int4_length: `u32`
- packed_int4 payload (`u8[]`)
- scale_count: `u32`
- scales payload (`f16[]` serialized as bytes)

## Footer

- sha256 digest of file bytes before footer (32 bytes)

## Determinism guarantees

- Stable tensor iteration order (sorted names)
- Fixed little-endian encoding
- Symmetric INT4 quantization (`[-8, 7]`)
- Header-level version pin for replay compatibility


## GitHub-safe transport wrapper

- Store committed artifacts as `*.base64.json` wrappers.
- Decode to binary `.scx` locally/runtime before GPU upload.
