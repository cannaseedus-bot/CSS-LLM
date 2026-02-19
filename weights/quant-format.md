# CLIF-1 (CSS-LLM INT4 Packed Format)

## Header (64 bytes)

- magic: 4 bytes
- version: 4 bytes
- hiddenDim: 4 bytes
- numLayers: 4 bytes
- vocabSize: 4 bytes
- quantScheme: 4 bytes
- checksum: 32 bytes
- reserved: 8 bytes

## Payload

- 4-bit weights (2 values per byte)
- Block quantization scale table (float16)
- Block size: 128
