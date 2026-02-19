// ============================================================
// FUSED INT4 SWIGLU (W1/W2 fused projection scaffold)
// ============================================================

struct SwiGLUConfig {
  hiddenDim: u32,
  intermediateDim: u32,
  blockSize: u32,
}

@group(0) @binding(0) var<storage, read> X : array<f16>;
@group(0) @binding(1) var<storage, read> W1Packed : array<u32>;
@group(0) @binding(2) var<storage, read> W2Packed : array<u32>;
@group(0) @binding(3) var<storage, read> Scale : array<f16>;
@group(0) @binding(4) var<storage, read_write> Out : array<f16>;
@group(0) @binding(5) var<uniform> cfg : SwiGLUConfig;

fn decode_int4(v: u32, shift: u32) -> f32 {
  let raw = (v >> shift) & 0xFu;
  if (raw >= 8u) { return f32(i32(raw) - 16); }
  return f32(raw);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let token = gid.x;
  let base = token * cfg.hiddenDim;

  for (var o = 0u; o < cfg.intermediateDim; o++) {
    var gate: f32 = 0.0;
    var up: f32 = 0.0;

    for (var i = 0u; i < cfg.hiddenDim; i++) {
      let x = f32(X[base + i]);
      let linear = o * cfg.hiddenDim + i;
      let idx = linear / 8u;
      let shift = (linear % 8u) * 4u;
      let s = f32(Scale[i / cfg.blockSize]);

      gate += x * decode_int4(W1Packed[idx], shift) * s;
      up += x * decode_int4(W2Packed[idx], shift) * s;
    }

    let silu = gate / (1.0 + exp(-gate));
    Out[token * cfg.intermediateDim + o] = f16(silu * up);
  }
}
