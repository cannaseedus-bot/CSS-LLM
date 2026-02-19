// ============================================================
// FLASH ATTENTION INT4 (streaming, deterministic traversal)
// ------------------------------------------------------------
// Notes:
// - This kernel targets headDim <= 128 in this implementation.
// - Q is fp16, K/V are int4 packed with per-block scale factors.
// - Sequence traversal is strictly left-to-right for determinism.
// ============================================================

struct AttnConfig {
  headDim: u32,
  seqLen: u32,
  blockSize: u32,
}

@group(0) @binding(0)
var<storage, read> Q : array<f16>;

@group(0) @binding(1)
var<storage, read> KPacked : array<u32>;

@group(0) @binding(2)
var<storage, read> VPacked : array<u32>;

@group(0) @binding(3)
var<storage, read> Scale : array<f16>;

@group(0) @binding(4)
var<storage, read_write> Out : array<f16>;

@group(0) @binding(5)
var<uniform> cfg : AttnConfig;

fn decode_int4(v: u32, shift: u32) -> f32 {
  let raw = (v >> shift) & 0xFu;
  if (raw >= 8u) {
    return f32(i32(raw) - 16);
  }
  return f32(raw);
}

@compute @workgroup_size(128)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let token = gid.x;
  if (token >= cfg.seqLen) { return; }

  // keep lid referenced to avoid unused-variable validation noise
  if (lid.x == 0u && cfg.headDim > 128u) {
    // unsupported configuration in this minimal kernel variant
    return;
  }

  var maxScore : f32 = -1e30;
  var sumExp : f32 = 0.0;

  var acc : array<f32, 128>;
  for (var d = 0u; d < 128u; d++) {
    acc[d] = 0.0;
  }

  // Causal streaming: j in [0, token)
  for (var j = 0u; j < token; j++) {
    var score : f32 = 0.0;

    for (var d = 0u; d < cfg.headDim; d++) {
      let q = f32(Q[token * cfg.headDim + d]);
      let packedIndex = (j * cfg.headDim + d) / 8u;
      let shift = (d % 8u) * 4u;
      let scaleIndex = d / cfg.blockSize;
      let scale = f32(Scale[scaleIndex]);
      let k = decode_int4(KPacked[packedIndex], shift) * scale;
      score += q * k;
    }

    score = score / sqrt(f32(cfg.headDim));

    // Online softmax recurrence (deterministic)
    let prevMax = maxScore;
    let newMax = max(prevMax, score);
    let prevScale = exp(prevMax - newMax);
    let curScale = exp(score - newMax);

    for (var d = 0u; d < cfg.headDim; d++) {
      let packedIndex = (j * cfg.headDim + d) / 8u;
      let shift = (d % 8u) * 4u;
      let scaleIndex = d / cfg.blockSize;
      let scale = f32(Scale[scaleIndex]);
      let v = decode_int4(VPacked[packedIndex], shift) * scale;
      acc[d] = acc[d] * prevScale + curScale * v;
    }

    sumExp = sumExp * prevScale + curScale;
    maxScore = newMax;
  }

  for (var d = 0u; d < cfg.headDim; d++) {
    let outIdx = token * cfg.headDim + d;
    Out[outIdx] = select(f16(0.0), f16(acc[d] / max(sumExp, 1e-20)), token > 0u);
  }
}
