// ============================================================
// FUSED INT4 RMSNORM (deterministic reduction)
// ------------------------------------------------------------
// Assumptions:
// - One token per workgroup (gid.x token index)
// - hiddenDim is a multiple of workgroup size (best efficiency)
// - GammaPacked stores 8 int4 values per u32
// - GammaScale stores per-block scales (blockSize typically 128)
// ============================================================

struct RMSConfig {
  hiddenDim: u32,
  blockSize: u32,
  eps: f32,
}

@group(0) @binding(0)
var<storage, read> X : array<f16>;

@group(0) @binding(1)
var<storage, read> GammaPacked : array<u32>;

@group(0) @binding(2)
var<storage, read> GammaScale : array<f16>;

@group(0) @binding(3)
var<storage, read_write> Y : array<f16>;

@group(0) @binding(4)
var<uniform> cfg : RMSConfig;

var<workgroup> partial : array<f32, 256>;

fn decode_int4(v: u32, shift: u32) -> f32 {
  let raw = (v >> shift) & 0xFu;
  if (raw >= 8u) {
    return f32(i32(raw) - 16);
  }
  return f32(raw);
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let token = gid.x;
  let base = token * cfg.hiddenDim;

  // ---- Sum of squares (fixed stride order) ----
  var sum : f32 = 0.0;
  for (var i = lid.x; i < cfg.hiddenDim; i += 256u) {
    let val = f32(X[base + i]);
    sum += val * val;
  }

  partial[lid.x] = sum;
  workgroupBarrier();

  // ---- Deterministic tree reduction ----
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) {
      partial[lid.x] += partial[lid.x + stride];
    }
    stride = stride >> 1u;
    workgroupBarrier();
  }

  let rms = sqrt(partial[0] / f32(cfg.hiddenDim) + cfg.eps);

  // ---- Normalize + dequant gamma ----
  for (var i = lid.x; i < cfg.hiddenDim; i += 256u) {
    let val = f32(X[base + i]);

    let packedIndex = i / 8u;
    let shift = (i % 8u) * 4u;

    let scaleIndex = i / cfg.blockSize;
    let scale = f32(GammaScale[scaleIndex]);

    let gamma = decode_int4(GammaPacked[packedIndex], shift) * scale;
    Y[base + i] = f16((val / rms) * gamma);
  }
}
