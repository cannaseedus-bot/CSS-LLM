// ============================================================
// FUSED RMSNORM + RESIDUAL + INT4 MATMUL (deterministic scaffold)
// ============================================================

struct FuseConfig {
  hiddenDim: u32,
  outDim: u32,
  blockSize: u32,
  eps: f32,
}

@group(0) @binding(0) var<storage, read> X : array<f16>;
@group(0) @binding(1) var<storage, read> Residual : array<f16>;
@group(0) @binding(2) var<storage, read> GammaPacked : array<u32>;
@group(0) @binding(3) var<storage, read> GammaScale : array<f16>;
@group(0) @binding(4) var<storage, read> WPacked : array<u32>;
@group(0) @binding(5) var<storage, read> WScale : array<f16>;
@group(0) @binding(6) var<storage, read_write> Y : array<f16>;
@group(0) @binding(7) var<uniform> cfg : FuseConfig;

var<workgroup> partial : array<f32, 256>;

fn decode_int4(v: u32, shift: u32) -> f32 {
  let raw = (v >> shift) & 0xFu;
  if (raw >= 8u) { return f32(i32(raw) - 16); }
  return f32(raw);
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let token = gid.x;
  let base = token * cfg.hiddenDim;

  var sum: f32 = 0.0;
  for (var i = lid.x; i < cfg.hiddenDim; i += 256u) {
    let v = f32(X[base + i]);
    sum += v * v;
  }

  partial[lid.x] = sum;
  workgroupBarrier();

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

  for (var o = lid.x; o < cfg.outDim; o += 256u) {
    var acc: f32 = 0.0;
    for (var i = 0u; i < cfg.hiddenDim; i++) {
      let xVal = f32(X[base + i]);

      let gIdx = i / 8u;
      let gShift = (i % 8u) * 4u;
      let gScale = f32(GammaScale[i / cfg.blockSize]);
      let gamma = decode_int4(GammaPacked[gIdx], gShift) * gScale;
      let normVal = (xVal / rms) * gamma;

      let wLinear = o * cfg.hiddenDim + i;
      let wIdx = wLinear / 8u;
      let wShift = (wLinear % 8u) * 4u;
      let wS = f32(WScale[i / cfg.blockSize]);
      let wVal = decode_int4(WPacked[wIdx], wShift) * wS;

      acc += normVal * wVal;
    }

    let res = f32(Residual[token * cfg.outDim + o]);
    Y[token * cfg.outDim + o] = f16(acc + res);
  }
}
