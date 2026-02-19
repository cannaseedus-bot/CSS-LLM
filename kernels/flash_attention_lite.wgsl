struct FlashLiteConfig {
  seqLen: u32,
  headDim: u32,
  scale: f32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> cfg: FlashLiteConfig;

const MAX_HEAD_DIM: u32 = 64u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let token = gid.x;
  if (token >= cfg.seqLen || cfg.headDim > MAX_HEAD_DIM) {
    return;
  }

  var maxScore: f32 = -1e30;
  var sumExp: f32 = 0.0;
  var acc: array<f32, MAX_HEAD_DIM>;
  for (var d: u32 = 0u; d < MAX_HEAD_DIM; d++) {
    acc[d] = 0.0;
  }

  for (var j: u32 = 0u; j <= token; j++) {
    var score: f32 = 0.0;

    for (var d: u32 = 0u; d < cfg.headDim; d++) {
      let qv = Q[token * cfg.headDim + d];
      let kv = K[j * cfg.headDim + d];
      score += qv * kv;
    }

    score *= cfg.scale;

    let prevMax = maxScore;
    let nextMax = max(maxScore, score);
    let prevScale = exp(prevMax - nextMax);
    let curScale = exp(score - nextMax);

    for (var d: u32 = 0u; d < cfg.headDim; d++) {
      let vv = V[j * cfg.headDim + d];
      acc[d] = acc[d] * prevScale + curScale * vv;
    }

    sumExp = sumExp * prevScale + curScale;
    maxScore = nextMax;
  }

  for (var d: u32 = 0u; d < cfg.headDim; d++) {
    O[token * cfg.headDim + d] = acc[d] / max(sumExp, 1e-20);
  }
}
