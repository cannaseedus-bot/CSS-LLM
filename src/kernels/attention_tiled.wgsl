struct AttentionCfg {
  seqLen: u32,
  headDim: u32,
  scale: f32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> cfg: AttentionCfg;

const TILE: u32 = 16u;
const MAX_HEAD_DIM: u32 = 64u;
var<workgroup> tileQ: array<f32, TILE * MAX_HEAD_DIM>;
var<workgroup> tileK: array<f32, TILE * MAX_HEAD_DIM>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x * TILE + lid.x;
  let col = wid.y * TILE + lid.y;

  if (row >= cfg.seqLen || col >= cfg.seqLen || cfg.headDim > MAX_HEAD_DIM || col > row) {
    return;
  }

  var score: f32 = 0.0;

  for (var t: u32 = 0u; t < cfg.headDim; t += TILE) {
    for (var kk: u32 = 0u; kk < TILE; kk++) {
      let dim = t + kk;
      let localQ = lid.x * MAX_HEAD_DIM + kk;
      let localK = lid.y * MAX_HEAD_DIM + kk;
      if (dim < cfg.headDim) {
        tileQ[localQ] = q[row * cfg.headDim + dim];
        tileK[localK] = k[col * cfg.headDim + dim];
      } else {
        tileQ[localQ] = 0.0;
        tileK[localK] = 0.0;
      }
    }

    workgroupBarrier();
    for (var kk: u32 = 0u; kk < TILE; kk++) {
      score += tileQ[lid.x * MAX_HEAD_DIM + kk] * tileK[lid.y * MAX_HEAD_DIM + kk];
    }
    workgroupBarrier();
  }

  scores[row * cfg.seqLen + col] = score * cfg.scale;
}
