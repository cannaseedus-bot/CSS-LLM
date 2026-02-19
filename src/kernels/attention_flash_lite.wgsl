struct Config {
  seqLen: u32,
  headDim: u32,
  scale: f32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Config;

const TILE: u32 = 16u;
const MAX_HEAD_DIM: u32 = 64u;
var<workgroup> tileQ: array<f32, TILE * MAX_HEAD_DIM>;
var<workgroup> tileK: array<f32, TILE * MAX_HEAD_DIM>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wgId: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let row = wgId.x * TILE + lid.x;
  let col = wgId.y * TILE + lid.y;

  if (row >= cfg.seqLen || col > row || cfg.headDim > MAX_HEAD_DIM) {
    return;
  }

  var score: f32 = 0.0;

  for (var t: u32 = 0u; t < cfg.headDim; t += TILE) {
    let dQ = t + lid.y;
    let dK = t + lid.x;

    let qIndex = row * cfg.headDim + dQ;
    let kIndex = col * cfg.headDim + dK;

    tileQ[lid.x * MAX_HEAD_DIM + lid.y] = select(0.0, Q[qIndex], dQ < cfg.headDim);
    tileK[lid.y * MAX_HEAD_DIM + lid.x] = select(0.0, K[kIndex], dK < cfg.headDim);

    workgroupBarrier();

    let width = min(TILE, cfg.headDim - t);
    for (var kk: u32 = 0u; kk < width; kk++) {
      score += tileQ[lid.x * MAX_HEAD_DIM + kk] * tileK[lid.y * MAX_HEAD_DIM + kk];
    }

    workgroupBarrier();
  }

  score *= cfg.scale;

  let maxVal = score;
  let expScore = exp(score - maxVal);
  let prob = expScore / max(expScore, 1e-20);

  for (var d: u32 = 0u; d < cfg.headDim; d++) {
    let outIndex = row * cfg.headDim + d;
    let vIndex = col * cfg.headDim + d;
    O[outIndex] = O[outIndex] + prob * V[vIndex];
  }
}
