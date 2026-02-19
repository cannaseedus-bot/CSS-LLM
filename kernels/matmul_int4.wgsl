// ============================================================
// INT4 TILED MATMUL (WebGPU/iGPU-safe scaffold)
// Computes: Y[M, N] = X[M, K] · W[N, K]^T
// - X: fp16 activations
// - W: symmetric int4 packed (8 weights per u32)
// - S: fp16 per-row/per-block scale for W
// - Y: fp16 output, f32 accumulation
// ============================================================

struct MatmulConfig {
  M: u32,
  N: u32,
  K: u32,
  blockSize: u32,
}

@group(0) @binding(0) var<storage, read> X : array<f16>;
@group(0) @binding(1) var<storage, read> W : array<u32>;
@group(0) @binding(2) var<storage, read> S : array<f16>;
@group(0) @binding(3) var<storage, read_write> Y : array<f16>;
@group(0) @binding(4) var<uniform> cfg : MatmulConfig;

const TILE_M: u32 = 8u;
const TILE_N: u32 = 8u;
const TILE_K: u32 = 128u;
const THREADS: u32 = TILE_M * TILE_N;

var<workgroup> tileX: array<f32, TILE_M * TILE_K>;
var<workgroup> tileW: array<f32, TILE_N * TILE_K>;

fn decode_int4(nibble: u32) -> f32 {
  if (nibble >= 8u) {
    return f32(i32(nibble) - 16);
  }
  return f32(nibble);
}

@compute @workgroup_size(TILE_M, TILE_N, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  if (cfg.blockSize == 0u) {
    return;
  }

  let row = wid.x * TILE_M + lid.x;
  let col = wid.y * TILE_N + lid.y;

  var acc: f32 = 0.0;
  let tid = lid.y * TILE_M + lid.x;
  let blocksPerRow = (cfg.K + cfg.blockSize - 1u) / cfg.blockSize;

  for (var kBase = 0u; kBase < cfg.K; kBase += TILE_K) {
    // Cooperative X tile load.
    for (var t = tid; t < TILE_M * TILE_K; t += THREADS) {
      let localRow = t / TILE_K;
      let localK = t % TILE_K;
      let globalRow = wid.x * TILE_M + localRow;
      let globalK = kBase + localK;

      var xv: f32 = 0.0;
      if (globalRow < cfg.M && globalK < cfg.K) {
        xv = f32(X[globalRow * cfg.K + globalK]);
      }
      tileX[t] = xv;
    }

    // Cooperative W tile load + int4 decode + scale.
    for (var t = tid; t < TILE_N * TILE_K; t += THREADS) {
      let localCol = t / TILE_K;
      let localK = t % TILE_K;
      let globalCol = wid.y * TILE_N + localCol;
      let globalK = kBase + localK;

      var wv: f32 = 0.0;
      if (globalCol < cfg.N && globalK < cfg.K) {
        let linear = globalCol * cfg.K + globalK;
        let packed = W[linear / 8u];
        let nibble = (packed >> ((linear % 8u) * 4u)) & 0xFu;
        let scaleIdx = globalCol * blocksPerRow + (globalK / cfg.blockSize);
        wv = decode_int4(nibble) * f32(S[scaleIdx]);
      }
      tileW[t] = wv;
    }

    workgroupBarrier();

    if (row < cfg.M && col < cfg.N) {
      let kLimit = min(TILE_K, cfg.K - kBase);
      for (var k = 0u; k < kLimit; k++) {
        let a = tileX[lid.x * TILE_K + k];
        let b = tileW[lid.y * TILE_K + k];
        acc += a * b;
      }
    }

    workgroupBarrier();
  }

  if (row < cfg.M && col < cfg.N) {
    Y[row * cfg.N + col] = f16(acc);
  }
}
