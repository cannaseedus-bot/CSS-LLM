// ============================================================
// EXPERT-PARALLEL INT4 MoE CLUSTER NODE (WebGPU/iGPU friendly)
//
// Dispatch model:
//   1) phase_compute_slots: dispatchWorkgroups(batchSize, topK)
//      each workgroup computes one (token, slot) expert output and writes
//      weighted results into Scratch[token, slot, o].
//   2) phase_reduce_slots: dispatchWorkgroups(batchSize, 1)
//      fixed-order reduction across slots into Output[token, o].
//
// Determinism:
//   - no atomics
//   - fixed traversal order for i/o/slot loops
//   - fixed slot combine sequence (slot 0 -> slot N-1)
// ============================================================

struct MoEConfig {
  batchSize: u32,
  hiddenDim: u32,
  intermediateDim: u32,
  blockSize: u32,
  numExperts: u32,
  topK: u32,
};

@group(0) @binding(0) var<storage, read> X : array<f16>;

// Packed expert weights, contiguous by expert then output row:
// [expert0 rows][expert1 rows]...
@group(0) @binding(1) var<storage, read> ExpertW1 : array<u32>;
@group(0) @binding(2) var<storage, read> ExpertW2 : array<u32>;

// Block scales laid out as:
// scaleIndex = ((expertId * intermediateDim + o) * blocksPerRow) + (i / blockSize)
@group(0) @binding(3) var<storage, read> ScaleW1 : array<f16>;
@group(0) @binding(4) var<storage, read> ScaleW2 : array<f16>;

// Router outputs sorted by descending weight per token.
@group(0) @binding(5) var<storage, read> RouterIndex : array<u32>;
@group(0) @binding(6) var<storage, read> RouterWeight : array<f32>;

// Scratch tensor shape: [batchSize, topK, intermediateDim]
@group(0) @binding(7) var<storage, read_write> Scratch : array<f16>;
// Final output shape: [batchSize, intermediateDim]
@group(0) @binding(8) var<storage, read_write> Output : array<f16>;

@group(0) @binding(9) var<uniform> cfg : MoEConfig;

fn decode_int4(v: u32, shift: u32) -> f32 {
  let raw = (v >> shift) & 0xFu;
  if (raw >= 8u) { return f32(i32(raw) - 16); }
  return f32(raw);
}

fn packed_weight_index(expertId: u32, o: u32, i: u32, hiddenDim: u32, intermediateDim: u32) -> u32 {
  let linear = expertId * (hiddenDim * intermediateDim) + o * hiddenDim + i;
  return linear / 8u;
}

fn scale_index(expertId: u32, o: u32, i: u32, blockSize: u32, hiddenDim: u32, intermediateDim: u32) -> u32 {
  let blocksPerRow = (hiddenDim + blockSize - 1u) / blockSize;
  return (expertId * intermediateDim + o) * blocksPerRow + (i / blockSize);
}

@compute @workgroup_size(128)
fn phase_compute_slots(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let token = wid.x;
  let slot = wid.y;

  if (token >= cfg.batchSize || slot >= cfg.topK || cfg.blockSize == 0u) {
    return;
  }

  let expertId = RouterIndex[token * cfg.topK + slot];
  if (expertId >= cfg.numExperts) {
    return;
  }

  let routerW = RouterWeight[token * cfg.topK + slot];
  let xBase = token * cfg.hiddenDim;

  for (var o = lid.x; o < cfg.intermediateDim; o += 128u) {
    var gate: f32 = 0.0;
    var up: f32 = 0.0;

    for (var i = 0u; i < cfg.hiddenDim; i++) {
      let xVal = f32(X[xBase + i]);
      let wPackedIndex = packed_weight_index(expertId, o, i, cfg.hiddenDim, cfg.intermediateDim);
      let shift = ((expertId * (cfg.hiddenDim * cfg.intermediateDim) + o * cfg.hiddenDim + i) % 8u) * 4u;

      let sIdx = scale_index(expertId, o, i, cfg.blockSize, cfg.hiddenDim, cfg.intermediateDim);
      let s1 = f32(ScaleW1[sIdx]);
      let s2 = f32(ScaleW2[sIdx]);

      gate += xVal * decode_int4(ExpertW1[wPackedIndex], shift) * s1;
      up += xVal * decode_int4(ExpertW2[wPackedIndex], shift) * s2;
    }

    let silu = gate / (1.0 + exp(-gate));
    let weighted = silu * up * routerW;
    let scratchIndex = (token * cfg.topK + slot) * cfg.intermediateDim + o;
    Scratch[scratchIndex] = f16(weighted);
  }
}

@compute @workgroup_size(128)
fn phase_reduce_slots(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let token = wid.x;
  if (token >= cfg.batchSize) {
    return;
  }

  for (var o = lid.x; o < cfg.intermediateDim; o += 128u) {
    var acc: f32 = 0.0;
    for (var slot = 0u; slot < cfg.topK; slot++) {
      let scratchIndex = (token * cfg.topK + slot) * cfg.intermediateDim + o;
      acc += f32(Scratch[scratchIndex]);
    }
    Output[token * cfg.intermediateDim + o] = f16(acc);
  }
}
