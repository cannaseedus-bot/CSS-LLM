// ============================================================
// EXPERT-PARALLEL INT4 MoE CLUSTER NODE (deterministic 2-phase)
//
// Phase 0:
//   dispatchWorkgroups(batchSize, topK)
//   compute slot output into Scratch[token, slot, o]
//
// Phase 1:
//   dispatchWorkgroups(batchSize, 1)
//   Output[token, o] = slot0 + slot1 in fixed order
//
// This avoids atomics and inter-workgroup races.
// ============================================================

struct MoEConfig {
  hiddenDim: u32,
  intermediateDim: u32,
  blockSize: u32,
  numExperts: u32,
  topK: u32,
  phase: u32,
}

@group(0) @binding(0) var<storage, read> X : array<f16>;
@group(0) @binding(1) var<storage, read> ExpertW1 : array<u32>;
@group(0) @binding(2) var<storage, read> ExpertW2 : array<u32>;
@group(0) @binding(3) var<storage, read> Scale : array<f16>;
@group(0) @binding(4) var<storage, read> RouterIndex : array<u32>;
@group(0) @binding(5) var<storage, read> RouterWeight : array<f32>;
@group(0) @binding(6) var<storage, read_write> Scratch : array<f16>;
@group(0) @binding(7) var<storage, read_write> Output : array<f16>;
@group(0) @binding(8) var<uniform> cfg : MoEConfig;

fn decode_int4(v: u32, shift: u32) -> f32 {
  let raw = (v >> shift) & 0xFu;
  if (raw >= 8u) { return f32(i32(raw) - 16); }
  return f32(raw);
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let token = wid.x;

  if (cfg.phase == 0u) {
    let slot = wid.y;
    if (slot >= cfg.topK) {
      return;
    }

    let expertId = RouterIndex[token * cfg.topK + slot];
    let routerW = RouterWeight[token * cfg.topK + slot];
    let xBase = token * cfg.hiddenDim;
    let expertOffset = expertId * cfg.hiddenDim * cfg.intermediateDim;

    for (var o = lid.x; o < cfg.intermediateDim; o += 128u) {
      var gate: f32 = 0.0;
      var up: f32 = 0.0;

      for (var i = 0u; i < cfg.hiddenDim; i++) {
        let xVal = f32(X[xBase + i]);
        let wLinear = expertOffset + o * cfg.hiddenDim + i;
        let packedIndex = wLinear / 8u;
        let shift = (wLinear % 8u) * 4u;
        let scale = f32(Scale[i / cfg.blockSize]);

        gate += xVal * decode_int4(ExpertW1[packedIndex], shift) * scale;
        up += xVal * decode_int4(ExpertW2[packedIndex], shift) * scale;
      }

      let silu = gate / (1.0 + exp(-gate));
      let weighted = silu * up * routerW;
      let scratchIndex = (token * cfg.topK + slot) * cfg.intermediateDim + o;
      Scratch[scratchIndex] = f16(weighted);
    }

    return;
  }

  // phase == 1: fixed-order reduction across slots (slot0 + slot1 + ...)
  if (wid.y != 0u) {
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
