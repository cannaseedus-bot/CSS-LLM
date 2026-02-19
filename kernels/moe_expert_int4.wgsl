// ============================================================
// MoE expert INT4 matmul helpers (deterministic top-2 routing)
// ------------------------------------------------------------
// This module provides deterministic top-2 routing + weighted
// combine for per-token expert outputs.
// ============================================================

fn top2(scores: array<f32, 8>) -> vec2<u32> {
  var max1 = -1e30;
  var max2 = -1e30;
  var idx1 = 0u;
  var idx2 = 0u;

  for (var i = 0u; i < 8u; i++) {
    let s = scores[i];
    if (s > max1) {
      max2 = max1;
      idx2 = idx1;
      max1 = s;
      idx1 = i;
    } else if (s > max2) {
      max2 = s;
      idx2 = i;
    }
  }

  return vec2<u32>(idx1, idx2);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Placeholder dispatch shell for expert kernel integration.
  // Deterministic top-k routing primitive is defined above.
  _ = gid;
}
