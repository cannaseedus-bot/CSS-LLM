// ============================================================
// Deterministic sampler primitive (PCG32)
// ============================================================

struct RNGState {
  state: u32,
  inc: u32,
}

fn pcg32(state: ptr<function, RNGState>) -> u32 {
  let old = (*state).state;
  (*state).state = old * 747796405u + (*state).inc;
  let xorshifted = ((old >> 18u) ^ old) >> 27u;
  let rot = (old >> 27u) & 31u;
  return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

fn random_float(state: ptr<function, RNGState>) -> f32 {
  return f32(pcg32(state)) / 4294967296.0;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = gid;
}
