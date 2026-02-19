@group(0) @binding(0) var<storage, read> up: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

fn sigmoid(v: f32) -> f32 {
  return 1.0 / (1.0 + exp(-v));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let g = gate[i] * sigmoid(gate[i]);
  out[i] = up[i] * g;
}
