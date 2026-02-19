struct MatmulConfig {
  M: u32,
  N: u32,
  K: u32,
  scale: f32,
};

@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> config: MatmulConfig;

fn decode_int4(word: u32, idx: u32) -> f32 {
  let shift = (idx & 7u) * 4u;
  let nibble = (word >> shift) & 0xFu;
  let signed = select(i32(nibble), i32(nibble) - 16, nibble > 7u);
  return f32(signed) * config.scale;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  let col = id.y;

  if (row >= config.M || col >= config.N) {
    return;
  }

  var acc: f32 = 0.0;

  for (var k: u32 = 0u; k < config.K; k++) {
    let aIndex = row * config.K + k;
    let bIndex = k * config.N + col;

    let aWord = A[aIndex >> 3u];
    let bWord = B[bIndex >> 3u];

    let aVal = decode_int4(aWord, aIndex);
    let bVal = decode_int4(bWord, bIndex);

    acc += aVal * bVal;
  }

  C[row * config.N + col] = acc;
}
