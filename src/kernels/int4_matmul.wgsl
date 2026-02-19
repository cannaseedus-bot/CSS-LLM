struct MatmulCfg {
  m: u32,
  n: u32,
  k: u32,
  scale: f32,
};

@group(0) @binding(0) var<storage, read> aPacked: array<u32>;
@group(0) @binding(1) var<storage, read> bPacked: array<u32>;
@group(0) @binding(2) var<storage, read_write> outF32: array<f32>;
@group(0) @binding(3) var<uniform> cfg: MatmulCfg;

fn decodeInt4(word: u32, linear: u32) -> f32 {
  let shift = (linear & 7u) * 4u;
  let nibble = (word >> shift) & 0xFu;
  let signed = select(i32(nibble), i32(nibble) - 16, nibble > 7u);
  return f32(signed) * cfg.scale;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= cfg.m || col >= cfg.n) { return; }

  var acc: f32 = 0.0;
  for (var kk: u32 = 0u; kk < cfg.k; kk++) {
    let aIdx = row * cfg.k + kk;
    let bIdx = kk * cfg.n + col;
    let a = decodeInt4(aPacked[aIdx >> 3u], aIdx);
    let b = decodeInt4(bPacked[bIdx >> 3u], bIdx);
    acc += a * b;
  }

  outF32[row * cfg.n + col] = acc;
}
