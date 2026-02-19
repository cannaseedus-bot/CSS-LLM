struct SamplerCfg {
  vocabSize: u32,
  topK: u32,
  seed: u32,
};

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> sampledToken: array<u32>;
@group(0) @binding(2) var<uniform> cfg: SamplerCfg;

@compute @workgroup_size(1)
fn main() {
  var bestIdx: u32 = 0u;
  var bestVal: f32 = -1e30;
  for (var i: u32 = 0u; i < cfg.vocabSize; i++) {
    if (logits[i] > bestVal) {
      bestVal = logits[i];
      bestIdx = i;
    }
  }
  sampledToken[0] = bestIdx;
}
