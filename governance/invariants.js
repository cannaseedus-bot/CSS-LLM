export function validateInvariants(config) {
  const errors = [];

  if (config.hiddenDim % config.numAttentionHeads !== 0) {
    errors.push("hiddenDim must be divisible by numAttentionHeads");
  }

  const derivedHeadDim = config.hiddenDim / config.numAttentionHeads;
  if (config.headDim !== undefined && config.headDim !== derivedHeadDim) {
    errors.push("headDim must equal hiddenDim / numAttentionHeads");
  }

  if (config.intermediateDim < config.hiddenDim * 2) {
    errors.push("intermediateDim must be at least 2x hiddenDim");
  }

  if (config.numKVHeads !== undefined && config.numKVHeads > config.numAttentionHeads) {
    errors.push("numKVHeads must be <= numAttentionHeads");
  }

  return {
    ok: errors.length === 0,
    errors,
  };
}
