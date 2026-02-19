import crypto from "node:crypto";

function sha256(input) {
  return crypto.createHash("sha256").update(input).digest("hex");
}

export function createReplayRecord({
  modelBytes,
  governanceCss,
  kernelBundle,
  inputTokens,
  rngSeed,
  samplingConfig,
  outputTokens,
}) {
  return {
    modelHash: sha256(modelBytes),
    governanceHash: sha256(governanceCss),
    kernelHash: sha256(kernelBundle),
    inputHash: sha256(JSON.stringify(inputTokens)),
    rngSeed,
    samplingConfig,
    outputHash: sha256(JSON.stringify(outputTokens)),
  };
}

export function verifyReplay(expectedRecord, outputTokens) {
  const outputHash = sha256(JSON.stringify(outputTokens));
  return {
    ok: outputHash === expectedRecord.outputHash,
    outputHash,
    expected: expectedRecord.outputHash,
  };
}
