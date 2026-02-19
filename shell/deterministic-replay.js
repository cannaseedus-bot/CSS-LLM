import crypto from "node:crypto";

function sha256(input) {
  return crypto.createHash("sha256").update(input).digest("hex");
}

export function createProofEnvelope({
  version = "1.0",
  governanceHash,
  modelHash,
  weightHash,
  kernelHash,
  inputHash,
  rngSeed,
  outputHash,
}) {
  const canonicalModelHash = modelHash ?? weightHash;
  const proofConcat = `${governanceHash}|${canonicalModelHash}|${kernelHash}|${inputHash}|${rngSeed}|${outputHash}`;
  const proofHash = sha256(proofConcat);

  return {
    version,
    modelHash: canonicalModelHash,
    governanceHash,
    // legacy alias retained for compatibility with older callers
    weightHash: canonicalModelHash,
    kernelHash,
    inputHash,
    rngSeed,
    outputHash,
    proofHash,
  };
}

export function verifyProofEnvelope(envelope) {
  const recomputed = createProofEnvelope(envelope);
  return {
    ok: recomputed.proofHash === envelope.proofHash,
    expected: envelope.proofHash,
    actual: recomputed.proofHash,
  };
}
