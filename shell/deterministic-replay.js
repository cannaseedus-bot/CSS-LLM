import crypto from "node:crypto";

function sha256(input) {
  return crypto.createHash("sha256").update(input).digest("hex");
}

export function createProofEnvelope({
  version = "1.0",
  governanceHash,
  weightHash,
  kernelHash,
  inputHash,
  rngSeed,
  outputHash,
}) {
  const proofConcat = `${governanceHash}|${weightHash}|${kernelHash}|${inputHash}|${rngSeed}|${outputHash}`;
  const proofHash = sha256(proofConcat);

  return {
    version,
    governanceHash,
    weightHash,
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
