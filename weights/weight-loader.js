export function parseHeader(view) {
  return {
    magic: view.getUint32(0, true),
    version: view.getUint32(4, true),
    hiddenDim: view.getUint32(8, true),
    numLayers: view.getUint32(12, true),
    vocabSize: view.getUint32(16, true),
    quantScheme: view.getUint32(20, true),
  };
}

export function loadInt4(buffer) {
  const view = new DataView(buffer);
  const header = parseHeader(view);
  const weights = new Uint8Array(buffer.slice(64));

  return {
    metadata: header,
    weights,
  };
}

export function decodeBase64Artifact(jsonText) {
  const artifact = JSON.parse(jsonText);
  if (!artifact?.dataBase64) {
    throw new Error("Invalid base64 artifact: missing dataBase64");
  }

  const bytes = Uint8Array.from(atob(artifact.dataBase64), (c) => c.charCodeAt(0));
  return loadInt4(bytes.buffer);
}
