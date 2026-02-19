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

export function decodeBase64Shards(shardJsonTexts) {
  const shards = shardJsonTexts.map((text) => JSON.parse(text));
  for (const shard of shards) {
    if (shard.format !== "CLIF-1-base64-shard") {
      throw new Error("Invalid shard artifact format");
    }
  }

  shards.sort((a, b) => a.byteOffset - b.byteOffset);

  let total = 0;
  const arrays = shards.map((shard) => {
    const arr = Uint8Array.from(atob(shard.dataBase64), (c) => c.charCodeAt(0));
    total += arr.length;
    return arr;
  });

  const merged = new Uint8Array(total);
  let offset = 0;
  for (const arr of arrays) {
    merged.set(arr, offset);
    offset += arr.length;
  }

  return loadInt4(merged.buffer);
}
