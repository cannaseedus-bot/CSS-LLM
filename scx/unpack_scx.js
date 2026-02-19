import fs from "node:fs";

function loadBytes(path) {
  const raw = fs.readFileSync(path);
  if (path.endsWith('.base64.json')) {
    const wrapped = JSON.parse(raw.toString('utf8'));
    return Buffer.from(wrapped.data, 'base64');
  }
  return raw;
}

export function unpackScx(path) {
  const data = loadBytes(path);
  const magic = data.subarray(0, 4).toString("utf8");
  const version = data.readUInt32LE(4);
  const tensorCount = data.readUInt32LE(8);

  if (magic !== "SCXM") {
    throw new Error(`Invalid SCX magic: ${magic}`);
  }

  return { path, magic, version, tensorCount, bytes: data.byteLength };
}
