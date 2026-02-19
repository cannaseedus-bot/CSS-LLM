import fs from "node:fs/promises";

async function readScxBytes(path) {
  const file = await fs.readFile(path);
  if (path.endsWith('.base64.json')) {
    const wrapped = JSON.parse(file.toString('utf8'));
    return Buffer.from(wrapped.data, 'base64');
  }
  return file;
}

export async function loadScxHeader(path) {
  const header = await readScxBytes(path);
  const magic = header.subarray(0, 4).toString("utf8");
  const version = header.readUInt32LE(4);
  if (magic !== "SCXM") {
    throw new Error(`Invalid SCX magic: ${magic}`);
  }

  return {
    path,
    magic,
    version,
    bytes: header.byteLength,
  };
}
