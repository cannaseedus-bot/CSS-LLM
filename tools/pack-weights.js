import fs from "node:fs";
import path from "node:path";

const outDir = process.argv[2] || "weights/shards";
const baseName = process.argv[3] || "css_llm";
const shardCount = Number(process.argv[4] || 3);

if (!Number.isInteger(shardCount) || shardCount < 1) {
  console.error("shardCount must be a positive integer");
  process.exit(1);
}

fs.mkdirSync(outDir, { recursive: true });

const header = Buffer.alloc(64);
header.writeUInt32LE(0x434C4946, 0); // CLIF
header.writeUInt32LE(1, 4);
header.writeUInt32LE(2048, 8);
header.writeUInt32LE(16, 12);
header.writeUInt32LE(50000, 16);
header.writeUInt32LE(4, 20);

const payload = Buffer.from("CSS-LLM-SHARDED-PLACEHOLDER", "utf8");
const fullBlob = Buffer.concat([header, payload]);

const shardSize = Math.ceil(fullBlob.length / shardCount);
const index = {
  format: "CLIF-1-base64-sharded",
  baseName,
  shardCount,
  totalBytes: fullBlob.length,
  shards: [],
};

for (let i = 0; i < shardCount; i += 1) {
  const start = i * shardSize;
  const end = Math.min(start + shardSize, fullBlob.length);
  const shard = fullBlob.slice(start, end);
  const num = String(i + 1).padStart(3, "0");
  const fileName = `${baseName}_${num}.bin.base64.json`;
  const fullPath = path.join(outDir, fileName);

  const artifact = {
    format: "CLIF-1-base64-shard",
    shardIndex: i,
    shardNumber: i + 1,
    shardCount,
    byteOffset: start,
    byteLength: shard.length,
    dataBase64: shard.toString("base64"),
  };

  fs.writeFileSync(fullPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  index.shards.push(fileName);
}

const indexPath = path.join(outDir, `${baseName}_index.json`);
fs.writeFileSync(indexPath, `${JSON.stringify(index, null, 2)}\n`, "utf8");

console.log(`Wrote ${shardCount} shard artifact(s) + index to ${outDir}`);
