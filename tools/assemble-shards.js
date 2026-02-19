import fs from "node:fs";
import path from "node:path";

const indexPath = process.argv[2] || "weights/shards/css_llm_index.json";
const outputPath = process.argv[3] || "weights/model.bin";

function decodeBase64(base64) {
  return Buffer.from(base64, "base64");
}

function main() {
  const index = JSON.parse(fs.readFileSync(indexPath, "utf8"));
  if (index.format !== "CLIF-1-base64-sharded") {
    throw new Error("Invalid shard index format");
  }

  const baseDir = path.dirname(indexPath);
  const chunks = [];

  for (const shardFile of index.shards) {
    const shardPath = path.join(baseDir, shardFile);
    const shard = JSON.parse(fs.readFileSync(shardPath, "utf8"));
    if (shard.format !== "CLIF-1-base64-shard") {
      throw new Error(`Invalid shard format: ${shardFile}`);
    }
    chunks.push({ offset: shard.byteOffset, bytes: decodeBase64(shard.dataBase64) });
  }

  chunks.sort((a, b) => a.offset - b.offset);
  const assembled = Buffer.concat(chunks.map((c) => c.bytes));

  if (assembled.length !== index.totalBytes) {
    throw new Error(`Assembled byte length mismatch: expected ${index.totalBytes}, got ${assembled.length}`);
  }

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, assembled);
  console.log(`Assembled binary: ${outputPath} (${assembled.length} bytes)`);
}

main();
