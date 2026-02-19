import fs from "node:fs";

const out = process.argv[2] || "weights/model-1b-int4.base64.json";

const header = Buffer.alloc(64);
header.writeUInt32LE(0x434C4946, 0); // CLIF
header.writeUInt32LE(1, 4);
header.writeUInt32LE(2048, 8);
header.writeUInt32LE(16, 12);
header.writeUInt32LE(50000, 16);
header.writeUInt32LE(4, 20); // int4 enum placeholder

// Small placeholder payload (non-model data) to exercise pipeline.
const payload = Buffer.from("CSS-LLM-PLACEHOLDER", "utf8");
const blob = Buffer.concat([header, payload]);

const artifact = {
  format: "CLIF-1-base64",
  note: "GitHub-safe text artifact; store real model binaries in releases/object storage.",
  byteLength: blob.byteLength,
  dataBase64: blob.toString("base64"),
};

fs.writeFileSync(out, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
console.log(`Wrote Base64 weight artifact: ${out}`);
