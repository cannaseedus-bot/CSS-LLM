import fs from "node:fs";
import crypto from "node:crypto";

const file = process.argv[2];
if (!file) {
  console.error("Usage: node tools/hash-model.js <model.bin>");
  process.exit(1);
}

const bytes = fs.readFileSync(file);
const hash = crypto.createHash("sha256").update(bytes).digest("hex");
console.log(hash);
