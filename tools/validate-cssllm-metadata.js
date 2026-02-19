import fs from "node:fs";
import crypto from "node:crypto";

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") {
    return Object.keys(value)
      .sort()
      .reduce((acc, k) => {
        acc[k] = stable(value[k]);
        return acc;
      }, {});
  }
  return value;
}

const file = process.argv[2];
if (!file) {
  console.error("Usage: node tools/validate-cssllm-metadata.js <file.json>");
  process.exit(1);
}

const obj = JSON.parse(fs.readFileSync(file, "utf8"));
const required = ["format", "version", "name", "canonicalization", "payload", "payloadHash"];
for (const key of required) {
  if (!(key in obj)) {
    console.error(`Missing key: ${key}`);
    process.exit(2);
  }
}
if (obj.format !== "css-llm/metadata-v1") {
  console.error("Invalid format");
  process.exit(3);
}
const canon = JSON.stringify(stable(obj.payload));
const hash = crypto.createHash("sha256").update(canon).digest("hex");
if (hash !== obj.payloadHash) {
  console.error("payloadHash mismatch");
  process.exit(4);
}
console.log("CSS-LLM metadata valid:", file);
