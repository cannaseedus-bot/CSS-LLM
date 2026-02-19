import fs from "node:fs";
import path from "node:path";
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

function sha256(s) {
  return crypto.createHash("sha256").update(s).digest("hex");
}

function main() {
  const inputPath = process.argv[2];
  const outputPath = process.argv[3];
  const name = process.argv[4] || path.basename(inputPath || "payload.json");

  if (!inputPath || !outputPath) {
    console.error("Usage: node tools/convert-to-cssllm-format.js <input.json> <output.json> [name]");
    process.exit(1);
  }

  const payload = JSON.parse(fs.readFileSync(inputPath, "utf8"));
  const payloadCanon = JSON.stringify(stable(payload));

  const out = {
    format: "css-llm/metadata-v1",
    version: "1.0.0",
    name,
    canonicalization: {
      encoding: "utf-8",
      sortKeys: true,
      whitespace: "none",
      floatNorm: "ieee754-f64",
    },
    payload,
    payloadHash: sha256(payloadCanon),
  };

  fs.writeFileSync(outputPath, `${JSON.stringify(out, null, 2)}\n`, "utf8");
  console.log(`Converted to CSS-LLM metadata: ${outputPath}`);
}

main();
