import fs from "node:fs";
import path from "node:path";
import { validateInvariants } from "./invariants.js";

const ALLOWED = new Set([
  "hidden-dim",
  "num-layers",
  "num-attention-heads",
  "num-kv-heads",
  "head-dim",
  "intermediate-dim",
  "vocab-size",
  "max-seq-len",
  "quant-scheme",
  "attention-mechanism",
  "use-moe"
]);

function parseCssVars(cssText) {
  const body = cssText.match(/:root\s*{([\s\S]*?)}/)?.[1] || "";
  const vars = {};
  const unknown = [];

  body.split(";").forEach((line) => {
    const match = line.match(/--([^:]+):\s*(.+)/);
    if (!match) return;
    const key = match[1].trim();
    const value = match[2].trim().replace(/^"|"$/g, "");

    if (!ALLOWED.has(key)) unknown.push(key);
    vars[key] = value;
  });

  return { vars, unknown };
}

function toConfig(vars) {
  return {
    hiddenDim: Number(vars["hidden-dim"]),
    numLayers: Number(vars["num-layers"]),
    numAttentionHeads: Number(vars["num-attention-heads"]),
    numKVHeads: vars["num-kv-heads"] ? Number(vars["num-kv-heads"]) : undefined,
    headDim: vars["head-dim"] ? Number(vars["head-dim"]) : undefined,
    intermediateDim: Number(vars["intermediate-dim"]),
    vocabSize: Number(vars["vocab-size"]),
    maxSeqLen: Number(vars["max-seq-len"]),
    quantScheme: vars["quant-scheme"],
    attentionMechanism: vars["attention-mechanism"],
    useMoE: vars["use-moe"] === "true"
  };
}

function main() {
  const cssPath = process.argv[2];
  if (!cssPath) {
    console.error("Usage: node governance/validate-config.js <governance.css>");
    process.exit(1);
  }

  const text = fs.readFileSync(path.resolve(cssPath), "utf8");
  const { vars, unknown } = parseCssVars(text);

  if (unknown.length > 0) {
    console.error("Invalid governance: unknown properties detected:", unknown);
    process.exit(2);
  }

  const config = toConfig(vars);
  const invariants = validateInvariants(config);
  if (!invariants.ok) {
    console.error("Invalid governance invariants:");
    invariants.errors.forEach((err) => console.error(`- ${err}`));
    process.exit(3);
  }

  console.log("Governance config valid.");
}

main();
