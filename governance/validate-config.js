import fs from "node:fs";
import path from "node:path";
import { validateInvariants } from "./invariants.js";

const ALLOWED = new Set([
  "model-name",
  "hidden-dim",
  "num-layers",
  "num-heads",
  "num-attention-heads",
  "num-kv-heads",
  "head-dim",
  "ffn-dim",
  "intermediate-dim",
  "vocab-size",
  "context-length",
  "max-seq-len",
  "precision",
  "quant-scheme",
  "attention-kernel",
  "attention-mechanism",
  "norm-type",
  "activation",
  "activation-type",
  "sampling",
  "sampling-default",
  "temperature",
  "top-p",
  "sampling-temperature",
  "sampling-top-p",
  "deterministic-reduction",
  "allow-gpu-atomics",
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
    modelName: vars["model-name"],
    hiddenDim: Number(vars["hidden-dim"]),
    numLayers: Number(vars["num-layers"]),
    numAttentionHeads: Number(vars["num-heads"] || vars["num-attention-heads"]),
    numKVHeads: vars["num-kv-heads"] ? Number(vars["num-kv-heads"]) : undefined,
    headDim: vars["head-dim"] ? Number(vars["head-dim"]) : undefined,
    intermediateDim: Number(vars["ffn-dim"] || vars["intermediate-dim"]),
    vocabSize: Number(vars["vocab-size"]),
    maxSeqLen: Number(vars["context-length"] || vars["max-seq-len"]),
    quantScheme: vars["precision"] || vars["quant-scheme"],
    attentionMechanism: vars["attention-kernel"] || vars["attention-mechanism"],
    normType: vars["norm-type"],
    activation: vars["activation"] || vars["activation-type"],
    sampling: vars["sampling"] || vars["sampling-default"],
    temperature: Number(vars["temperature"] || vars["sampling-temperature"]),
    topP: Number(vars["top-p"] || vars["sampling-top-p"]),
    deterministicReduction: vars["deterministic-reduction"] === "true",
    allowGpuAtomics: vars["allow-gpu-atomics"] === "true",
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
