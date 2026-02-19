import fs from "node:fs";
import { verifyReplay } from "../shell/replay-engine.js";

const [recordPath, outputPath] = process.argv.slice(2);

if (!recordPath || !outputPath) {
  console.error("Usage: node replay/verify.js <record.json> <output.json>");
  process.exit(1);
}

const record = JSON.parse(fs.readFileSync(recordPath, "utf8"));
const output = JSON.parse(fs.readFileSync(outputPath, "utf8"));

const result = verifyReplay(record, output);
if (!result.ok) {
  console.error("Replay verification failed", result);
  process.exit(2);
}

console.log("Replay verified", result);
