import fs from "node:fs";
import path from "node:path";
import https from "node:https";

function download(url, outPath) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(outPath);
    https
      .get(url, (res) => {
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          file.close();
          if (fs.existsSync(outPath)) fs.unlinkSync(outPath);
          download(res.headers.location, outPath).then(resolve).catch(reject);
          return;
        }

        if (res.statusCode !== 200) {
          reject(new Error(`Failed download (${res.statusCode}): ${url}`));
          return;
        }

        res.pipe(file);
        file.on("finish", () => {
          file.close();
          resolve();
        });
      })
      .on("error", (err) => {
        file.close();
        reject(err);
      });
  });
}

async function main() {
  const indexUrl = process.argv[2];
  const outDir = process.argv[3] || "weights/shards";

  if (!indexUrl) {
    console.error("Usage: node tools/fetch-weights.js <index_json_url> [output_dir]");
    process.exit(1);
  }

  if (!indexUrl.startsWith("https://")) {
    console.error("Only https:// URLs are allowed");
    process.exit(2);
  }

  fs.mkdirSync(outDir, { recursive: true });
  const localIndexPath = path.join(outDir, path.basename(new URL(indexUrl).pathname) || "css_llm_index.json");

  await download(indexUrl, localIndexPath);
  const index = JSON.parse(fs.readFileSync(localIndexPath, "utf8"));

  if (index.format !== "CLIF-1-base64-sharded") {
    throw new Error("Index is not CLIF-1-base64-sharded");
  }

  const indexBase = indexUrl.slice(0, indexUrl.lastIndexOf("/") + 1);
  for (const shardFile of index.shards) {
    const shardUrl = `${indexBase}${shardFile}`;
    const shardOut = path.join(outDir, shardFile);
    await download(shardUrl, shardOut);
  }

  console.log(`Downloaded shard index + ${index.shards.length} shard(s) to ${outDir}`);
}

main().catch((err) => {
  console.error(err.message);
  process.exit(3);
});
