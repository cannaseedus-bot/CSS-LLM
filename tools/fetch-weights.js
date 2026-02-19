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
          fs.unlinkSync(outPath);
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
  const url = process.argv[2];
  const out = process.argv[3] || "weights/model.bin";

  if (!url) {
    console.error("Usage: node tools/fetch-weights.js <https_url> [output_path]");
    process.exit(1);
  }

  if (!url.startsWith("https://")) {
    console.error("Only https:// URLs are allowed");
    process.exit(2);
  }

  fs.mkdirSync(path.dirname(out), { recursive: true });
  await download(url, out);
  console.log(`Downloaded weights to: ${out}`);
}

main().catch((err) => {
  console.error(err.message);
  process.exit(3);
});
