import { TensorShell } from "../shell/tensor-shell.js";

async function run() {
  const css = await fetch("./governance.css").then((r) => r.text());
  const root = css.match(/:root\s*{([\s\S]*?)}/)?.[1] || "";
  const read = (k) => Number(root.match(new RegExp(`--${k}:\\s*([^;]+);`))?.[1]?.replace(/"/g, "").trim());

  const shell = new TensorShell({
    hiddenDim: read("hidden-dim"),
    numLayers: read("num-layers"),
    numAttentionHeads: read("num-attention-heads"),
    maxBatchSize: 4,
    maxSeqLen: read("max-seq-len"),
  });

  await shell.initialize();
  document.querySelector("#out").textContent = "Shell initialized";
}

run().catch((err) => {
  document.querySelector("#out").textContent = String(err);
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./service-worker.js").catch((err) => {
      console.warn("Service worker registration failed", err);
    });
  });
}
