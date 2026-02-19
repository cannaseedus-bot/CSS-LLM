const CACHE_NAME = "css-mini-v1";
const ASSETS = [
  "/demo/index.html",
  "/demo/main.js",
  "/demo/style.css",
  "/governance/model.css",
  "/models/mini-64m-int4.scx.base64.json",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key)))),
  );
});

self.addEventListener("fetch", (event) => {
  event.respondWith(caches.match(event.request).then((cached) => cached || fetch(event.request)));
});
