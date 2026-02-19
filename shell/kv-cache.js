export class KVCache {
  constructor(maxEntries = 4096) {
    this.maxEntries = maxEntries;
    this.entries = new Map();
  }

  get(key) {
    const item = this.entries.get(key);
    if (!item) return null;
    item.lastAccess = Date.now();
    return item.value;
  }

  set(key, value) {
    if (this.entries.size >= this.maxEntries) this.evictOldest();
    this.entries.set(key, { value, lastAccess: Date.now() });
  }

  evictOldest() {
    let oldestKey = null;
    let oldestTs = Infinity;

    for (const [key, item] of this.entries) {
      if (item.lastAccess < oldestTs) {
        oldestTs = item.lastAccess;
        oldestKey = key;
      }
    }

    if (oldestKey) this.entries.delete(oldestKey);
  }
}
