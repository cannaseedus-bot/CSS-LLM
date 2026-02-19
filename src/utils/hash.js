export function hashUint32(values) {
  let h = 2166136261 >>> 0;
  for (const value of values) {
    h ^= value >>> 0;
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}
