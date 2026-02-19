import { DeterministicRNG } from "./deterministic_rng.js";

export function sampleTopP(logits, temperature, topP, seed) {
  const rng = new DeterministicRNG(seed);
  const scaled = logits.map((value) => value / temperature);
  const maxLogit = Math.max(...scaled);
  const exps = scaled.map((value) => Math.exp(value - maxLogit));

  const total = exps.reduce((acc, value) => acc + value, 0);
  const probs = exps.map((value) => value / total);

  const sorted = probs
    .map((probability, index) => ({ probability, index }))
    .sort((a, b) => (b.probability - a.probability) || (a.index - b.index));

  let cumulative = 0;
  const filtered = [];
  for (const pair of sorted) {
    cumulative += pair.probability;
    filtered.push(pair);
    if (cumulative >= topP) break;
  }

  const mass = filtered.reduce((acc, pair) => acc + pair.probability, 0);
  const r = rng.next();
  let running = 0;

  for (const pair of filtered) {
    running += pair.probability / mass;
    if (r < running) return pair.index;
  }

  return filtered[0].index;
}
