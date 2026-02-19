export class DeterministicRNG {
  constructor(seed) {
    this.state = seed >>> 0;
  }

  next() {
    let x = this.state;
    x ^= (x << 13) >>> 0;
    x ^= x >>> 17;
    x ^= (x << 5) >>> 0;
    this.state = x >>> 0;
    return this.state / 4294967296;
  }
}

export function xorshift32(seed) {
  const rng = new DeterministicRNG(seed);
  rng.next();
  return rng.state;
}

export function randomFloat01(state) {
  const rng = new DeterministicRNG(state);
  return { next: xorshift32(state), value: rng.next() };
}
