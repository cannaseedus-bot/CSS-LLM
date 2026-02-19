import fs from "node:fs/promises";

function normalizeMerges(rawMerges) {
  return rawMerges
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith("#"))
    .map((line) => {
      const [left, right] = line.split(/\s+/);
      return [left, right];
    });
}

export async function loadTokenizer(vocabPath, mergesPath) {
  const [vocabRaw, mergesRaw] = await Promise.all([
    fs.readFile(vocabPath, "utf8"),
    fs.readFile(mergesPath, "utf8"),
  ]);

  return new DeterministicTokenizer(JSON.parse(vocabRaw), normalizeMerges(mergesRaw));
}

export class DeterministicTokenizer {
  constructor(vocab, merges) {
    this.vocab = vocab;
    this.merges = merges;
    this.unk = vocab["<unk>"] ?? 0;
  }

  tokenize(text) {
    const tokens = Array.from(text);

    for (const [left, right] of this.merges) {
      for (let i = 0; i < tokens.length - 1; i += 1) {
        if (tokens[i] === left && tokens[i + 1] === right) {
          tokens.splice(i, 2, `${left}${right}`);
        }
      }
    }

    return tokens.map((token) => this.vocab[token] ?? this.unk);
  }
}

export function encodeAscii(text, vocabSize = 32000) {
  return Array.from(text).map((ch) => ch.charCodeAt(0) % vocabSize);
}
