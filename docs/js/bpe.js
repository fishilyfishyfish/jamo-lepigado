// BPE 인코더/디코더. jamo/bpe.py와 똑같은 결과를 내야 함 (tests/test_js.py로 확인).
import { compose, decompose } from "./hangul.js";

export const SPACE = "▁";
export const EOS = 0, UNK = 1;

export class BPE {
  constructor({ mode, specials, alphabet, merges }) {
    this.mode = mode;
    this.alphabet = alphabet;
    this.merges = merges;
    this.base = specials.length + alphabet.length;
    this.sym2id = new Map(alphabet.map((ch, i) => [ch, specials.length + i]));
    this.ranks = new Map(merges.map(([a, b], r) => [a * 65536 + b, r]));
    this.vocab = [...specials, ...alphabet];
    for (const [a, b] of merges) this.vocab.push(this.vocab[a] + this.vocab[b]);
    this.cache = new Map();
  }

  static async load(url) {
    return new BPE(await (await fetch(url)).json());
  }

  get size() { return this.vocab.length; }

  words(text) {
    if (this.mode === "jamo") text = decompose(text);
    return text.split(" ").filter((w) => w).map((w) => SPACE + w);
  }

  encodeWord(word) {
    const hit = this.cache.get(word);
    if (hit) return hit;
    let ids = Array.from(word, (ch) => this.sym2id.get(ch) ?? UNK);
    while (ids.length > 1) {
      let bestRank = -1, bestPos = -1;
      for (let i = 0; i + 1 < ids.length; i++) {
        const r = this.ranks.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && (bestRank < 0 || r < bestRank)) { bestRank = r; bestPos = i; }
      }
      if (bestRank < 0) break;
      const a = ids[bestPos], b = ids[bestPos + 1], nw = this.base + bestRank;
      const out = [];
      for (let i = 0; i < ids.length; ) {
        if (i + 1 < ids.length && ids[i] === a && ids[i + 1] === b) { out.push(nw); i += 2; }
        else { out.push(ids[i]); i++; }
      }
      ids = out;
    }
    if (this.cache.size < 100000) this.cache.set(word, ids);
    return ids;
  }

  encode(text) {
    const out = [];
    for (const w of this.words(text)) out.push(...this.encodeWord(w));
    return out;
  }

  decode(ids) {
    let s = ids.filter((i) => i !== EOS).map((i) => (i === UNK ? "�" : this.vocab[i])).join("");
    s = s.split(SPACE).join(" ");
    if (s.startsWith(" ")) s = s.slice(1);
    return this.mode === "jamo" ? compose(s) : s;
  }
}
