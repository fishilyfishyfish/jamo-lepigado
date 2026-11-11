// 치는 도중의 단어 맞히기.
// 지금까지 친 것을 자모로 풀어서(서우 -> ㅅㅓㅇㅜ), 자모로 풀었을 때 앞부분이 맞는 토큰만 남기고 단어 끝까지 이어 본다.
// 음절 모델도 토큰을 자모로 풀어서 비교하니까 두 모델이 같은 규칙으로 겨룬다.
import { EOS, SPACE } from "./bpe.js";
import { compose, decompose } from "./hangul.js";

const PUNCT = /^[.,!?;:)\]}'"]/;

function logSoftmax(lg) {
  let mx = -Infinity;
  for (let i = 0; i < lg.length; i++) if (lg[i] > mx) mx = lg[i];
  let z = 0;
  for (let i = 0; i < lg.length; i++) z += Math.exp(lg[i] - mx);
  const lz = mx + Math.log(z), out = new Float32Array(lg.length);
  for (let i = 0; i < lg.length; i++) out[i] = lg[i] - lz;
  return out;
}

function logSumExp(xs) {
  let mx = -Infinity;
  for (const x of xs) if (x > mx) mx = x;
  if (mx === -Infinity) return -Infinity;
  let z = 0;
  for (const x of xs) z += Math.exp(x - mx);
  return mx + Math.log(z);
}

const tables = new WeakMap();
function table(tok) {
  let t = tables.get(tok);
  if (!t) {
    const dec = tok.vocab.map((s, i) => (i < 2 ? "" : decompose(s)));  // 0, 1은 특수 토큰
    const start = dec.map((s) => s.startsWith(SPACE));                  // 새 단어를 여는 토큰
    const ender = dec.map((s, i) => i === EOS || start[i] || PUNCT.test(s));  // 단어를 끝내는 토큰
    t = { dec, start, ender };
    tables.set(tok, t);
  }
  return t;
}

const fits = (s, p) => s.startsWith(p) || p.startsWith(s);

// 앞 문맥의 K/V 캐시. 단어를 치는 동안은 앞 문맥이 그대로라 다시 계산하지 않음.
export class ContextCache {
  constructor(gpt) { this.gpt = gpt; this.ids = []; this.cache = gpt.newCache(); this.logits = null; }
  update(ids) {
    const g = this.gpt, max = g.cfg.block - 12;
    if (ids.length > max) ids = [EOS, ...ids.slice(-(max - 1))];
    const same = this.logits && ids.length >= this.ids.length && this.ids.every((t, i) => t === ids[i]);
    if (!same) { this.ids = []; this.cache = g.newCache(); }
    for (let i = this.ids.length; i < ids.length; i++) this.logits = Float32Array.from(g.step(ids[i], this.cache));
    this.ids = ids.slice();
  }
}

// prefixes: 자모로 푼 단어 앞부분 후보들 (SPACE로 시작). 단어 후보 n개와 상대 확률을 돌려줌.
export function suggestWords(tok, gpt, ctx, prefixes, { n = 3, branch = 6, steps = 4 } = {}) {
  const { dec, start, ender } = table(tok);
  const V = dec.length;
  const lp0 = logSoftmax(ctx.logits);
  const firsts = [];
  for (let t = 2; t < V; t++) {
    if (!start[t]) continue;
    const p = prefixes.find((q) => fits(dec[t], q));
    if (p !== undefined) firsts.push({ t, p, lp: lp0[t] });
  }
  firsts.sort((a, b) => b.lp - a.lp);

  const found = new Map();
  for (const f of firsts.slice(0, branch)) {
    const cache = gpt.cloneCache(ctx.cache, steps + 2);
    let word = dec[f.t], score = f.lp, ended = false;
    let rest = f.p.length > word.length ? f.p.slice(word.length) : "";
    let lp = logSoftmax(gpt.step(f.t, cache));
    for (let k = 0; k < steps && Number.isFinite(score); k++) {
      if (rest) {  // 친 것을 아직 다 못 덮음: 이어지는 토큰 중 가장 그럴듯한 것
        let best = -1;
        for (let t = 2; t < V; t++) {
          if (!start[t] && dec[t] && fits(dec[t], rest) && (best < 0 || lp[t] > lp[best])) best = t;
        }
        if (best < 0) { score = -Infinity; break; }
        word += dec[best];
        score += lp[best];
        rest = rest.length > dec[best].length ? rest.slice(dec[best].length) : "";
        lp = logSoftmax(gpt.step(best, cache));
        continue;
      }
      // 친 것은 다 덮었음: 여기서 단어를 끝낼지, 한 토큰 더 이을지
      const ends = [];
      let best = -1;
      for (let t = 0; t < V; t++) {
        if (ender[t]) ends.push(lp[t]);
        else if (t > 1 && dec[t] && (best < 0 || lp[t] > lp[best])) best = t;
      }
      const endLp = logSumExp(ends);
      if (best < 0 || endLp >= lp[best]) { score += endLp; ended = true; break; }
      word += dec[best];
      score += lp[best];
      lp = logSoftmax(gpt.step(best, cache));
    }
    if (!Number.isFinite(score) || rest) continue;
    if (!ended) score += logSumExp(Array.from(lp).filter((_, t) => ender[t]));
    const w = compose(word.split(SPACE).join(""));
    if (!found.has(w) || found.get(w) < score) found.set(w, score);
  }
  const list = [...found].map(([word, score]) => ({ word, score }))
    .sort((a, b) => b.score - a.score).slice(0, n);
  const z = logSumExp(list.map((x) => x.score));
  return list.map((x) => ({ word: x.word, p: Math.exp(x.score - z) }));
}
