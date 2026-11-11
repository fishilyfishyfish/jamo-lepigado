// 모델 추론은 화면이 버벅이지 않게 워커에서 돌림.
import { BPE, EOS, SPACE } from "./bpe.js";
import { GPT, rng, sample } from "./gpt.js";
import { typingVariants } from "./hangul.js";
import { ContextCache, suggestWords } from "./suggest.js";

const models = {};
const REP_PENALTY = 1.3;
const contexts = {};      // 모드별 앞 문맥 캐시 (단어 추천용)
const latestSuggest = {}; // 모드별 가장 최근 추천 요청 id
let stopped = false;
const url = (p) => new URL(p, import.meta.url).href;

function get(mode, id) {
  if (!models[mode]) {
    models[mode] = (async () => {
      const tok = await BPE.load(url(`../models/${mode}/tokenizer.json`));
      const gpt = await GPT.load(url(`../models/${mode}`), (p) => postMessage({ id, type: "progress", mode, p }));
      return { tok, gpt };
    })();
    models[mode].catch(() => delete models[mode]);
  }
  return models[mode];
}

async function generate(id, { mode, prompt, len, temp, topk, seed }) {
  const { tok, gpt } = await get(mode, id);
  const rand = rng(seed);
  const block = gpt.cfg.block;
  const ids = [EOS, ...tok.encode(prompt)].slice(-block);
  let cache = gpt.newCache();
  let logits;
  for (const t of ids) logits = gpt.step(t, cache);
  for (let i = 0; i < len && !stopped; i++) {
    // 작은 모델은 같은 말을 반복하기 쉬워서, 최근 64토큰 안에 나온 토큰은 확률을 깎음
    for (const t of new Set(ids.slice(-64))) {
      if (t !== EOS) logits[t] = logits[t] > 0 ? logits[t] / REP_PENALTY : logits[t] * REP_PENALTY;
    }
    const next = sample(logits, temp, topk, rand);
    if (next === EOS) break;
    ids.push(next);
    postMessage({ id, type: "text", text: tok.decode(ids) });
    if (cache.len >= block) {
      // 문맥이 꽉 차면 뒤쪽 절반만 남기고 캐시를 다시 만듦
      cache = gpt.newCache();
      for (const t of ids.slice(-Math.floor(block / 2))) logits = gpt.step(t, cache);
    } else {
      logits = gpt.step(next, cache);
    }
    await new Promise((r) => setTimeout(r, 0));  // 멈추기 메시지를 받을 틈
  }
}

self.onmessage = async (e) => {
  const { id, cmd, ...args } = e.data;
  if (cmd === "stop") { stopped = true; return; }
  try {
    let result = null;
    if (cmd === "load") {
      await Promise.all(args.modes.map((m) => get(m, id)));
    } else if (cmd === "josa") {
      const { tok, gpt } = await get(args.mode, id);
      result = args.sentences.map((s) => gpt.logProb([EOS, ...tok.encode(s)]));
    } else if (cmd === "suggest") {
      latestSuggest[args.mode] = id;
      await new Promise((r) => setTimeout(r, 0));  // 그사이 들어온 새 요청이 있으면 이건 건너뜀
      if (latestSuggest[args.mode] === id) {
        const { tok, gpt } = await get(args.mode, id);
        const cc = (contexts[args.mode] ||= new ContextCache(gpt));
        cc.update([EOS, ...tok.encode(args.context)]);
        result = suggestWords(tok, gpt, cc, typingVariants(args.partial).map((p) => SPACE + p));
      }
    } else if (cmd === "generate") {
      stopped = false;
      await generate(id, args);
    }
    postMessage({ id, type: "done", result });
  } catch (err) {
    postMessage({ id, type: "error", message: String(err && err.message || err) });
  }
};
