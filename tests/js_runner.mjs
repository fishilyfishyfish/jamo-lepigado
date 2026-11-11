// tests/test_js.py가 부르는 Node 스크립트. 파이썬 구현과 결과를 비교하려고 JSON으로 출력함.
//   node js_runner.mjs tok <tokenizer.json> <lines.json>
//   node js_runner.mjs gpt <model_dir> <ids.json>
import { readFileSync } from "node:fs";
import { BPE } from "../docs/js/bpe.js";
import { GPT } from "../docs/js/gpt.js";
import { clean, compose, decompose } from "../docs/js/hangul.js";

const [cmd, a, b] = process.argv.slice(2);
const json = (p) => JSON.parse(readFileSync(p, "utf8"));

if (cmd === "tok") {
  const tok = new BPE(json(a));
  const lines = json(b);
  console.log(JSON.stringify(lines.map((s) => {
    const ids = tok.encode(s);
    return { ids, dec: tok.decode(ids), clean: clean(s), rt: compose(decompose(s)) === s };
  })));
} else if (cmd === "gpt") {
  const buf = readFileSync(`${a}/weights.bin`);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const m = new GPT(json(`${a}/model.json`), ab);
  const ids = json(b);
  const cache = m.newCache();
  const logits = [];
  for (const t of ids) logits.push(Array.from(m.step(t, cache)));
  console.log(JSON.stringify({ logits, logprob: m.logProb(ids) }));
}
