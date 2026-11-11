import { BPE, UNK, SPACE } from "./bpe.js";
import { clean, isLead, isSyllable, isTail, isVowel, splitSyllable, tailIndex, T_RIEUL, toCompat } from "./hangul.js";

const MODES = ["syl", "jamo"];
const NAME = { syl: "음절 모델", jamo: "자모 모델" };
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const toks = {};
let loaded = false;

// ---------------------------------------------------------------- worker
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let seq = 0;
const pending = new Map();
worker.onmessage = (e) => {
  const m = e.data, p = pending.get(m.id);
  if (!p) return;
  if (m.type === "done") { pending.delete(m.id); p.res(m.result); }
  else if (m.type === "error") { pending.delete(m.id); p.rej(new Error(m.message)); }
  else if (p.on) p.on(m);
};
function call(cmd, args = {}, on) {
  const id = ++seq;
  return new Promise((res, rej) => {
    pending.set(id, { res, rej, on });
    worker.postMessage({ id, cmd, ...args });
  });
}

// ---------------------------------------------------------------- 토큰 자르기
const specimen = $("specimen");

function tokenNode(str, mode, isUnk) {
  const chip = el("span", "chip" + (isUnk ? " unk" : ""));
  if (isUnk) { chip.textContent = "?"; chip.title = "어휘에 없는 글자"; return chip; }
  for (const ch of str) {
    if (ch === SPACE) chip.append(el("span", "sp", "▁"));
    else if (mode === "jamo" && isLead(ch)) chip.append(el("span", "j-l", toCompat(ch)));
    else if (mode === "jamo" && isVowel(ch)) chip.append(el("span", "j-v", toCompat(ch)));
    else if (mode === "jamo" && isTail(ch)) chip.append(el("span", "j-t", toCompat(ch)));
    else chip.append(ch);
  }
  return chip;
}

function renderCuts() {
  const text = clean(specimen.value);
  for (const m of MODES) {
    const ids = toks[m].encode(text);
    const box = $(`chips-${m}`);
    box.replaceChildren(...ids.map((i) => tokenNode(toks[m].vocab[i], m, i === UNK)));
    $(`count-${m}`).textContent = `토큰 ${ids.length}개`;
  }
}

function fitSpecimen() {
  specimen.style.height = "auto";
  specimen.style.height = specimen.scrollHeight + "px";
}

specimen.addEventListener("input", () => { fitSpecimen(); if (toks.syl) renderCuts(); });
specimen.addEventListener("keydown", (e) => { if (e.key === "Enter") e.preventDefault(); });
window.addEventListener("resize", fitSpecimen);

async function initTokenizers() {
  try {
    const loadedToks = await Promise.all(MODES.map((m) => BPE.load(`models/${m}/tokenizer.json`)));
    MODES.forEach((m, i) => (toks[m] = loadedToks[i]));
  } catch {
    $("cuts").hidden = true;
    $("tok-empty").hidden = false;
    $("josa-hint").textContent = $("gen-hint").textContent = $("typing-hint").textContent = "학습된 모델이 아직 없다.";
    return false;
  }
  renderCuts();
  return true;
}

// ---------------------------------------------------------------- 모델 불러오기
async function initLoader() {
  const sizes = await Promise.all(MODES.map(async (m) => {
    try {
      const r = await fetch(`models/${m}/weights.bin`, { method: "HEAD" });
      return +r.headers.get("content-length") || 0;
    } catch { return 0; }
  }));
  const total = sizes.reduce((a, b) => a + b, 0);
  const btn = $("load-btn");
  if (total) btn.textContent = `모델 불러오기 (약 ${Math.round(total / 1e6)}MB)`;
  $("loader").hidden = false;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "불러오는 중";
    const prog = {};
    $("progress").hidden = false;
    try {
      await call("load", { modes: MODES }, (m) => {
        if (m.type !== "progress") return;
        prog[m.mode] = m.p;
        const avg = MODES.reduce((s, k) => s + (prog[k] || 0), 0) / MODES.length;
        $("progress-bar").style.width = `${(avg * 100).toFixed(1)}%`;
      });
      loaded = true;
      $("loader").hidden = true;
      $("gen-btn").disabled = false;
      $("gen-hint").hidden = true;
      $("typing-hint").hidden = true;
      runJosa();
      runSuggest();
    } catch (err) {
      btn.disabled = false;
      btn.textContent = "다시 불러오기";
      $("loader").querySelector("p").textContent = `모델을 받지 못했다: ${err.message}`;
    }
  });
}

// ---------------------------------------------------------------- 치는 도중 단어 맞히기
const typing = $("typing-input");
let sugRun = 0, sugTimer = null;

// 입력을 앞 문맥과 지금 치는 단어로 나눔. 끝이 공백이면 다음 단어를 맞힘.
function splitTyping(v) {
  const raw = v.normalize("NFC").replace(/[^\u0020-\u007e\uac00-\ud7a3\u3131-\u318e]/g, "").replace(/ {2,}/g, " ");
  const body = raw.trim();
  if (/ $/.test(raw) || !body) return { context: body, partial: "" };
  const i = body.lastIndexOf(" ");
  return { context: i < 0 ? "" : body.slice(0, i), partial: body.slice(i + 1) };
}

async function runSuggest() {
  if (!loaded) return;
  const run = ++sugRun;
  const { context, partial } = splitTyping(typing.value);
  const order = ["jamo", "syl"];
  let results;
  try {
    results = await Promise.all(order.map((m) => call("suggest", { mode: m, context, partial })));
  } catch (err) {
    $("sugs").replaceChildren(el("p", "sug-none", `추천하다 멈췄다: ${err.message}`));
    return;
  }
  if (run !== sugRun || results.some((r) => r === null)) return;  // 그사이 입력이 바뀜
  const rows = order.map((m, k) => {
    const row = el("div", "sug-row");
    const chips = el("div", "sug-chips");
    const list = results[k].filter((x, i) => i === 0 || x.p >= 0.01);  // 1% 미만은 안 보여줌
    if (!list.length) chips.append(el("span", "sug-none", "후보 없음"));
    for (const { word, p } of list) {
      const b = el("button", "sug", word);
      b.type = "button";
      b.append(el("span", "pct", `${Math.round(p * 100)}%`));
      b.addEventListener("click", () => {
        const { context: c } = splitTyping(typing.value);
        typing.value = (c ? c + " " : "") + word + " ";
        typing.focus();
        runSuggest();
      });
      chips.append(b);
    }
    row.append(el("span", "who", NAME[m]), chips);
    return row;
  });
  $("sugs").replaceChildren(...rows);
}

typing.addEventListener("input", () => {
  clearTimeout(sugTimer);
  sugTimer = setTimeout(runSuggest, 180);
});

// ---------------------------------------------------------------- 받침-조사
const batchim = (w) => tailIndex(w) !== 0;
const notRieul = (w) => batchim(w) && tailIndex(w) !== T_RIEUL;
const TEMPLATES = [
  ["나는 ", " 좋아한다", ["을", "를"], batchim],
  ["여기에 ", " 있다", ["이", "가"], batchim],
  ["그 ", " 정말 좋다", ["은", "는"], batchim],
  ["우리는 ", " 갔다", ["으로", "로"], notRieul],
];
let tIdx = 0, josaRun = 0, josaTimer = null;

function nounValue() {
  return clean($("noun").value).replace(/ /g, "");
}

function renderSentence() {
  const noun = nounValue();
  const [pre, post, opts, rule] = TEMPLATES[tIdx];
  const s = $("josa-sentence");
  s.replaceChildren();
  const last = Array.from(noun).pop();
  if (!noun || !isSyllable(last)) {
    s.textContent = "";
    $("josa-hint").hidden = false;
    $("josa-hint").textContent = "한글 음절로 끝나는 단어를 넣어야 한다.";
    $("verdicts").replaceChildren();
    return null;
  }
  s.append(pre + noun, el("span", "slot", opts.join("/")), post);
  const right = rule(noun) ? opts[0] : opts[1];
  const t = splitSyllable(last)[2];
  const why = t === 0 ? "받침이 없으니" : `받침 ${toCompat(String.fromCharCode(0x11a7 + t))}이 있으니`;
  const extra = tIdx === 3 && t === T_RIEUL ? " (ㄹ 받침은 예외로 ‘로’)" : "";
  $("josa-hint").hidden = false;
  $("josa-hint").textContent = loaded ? `${why} 정답은 ‘${right}’${extra}` : "모델을 불러오면 결과가 나온다.";
  return { noun, pre, post, opts, right };
}

async function runJosa() {
  const q = renderSentence();
  if (!q || !loaded) return;
  const run = ++josaRun;
  const sentences = q.opts.map((o) => q.pre + q.noun + o + q.post);
  const rows = [];
  for (const m of MODES) {
    const [la, lb] = await call("josa", { mode: m, sentences });
    if (run !== josaRun) return;  // 그사이 입력이 바뀜
    const pA = 1 / (1 + Math.exp(lb - la));
    const pRight = q.right === q.opts[0] ? pA : 1 - pA;
    const row = el("div", "verdict");
    const bar = el("div", "bar");
    const fill = el("i");
    fill.style.width = `${(pA * 100).toFixed(1)}%`;
    bar.append(fill);
    bar.setAttribute("role", "img");
    bar.setAttribute("aria-label", `${q.opts[0]} ${Math.round(pA * 100)}%, ${q.opts[1]} ${Math.round((1 - pA) * 100)}%`);
    const a = el("span", "opt" + (q.right === q.opts[0] ? " correct" : ""), `${q.opts[0]} ${Math.round(pA * 100)}`);
    const b = el("span", "opt right" + (q.right === q.opts[1] ? " correct" : ""), `${Math.round((1 - pA) * 100)} ${q.opts[1]}`);
    const ok = pRight > 0.5;
    row.append(el("span", "who", NAME[m]), a, bar, b, el("span", "mark" + (ok ? "" : " miss"), ok ? "맞힘" : "틀림"));
    rows.push(row);
  }
  $("verdicts").replaceChildren(...rows);
}

function scheduleJosa() {
  renderSentence();
  clearTimeout(josaTimer);
  josaTimer = setTimeout(runJosa, 250);
}

$("noun").addEventListener("input", scheduleJosa);
document.querySelectorAll(".pair").forEach((b) => b.addEventListener("click", () => {
  tIdx = +b.dataset.t;
  document.querySelectorAll(".pair").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
  scheduleJosa();
}));

$("nonce-btn").addEventListener("click", () => {
  // 음절 어휘의 기본 글자는 빈도순이라 앞쪽이 자주 쓰는 음절. 그걸 섞어서 가짜 단어를 만듦.
  const top = (toks.syl ? toks.syl.alphabet : Array.from("가나다라마바사아자차카타파하강산물불길")).filter(isSyllable).slice(0, 300);
  const groups = [
    top.filter((c) => splitSyllable(c)[2] === 0),
    top.filter((c) => splitSyllable(c)[2] === T_RIEUL),
    top.filter((c) => ![0, T_RIEUL].includes(splitSyllable(c)[2])),
  ].filter((g) => g.length);
  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
  $("noun").value = pick(top) + pick(pick(groups));
  scheduleJosa();
});

// ---------------------------------------------------------------- 이어 쓰기
let genMode = "both", generating = false;
document.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => {
  genMode = b.dataset.m;
  document.querySelectorAll(".seg-btn").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
}));

$("gen-btn").addEventListener("click", async () => {
  if (generating) { worker.postMessage({ cmd: "stop" }); return; }
  const prompt = clean($("prompt").value);
  const modes = genMode === "both" ? ["jamo", "syl"] : [genMode];
  const outs = $("outs");
  outs.className = "outs" + (modes.length === 2 ? " two" : "");
  outs.replaceChildren();
  const seed = Math.floor(Math.random() * 1e9);
  const jobs = modes.map((m) => {
    const box = el("div", "out");
    const p = el("p");
    const given = el("span", "given", prompt);
    const rest = el("span");
    p.append(given, rest);
    box.append(el("h3", null, NAME[m]), p);
    outs.append(box);
    return call("generate", { mode: m, prompt, len: 120, temp: 0.8, topk: 40, seed }, (msg) => {
      if (msg.type !== "text") return;
      const t = msg.text;
      if (t.startsWith(prompt)) { given.textContent = prompt; rest.textContent = t.slice(prompt.length); }
      else { given.textContent = ""; rest.textContent = t; }
    });
  });
  generating = true;
  $("gen-btn").textContent = "멈추기";
  try { await Promise.all(jobs); }
  catch (err) { outs.append(el("p", "hint", `생성하다 멈췄다: ${err.message}`)); }
  generating = false;
  $("gen-btn").textContent = "이어 쓰기";
});

// ---------------------------------------------------------------- 결과
const SVG = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs, text) => {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};
const LINE = { jamo: { color: "#1d232b", label: "자모" }, syl: { color: "#8c96a1", label: "음절" } };

function drawChart(rs) {
  const series = Object.entries(rs)
    .filter(([, r]) => r.curve && r.curve.length > 1)
    .map(([m, r]) => [m, r.curve.filter((c) => c[0] > 0).map((c) => [c[0], c[2]])]);
  if (!series.length) return;
  const fig = $("chart");
  const W = Math.max(300, Math.round(fig.clientWidth || 700)), H = W < 500 ? 230 : 300;
  const L = 44, R = 46, T = 14, B = 34;
  const xs = series.flatMap(([, s]) => s.map((p) => p[0]));
  const ys = series.flatMap(([, s]) => s.map((p) => p[1]));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = (y1 - y0) * 0.08 || 0.1;
  y0 -= pad; y1 += pad;
  const X = (v) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - R);
  const Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "학습 스텝별 검증 bpc" });
  const step = niceStep((y1 - y0) / 4);
  for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) {
    svg.append(svgEl("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: "#d3d9e1" }));
    svg.append(svgEl("text", { x: L - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 12, fill: "#4c5560" }, v.toFixed(2)));
  }
  const xstep = niceStep((x1 - x0) / 5);
  for (let v = Math.ceil(x0 / xstep) * xstep; v <= x1; v += xstep) {
    svg.append(svgEl("text", { x: X(v), y: H - 12, "text-anchor": "middle", "font-size": 12, fill: "#4c5560" }, v.toLocaleString()));
  }
  const labels = [];
  for (const [m, s] of series) {
    svg.append(svgEl("polyline", {
      points: s.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" "),
      fill: "none", stroke: LINE[m].color, "stroke-width": 2.5, "stroke-linejoin": "round",
    }));
    const last = s[s.length - 1];
    labels.push({ m, x: X(last[0]) + 8, y: Y(last[1]) + 4 });
  }
  labels.sort((a, b) => a.y - b.y);  // 선 끝 이름표가 겹치지 않게 위아래로 벌림
  for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 16);
  for (const { m, x, y } of labels) {
    svg.append(svgEl("text", { x, y, "font-size": 14, fill: LINE[m].color, "font-weight": 600 }, LINE[m].label));
  }
  fig.replaceChildren(svg, el("figcaption", null, "학습 스텝(가로)에 따른 검증 bpc(세로)"));
}

function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw));
  const n = raw / p;
  return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * p;
}

function drawTable(rs) {
  const ms = Object.keys(rs);
  const pct = (v) => `${(v * 100).toFixed(0)}%`;
  const rows = [
    ["bpc", (r) => r.bpc, (v) => v.toFixed(4), "low"],
    ["글자당 토큰 수", (r) => r.tokens_per_char, (v) => v.toFixed(3)],
    ["어휘에 없는 토큰 비율", (r) => r.unk_rate, (v) => `${(v * 100).toFixed(3)}%`, "low"],
  ];
  const keys = Object.keys(rs[ms[0]].josa_real || {});
  for (const k of keys) rows.push([`조사 ${k}, 실제 명사`, (r) => r.josa_real && r.josa_real[k], pct, "high"]);
  for (const k of keys) rows.push([`조사 ${k}, 가짜 명사`, (r) => r.josa_nonce && r.josa_nonce[k], pct, "high"]);
  const t = $("table");
  const head = el("thead");
  const hr = el("tr");
  hr.append(el("th", null, ""), ...ms.map((m) => el("th", null, LINE[m].label)));
  head.append(hr);
  const body = el("tbody");
  for (const [name, get, fmt, better] of rows) {
    const vals = ms.map((m) => get(rs[m]));
    if (vals.every((v) => v === undefined)) continue;
    const ok = vals.filter((v) => v !== undefined);
    const best = better === "low" ? Math.min(...ok) : better === "high" ? Math.max(...ok) : null;
    const tr = el("tr");
    tr.append(el("td", null, name));
    vals.forEach((v) => tr.append(el("td", best !== null && v === best && ms.length > 1 && new Set(ok).size > 1 ? "win" : "", v === undefined ? "-" : fmt(v))));
    body.append(tr);
  }
  t.replaceChildren(head, body);
}

async function initResults() {
  const rs = {};
  for (const m of ["jamo", "syl"]) {
    try {
      const r = await fetch(`models/${m}/results.json`);
      if (r.ok) rs[m] = await r.json();
    } catch { /* 없음 */ }
  }
  if (!Object.keys(rs).length) return;
  $("results").hidden = false;
  drawChart(rs);
  drawTable(rs);
  let t;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => drawChart(rs), 200); });
}

// ---------------------------------------------------------------- 시작
fitSpecimen();
renderSentence();
initTokenizers().then((ok) => { if (ok) initLoader(); });
initResults();
