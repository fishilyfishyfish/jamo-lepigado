#!/usr/bin/env python3
"""
jamo-lepigado: 자모 BPE vs 음절 BPE로 같은 한국어 미니 GPT를 학습시켜 비교.

  python main.py corpus                 # 한국어 위키 5천만 자 -> corpus.txt
  python main.py all --mode jamo        # tok -> prep -> train -> eval -> export
  python main.py all --mode syl
  python main.py compare                # 두 결과를 마크다운 표로
  python main.py gen --mode jamo --prompt "세종대왕은"

단계별: tok / prep / train / eval / export
작업 파일은 out/, 결과 수치는 results/, 웹 데모용 모델은 docs/models/ 에 저장됨.
"""
import argparse
import contextlib
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np

from jamo.bpe import BPE, EOS, UNK
from jamo.data import download_wiki, read_split
from jamo.hangul import to_compat


def paths(a):
    o = a.out
    return {
        "tok": f"{o}/tok_{a.mode}.json",
        "data": f"{o}/data_{a.mode}",
        "ckpt": f"{o}/ckpt_{a.mode}.pt",
        "result": f"results/{a.mode}.json",
        "web": f"docs/models/{a.mode}",
    }


def update_result(a, **kv):
    os.makedirs("results", exist_ok=True)
    p = paths(a)["result"]
    r = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"mode": a.mode}
    r.update(kv)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- corpus
def cmd_corpus(a):
    download_wiki(a.corpus, a.max_chars)


# ---------------------------------------------------------------- tokenizer
def cmd_tok(a):
    train, _ = read_split(a.corpus)
    sub, n = [], 0
    for s in train:  # 토크나이저는 앞쪽 tok_chars자만으로 학습 (시간/메모리 절약)
        sub.append(s)
        n += len(s)
        if n >= a.tok_chars:
            break
    print(f"[tok:{a.mode}] {len(sub):,}줄 {n:,}자로 학습")
    t0 = time.time()
    tok = BPE.train(sub, a.mode, a.vocab, a.alphabet)
    os.makedirs(a.out, exist_ok=True)
    tok.save(paths(a)["tok"])
    sample = "낮과 낯은 소리가 같다"
    print(f"[tok:{a.mode}] 저장 {paths(a)['tok']} ({time.time() - t0:.0f}초)")
    print("  예시:", " | ".join(to_compat(t) for t in tok.tokens(sample)))
    print("  복원:", tok.decode(tok.encode(sample)))


# ---------------------------------------------------------------- prep
_TOK = None


def _init_worker(path):
    global _TOK
    _TOK = BPE.load(path)


def _encode_chunk(lines):
    out = []
    for s in lines:
        out.extend(_TOK.encode(s))
        out.append(EOS)
    return np.array(out, dtype=np.uint16)


def cmd_prep(a):
    p = paths(a)
    tok = BPE.load(p["tok"])
    train, val = read_split(a.corpus)
    os.makedirs(p["data"], exist_ok=True)
    meta = {"vocab": len(tok)}
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    with ctx.Pool(os.cpu_count(), initializer=_init_worker, initargs=(p["tok"],)) as pool:
        for name, lines in (("train", train), ("val", val)):
            t0 = time.time()
            chunks = [lines[i:i + 20000] for i in range(0, len(lines), 20000)]
            parts = []
            for k, arr in enumerate(pool.imap(_encode_chunk, chunks)):
                parts.append(arr)
                if (k + 1) % 20 == 0:
                    print(f"  [prep:{a.mode}] {name} {k + 1}/{len(chunks)} 묶음", flush=True)
            arr = np.concatenate(parts)
            arr.tofile(f"{p['data']}/{name}.bin")
            chars = sum(len(s) + 1 for s in lines)  # +1은 줄바꿈(EOS) 몫
            m = meta[name] = {"tokens": int(len(arr)), "chars": chars, "unk": int((arr == UNK).sum())}
            print(f"[prep:{a.mode}] {name}: {chars:,}자 -> {m['tokens']:,}토큰 "
                  f"(글자당 {m['tokens'] / chars:.3f}토큰, UNK {m['unk'] / m['tokens']:.4%}) "
                  f"{time.time() - t0:.0f}초", flush=True)
    with open(f"{p['data']}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    update_result(a, tokens_per_char=meta["val"]["tokens"] / meta["val"]["chars"],
                  unk_rate=meta["val"]["unk"] / meta["val"]["tokens"], vocab=len(tok))


# ---------------------------------------------------------------- train
def get_batch(data, B, T, dev):
    import torch
    ix = np.random.randint(0, len(data) - T - 1, B)
    x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)


def cmd_train(a):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from jamo.model import GPT, device

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    p = paths(a)
    meta = json.load(open(f"{p['data']}/meta.json"))
    tr = np.memmap(f"{p['data']}/train.bin", dtype=np.uint16, mode="r")
    va = np.memmap(f"{p['data']}/val.bin", dtype=np.uint16, mode="r")
    dev = device()
    cfg = dict(vocab=meta["vocab"], block=a.block, n_layer=a.layers, n_head=a.heads,
               n_embd=a.embd, dropout=a.dropout)
    model = GPT(cfg).to(dev)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"[train:{a.mode}] device {dev}, 파라미터 {n_params / 1e6:.1f}M, "
          f"{a.steps}스텝 x {a.batch * a.block:,}토큰")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    amp = dev == "cuda"  # T4는 bf16이 안 돼서 fp16 + GradScaler
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if amp else contextlib.nullcontext
    tpc = meta["val"]["tokens"] / meta["val"]["chars"]
    warm = max(1, a.steps // 20)
    curve, t0 = [], time.time()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            with torch.no_grad(), ctx():
                ls = [F.cross_entropy(model(x).flatten(0, 1).float(), y.flatten()).item()
                      for x, y in (get_batch(va, a.batch, a.block, dev) for _ in range(20))]
            model.train()
            loss = sum(ls) / len(ls)
            b = loss * tpc / math.log(2)
            curve.append([step, round(loss, 4), round(b, 4)])
            print(f"[train:{a.mode}] step {step:6d}  val loss {loss:.3f}  bpc {b:.3f}  "
                  f"{time.time() - t0:.0f}초", flush=True)
        if step == a.steps:
            break
        lr = a.lr * min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(tr, a.batch, a.block, dev)
        with ctx():
            loss = F.cross_entropy(model(x).flatten(0, 1).float(), y.flatten())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    torch.save({"cfg": cfg, "model": model.state_dict()}, p["ckpt"])
    update_result(a, config=cfg, params=n_params, steps=a.steps, batch_tokens=a.batch * a.block,
                  train_seconds=round(time.time() - t0), curve=curve, device=dev)
    print(f"[train:{a.mode}] 저장 {p['ckpt']}")


# ---------------------------------------------------------------- eval
PROMPTS = ["세종대왕은", "대한민국의 수도는", "고양이는"]


def cmd_eval(a):
    from jamo.evaluate import NOUNS, bpc, generate, josa_accuracy, nonce_nouns
    from jamo.model import device, load_model

    p = paths(a)
    model, tok = load_model(p["ckpt"], device()), BPE.load(p["tok"])
    meta = json.load(open(f"{p['data']}/meta.json"))
    va = np.memmap(f"{p['data']}/val.bin", dtype=np.uint16, mode="r")
    tpc = meta["val"]["tokens"] / meta["val"]["chars"]
    b = bpc(model, va, tpc, a.eval_tokens)
    train, _ = read_split(a.corpus)
    fake = nonce_nouns(train, seed=a.seed)
    real_acc, fake_acc = josa_accuracy(model, tok, NOUNS), josa_accuracy(model, tok, fake)
    samples = {q: generate(model, tok, q, 60, seed=a.seed) for q in PROMPTS}
    update_result(a, bpc=b, josa_real=real_acc, josa_nonce=fake_acc, nonce_examples=fake[:12],
                  samples=samples)
    print(f"[eval:{a.mode}] bpc {b:.4f}")
    print(f"  조사 정답률 실제 명사 {len(NOUNS)}개: " + "  ".join(f"{k} {v:.0%}" for k, v in real_acc.items()))
    print(f"  조사 정답률 가짜 명사 {len(fake)}개: " + "  ".join(f"{k} {v:.0%}" for k, v in fake_acc.items()))
    print(f"  가짜 명사 예시: {' '.join(fake[:12])}")
    for q, s in samples.items():
        print(f"  생성> {s}")


# ---------------------------------------------------------------- export
def cmd_export(a):
    import torch
    from jamo.evaluate import bpc
    from jamo.export import export_web
    from jamo.model import GPT, device, load_model

    p = paths(a)
    dev = device()
    model = load_model(p["ckpt"], dev)
    deq, size = export_web(model, p["tok"], p["web"])
    qmodel = GPT(model.c).to(dev).eval()
    qmodel.load_state_dict({k: torch.from_numpy(v) for k, v in deq.items()})
    meta = json.load(open(f"{p['data']}/meta.json"))
    va = np.memmap(f"{p['data']}/val.bin", dtype=np.uint16, mode="r")
    tpc = meta["val"]["tokens"] / meta["val"]["chars"]
    b_f, b_q = bpc(model, va, tpc, 200_000), bpc(qmodel, va, tpc, 200_000)
    update_result(a, web_bytes=size, bpc_fp32_200k=b_f, bpc_int8_200k=b_q)
    with open(p["result"], encoding="utf-8") as f:
        r = json.load(f)
    with open(f"{p['web']}/results.json", "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False)
    print(f"[export:{a.mode}] {p['web']}/ ({size / 1e6:.1f}MB)  bpc fp32 {b_f:.4f} -> int8 {b_q:.4f}")


# ---------------------------------------------------------------- compare / gen
def cmd_compare(a):
    rs = {}
    for m in ("jamo", "syl"):
        if os.path.exists(f"results/{m}.json"):
            rs[m] = json.load(open(f"results/{m}.json", encoding="utf-8"))
    if not rs:
        print("results/ 가 비어 있음")
        return
    rows = [("bpc (낮을수록 좋음)", lambda r: f"{r['bpc']:.4f}"),
            ("글자당 토큰", lambda r: f"{r['tokens_per_char']:.3f}"),
            ("UNK 비율", lambda r: f"{r['unk_rate']:.4%}")]
    keys = list(next(iter(rs.values())).get("josa_real", {}))
    rows += [(f"조사 {k} 실제 명사", lambda r, k=k: f"{r['josa_real'][k]:.0%}") for k in keys]
    rows += [(f"조사 {k} 가짜 명사", lambda r, k=k: f"{r['josa_nonce'][k]:.0%}") for k in keys]
    print("| | " + " | ".join(rs) + " |")
    print("|---|" + "---|" * len(rs))
    for name, f in rows:
        cells = []
        for r in rs.values():
            try:
                cells.append(f(r))
            except KeyError:
                cells.append("-")
        print(f"| {name} | " + " | ".join(cells) + " |")


def cmd_gen(a):
    from jamo.evaluate import generate
    from jamo.model import device, load_model
    p = paths(a)
    model, tok = load_model(p["ckpt"], device()), BPE.load(p["tok"])
    print(generate(model, tok, a.prompt, a.len, a.temp, a.topk))


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["corpus", "tok", "prep", "train", "eval", "export", "all", "compare", "gen"])
    ap.add_argument("--mode", choices=["jamo", "syl"], default="jamo")
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--out", default="out")
    ap.add_argument("--max_chars", type=int, default=50_000_000)
    ap.add_argument("--tok_chars", type=int, default=10_000_000, help="토크나이저 학습에 쓸 글자 수")
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--alphabet", type=int, default=3000, help="기본 글자 수 상한 (나머지는 <unk>)")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_tokens", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--len", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=50)
    a = ap.parse_args(argv)
    steps = {"corpus": [cmd_corpus], "tok": [cmd_tok], "prep": [cmd_prep], "train": [cmd_train],
             "eval": [cmd_eval], "export": [cmd_export], "compare": [cmd_compare], "gen": [cmd_gen],
             "all": [cmd_tok, cmd_prep, cmd_train, cmd_eval, cmd_export]}
    for f in steps[a.cmd]:
        f(a)


if __name__ == "__main__":
    main()
