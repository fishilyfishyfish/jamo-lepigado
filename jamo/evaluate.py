"""평가: 글자당 비트(bpc), 받침-조사 선택 정답률, 생성 샘플."""
import math
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from .bpe import EOS
from .hangul import has_batchim, is_syllable, split_syllable

T_RIEUL = 8  # 종성 ㄹ 인덱스


@torch.no_grad()
def bpc(model, val, tokens_per_char, max_tokens=500_000, batch=32):
    """검증 토큰열을 block 길이로 겹치지 않게 잘라 평균 nll을 구하고 글자당 비트로 환산.
    토크나이저마다 같은 글을 표현하는 토큰 수가 달라서, 토큰당 loss가 아니라 이 값을 비교해야 공정함."""
    dev = next(model.parameters()).device
    T = model.c["block"]
    n = min(len(val), max_tokens)
    starts = list(range(0, n - T - 1, T))
    nll = cnt = 0
    for i in range(0, len(starts), batch):
        idx = starts[i:i + batch]
        x = torch.from_numpy(np.stack([val[j:j + T] for j in idx]).astype(np.int64)).to(dev)
        y = torch.from_numpy(np.stack([val[j + 1:j + T + 1] for j in idx]).astype(np.int64)).to(dev)
        nll += F.cross_entropy(model(x).flatten(0, 1).float(), y.flatten(), reduction="sum").item()
        cnt += y.numel()
    return nll / cnt * tokens_per_char / math.log(2)


# ------------------------------------------------------------------ 받침-조사 테스트
def _batchim(w):
    return has_batchim(w)


def _batchim_not_rieul(w):
    # '으로/로'는 받침이 ㄹ이면 '로' (서울로, 교실로)
    return has_batchim(w) and split_syllable(w[-1])[2] != T_RIEUL


# (문장 틀, (받침 있을 때, 없을 때), 앞쪽 조사를 고르는 조건)
TEMPLATES = [
    ("나는 {}{} 좋아한다", ("을", "를"), _batchim),
    ("여기에 {}{} 있다", ("이", "가"), _batchim),
    ("그 {}{} 정말 좋다", ("은", "는"), _batchim),
    ("우리는 {}{} 갔다", ("으로", "로"), _batchim_not_rieul),
]

NOUNS = ("사과 바나나 책 연필 컴퓨터 물 학교 고양이 강아지 집 자동차 꽃 나무 의자 밥 우유 "
         "신발 가방 시계 창문 선생님 친구 음악 영화 그림 사진 편지 공책 지우개 볼펜 "
         "바다 산 하늘 구름 도시 나라 사람 아이 노래 문제 서울 교실 마을 거실 병원 공원").split()


def nonce_nouns(train_lines, n=120, seed=1337):
    """코퍼스에 자주 나오는 음절로 가짜 명사를 만듦. 어떤 단어의 앞 두 글자로도 안 나온 것만.
    끝 음절은 (받침 없음 / ㄹ 받침 / 다른 받침)을 번갈아 골라서 세 경우가 고르게 들어가게 함."""
    syl = Counter(ch for s in train_lines[:200_000] for ch in s if is_syllable(ch))
    top = [c for c, _ in syl.most_common(300)]
    groups = [
        [c for c in top if split_syllable(c)[2] == 0],
        [c for c in top if split_syllable(c)[2] == T_RIEUL],
        [c for c in top if split_syllable(c)[2] not in (0, T_RIEUL)],
    ]
    groups = [g for g in groups if g]
    seen = {w[:2] for s in train_lines for w in s.split()}
    rng = random.Random(seed)
    out, tries = [], 0
    while len(out) < n and tries < 100_000:
        tries += 1
        w = rng.choice(top) + rng.choice(groups[len(out) % len(groups)])
        if w not in seen and w not in out:
            out.append(w)
    return out


@torch.no_grad()
def logp(model, tok, text):
    dev = next(model.parameters()).device
    ids = [EOS] + tok.encode(text)
    x = torch.tensor([ids[-model.c["block"]:]], device=dev)
    lp = F.log_softmax(model(x)[0, :-1].float(), -1)
    return lp.gather(1, x[0, 1:, None]).sum().item()


def josa_accuracy(model, tok, nouns):
    res = {}
    for tpl, (first, second), rule in TEMPLATES:
        ok = 0
        for w in nouns:
            right, wrong = (first, second) if rule(w) else (second, first)
            ok += logp(model, tok, tpl.format(w, right)) > logp(model, tok, tpl.format(w, wrong))
        res[f"{first}/{second}"] = ok / len(nouns)
    return res


# ------------------------------------------------------------------ 생성
@torch.no_grad()
def generate(model, tok, prompt="", length=100, temp=0.8, topk=50, seed=None):
    dev = next(model.parameters()).device
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    ids = [EOS] + (tok.encode(prompt) if prompt else [])
    for _ in range(length):
        x = torch.tensor([ids[-model.c["block"]:]], device=dev)
        logits = model(x)[0, -1].float().cpu() / temp
        v, _ = torch.topk(logits, min(topk, logits.numel()))
        logits[logits < v[-1]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=g).item()
        if nxt == EOS:
            break
        ids.append(nxt)
    return tok.decode(ids)
