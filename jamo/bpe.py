"""
BPE 토크나이저를 라이브러리 없이 직접 구현.

두 가지 모드:
    jamo : 학습/인코딩 전에 한글 음절을 자모로 분해 (낮 -> ᄂ ᅡ ᆽ)
    syl  : 음절 그대로

단어 경계는 SentencePiece처럼 각 단어 앞에 '▁'를 붙여서 표시하고, 병합은 단어 안에서만 일어남.

어휘 id 배치:
    0, 1             특수 토큰 <eos>, <unk>
    2 .. 2+A-1       기본 글자(alphabet) A개
    그 뒤            병합으로 생긴 토큰. k번째 병합의 결과 = 2 + A + k
"""
import heapq
import json
from collections import Counter, defaultdict

from .hangul import compose, decompose

SPACE = "▁"  # ▁
SPECIALS = ["<eos>", "<unk>"]
EOS, UNK = 0, 1


def _merge(ids, a, b, new):
    """ids 안의 (a, b) 쌍을 왼쪽부터 겹치지 않게 전부 new로 바꿈."""
    out, i, n = [], 0, len(ids)
    while i < n:
        if i + 1 < n and ids[i] == a and ids[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPE:
    def __init__(self, mode, alphabet, merges):
        assert mode in ("jamo", "syl")
        self.mode = mode
        self.alphabet = list(alphabet)
        self.merges = [tuple(m) for m in merges]
        self.base = len(SPECIALS) + len(self.alphabet)
        self.sym2id = {ch: len(SPECIALS) + i for i, ch in enumerate(self.alphabet)}
        self.ranks = {m: r for r, m in enumerate(self.merges)}
        self.vocab = SPECIALS + self.alphabet
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.cache = {}

    def __len__(self):
        return len(self.vocab)

    # ------------------------------------------------------------ text <-> words
    @staticmethod
    def words(text, mode):
        if mode == "jamo":
            text = decompose(text)
        return [SPACE + w for w in text.split(" ") if w]

    # ------------------------------------------------------------ encode / decode
    def encode_word(self, word):
        hit = self.cache.get(word)
        if hit is not None:
            return hit
        ids = [self.sym2id.get(ch, UNK) for ch in word]
        ranks = self.ranks
        while len(ids) > 1:
            best, best_rank = None, None
            for p in zip(ids, ids[1:]):
                r = ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            ids = _merge(ids, best[0], best[1], self.base + best_rank)
        if len(self.cache) < 2_000_000:
            self.cache[word] = ids
        return ids

    def encode(self, text):
        out = []
        for w in self.words(text, self.mode):
            out.extend(self.encode_word(w))
        return out

    def decode(self, ids):
        s = "".join("�" if i == UNK else self.vocab[i] for i in ids if i != EOS)
        s = s.replace(SPACE, " ")
        if s.startswith(" "):
            s = s[1:]
        return compose(s) if self.mode == "jamo" else s

    def tokens(self, text):
        """시각화용: 토큰 문자열 목록."""
        return [self.vocab[i] for i in self.encode(text)]

    # ------------------------------------------------------------ save / load
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "mode": self.mode, "specials": SPECIALS,
                       "alphabet": self.alphabet, "merges": [list(m) for m in self.merges]},
                      f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["mode"], d["alphabet"], d["merges"])

    # ------------------------------------------------------------ train
    @classmethod
    def train(cls, lines, mode, vocab_size=8000, alphabet_size=3000, min_freq=2, log=print):
        # 1) 단어 빈도
        wc = Counter()
        for line in lines:
            wc.update(cls.words(line, mode))
        log(f"  고유 단어 {len(wc):,}개")

        # 2) 기본 글자: 빈도순 상위 alphabet_size개 (나머지는 <unk>)
        cc = Counter()
        for w, f in wc.items():
            for ch in w:
                cc[ch] += f
        alphabet = [ch for ch, _ in sorted(cc.items(), key=lambda x: (-x[1], x[0]))[:alphabet_size]]
        base = len(SPECIALS) + len(alphabet)
        n_merges = vocab_size - base
        if n_merges <= 0:
            raise ValueError(f"vocab_size {vocab_size}가 기본 글자 {len(alphabet)}개보다 작음")
        log(f"  기본 글자 {len(alphabet)}개 (전체 {len(cc)}종), 병합 {n_merges}번 예정")
        sym2id = {ch: len(SPECIALS) + i for i, ch in enumerate(alphabet)}

        # 3) 쌍 빈도와 "그 쌍이 들어 있는 단어" 색인
        words = [[sym2id.get(ch, UNK) for ch in w] for w in wc]
        freqs = list(wc.values())
        del wc
        count = defaultdict(int)
        where = defaultdict(set)
        for wi, ids in enumerate(words):
            f = freqs[wi]
            for p in zip(ids, ids[1:]):
                if UNK not in p:
                    count[p] += f
                    where[p].add(wi)
        heap = [(-c, p) for p, c in count.items()]
        heapq.heapify(heap)

        # 4) 가장 잦은 쌍부터 병합. 바뀐 단어만 다시 세는 증분 방식.
        merges = []
        while len(merges) < n_merges and heap:
            negc, p = heapq.heappop(heap)
            c = count.get(p, 0)
            if c != -negc:      # 옛날 기록(이미 빈도가 바뀜) -> 버림
                continue
            if c < min_freq:
                break
            new = base + len(merges)
            merges.append(p)
            a, b = p
            changed = set()
            for wi in where.pop(p):
                ids = words[wi]
                merged = _merge(ids, a, b, new)
                if len(merged) == len(ids):
                    continue
                f = freqs[wi]
                for q in zip(ids, ids[1:]):
                    if UNK not in q:
                        count[q] -= f
                        changed.add(q)
                for q in zip(merged, merged[1:]):
                    if UNK not in q:
                        count[q] += f
                        where[q].add(wi)
                        changed.add(q)
                words[wi] = merged
            count.pop(p, None)
            for q in changed:
                cq = count.get(q, 0)
                if cq > 0:
                    heapq.heappush(heap, (-cq, q))
                else:
                    count.pop(q, None)
            if len(merges) % 1000 == 0:
                log(f"  병합 {len(merges)}/{n_merges}")
        log(f"  병합 {len(merges)}번 완료, 어휘 {base + len(merges)}")
        return cls(mode, alphabet, merges)
