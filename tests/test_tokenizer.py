import random
import unicodedata
from collections import Counter

import pytest

from jamo.bpe import BPE, SPECIALS, UNK, _merge
from jamo.hangul import compose, decompose, has_batchim, to_compat

SENTENCES = [
    "낮과 낯은 소리가 같다",
    "오늘은 하늘이 맑아서 산책을 나갔다",
    "고양이가 창문 밖을 한참 동안 바라보았다",
    "받침이 있는 명사 뒤에는 을이 오고 받침이 없으면 를이 온다",
    "세종은 훈민정음을 만들어 백성에게 널리 알렸다",
    "나는 사과를 좋아하고 동생은 바나나를 좋아한다",
    "여기에 책이 있고 저기에 의자가 있다",
    "ㅋㅋㅋ 이건 좀 웃기네 ㅠㅠ",
    "GPT-2 모델은 2019년에 공개됐다",
    "값이 싸서 닭을 샀는데 흙이 묻어 있었다",
]


def fixture_lines(n=400, seed=0):
    """직접 쓴 문장 + 무작위 음절 단어 (드문 받침 조합까지 섞이게)."""
    rng = random.Random(seed)
    lines = list(SENTENCES) * 5
    for _ in range(n):
        words = []
        for _ in range(rng.randint(1, 6)):
            w = "".join(chr(0xAC00 + rng.randrange(11172)) for _ in range(rng.randint(1, 3)))
            words.append(w + rng.choice(["을", "를", "이", "가", "은", "는", ""]))
        lines.append(" ".join(words))
    return lines


# ------------------------------------------------------------------ hangul
def test_decompose_matches_unicodedata_for_all_syllables():
    for cp in range(0xAC00, 0xD7A4):
        ch = chr(cp)
        assert decompose(ch) == unicodedata.normalize("NFD", ch)
        assert compose(decompose(ch)) == ch


def test_examples():
    assert [hex(ord(c)) for c in decompose("낮")] == ["0x1102", "0x1161", "0x11bd"]
    assert to_compat(decompose("낯")) == "ㄴㅏㅊ"
    assert has_batchim("책") and not has_batchim("사과") and not has_batchim("GPT")


def test_compose_tolerates_broken_jamo():
    lead_only = decompose("가")[0]
    assert compose(lead_only + "x") == lead_only + "x"
    assert compose(decompose("한글") + lead_only) == "한글" + lead_only


# ------------------------------------------------------------------ bpe
def naive_train(lines, mode, vocab_size, alphabet_size, min_freq=2):
    """매 병합마다 쌍을 처음부터 다시 세는 느린 정답 구현. 증분 구현과 결과가 같아야 함."""
    wc = Counter()
    for line in lines:
        wc.update(BPE.words(line, mode))
    cc = Counter()
    for w, f in wc.items():
        for ch in w:
            cc[ch] += f
    alphabet = [ch for ch, _ in sorted(cc.items(), key=lambda x: (-x[1], x[0]))[:alphabet_size]]
    sym = {ch: len(SPECIALS) + i for i, ch in enumerate(alphabet)}
    words = {w: [sym.get(ch, UNK) for ch in w] for w in wc}
    base = len(SPECIALS) + len(alphabet)
    merges = []
    while len(merges) < vocab_size - base:
        pc = Counter()
        for w, ids in words.items():
            for p in zip(ids, ids[1:]):
                if UNK not in p:
                    pc[p] += wc[w]
        if not pc:
            break
        p = min(pc, key=lambda q: (-pc[q], q))
        if pc[p] < min_freq:
            break
        new = base + len(merges)
        merges.append(p)
        words = {w: _merge(ids, p[0], p[1], new) for w, ids in words.items()}
    return alphabet, merges, words


@pytest.mark.parametrize("mode", ["jamo", "syl"])
def test_incremental_training_matches_naive(mode):
    lines = fixture_lines(150)
    tok = BPE.train(lines, mode, vocab_size=600, alphabet_size=400, log=lambda *_: None)
    alphabet, merges, words = naive_train(lines, mode, 600, 400)
    assert tok.alphabet == alphabet
    assert tok.merges == merges
    # 학습 때 만들어진 분할 == 저장된 병합 규칙으로 다시 인코딩한 분할
    for w, ids in words.items():
        assert tok.encode_word(w) == ids


@pytest.mark.parametrize("mode", ["jamo", "syl"])
def test_roundtrip_and_save_load(mode, tmp_path):
    lines = fixture_lines()
    tok = BPE.train(lines, mode, vocab_size=4000, alphabet_size=3000, log=lambda *_: None)
    for s in lines:
        assert tok.decode(tok.encode(s)) == s
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = BPE.load(p)
    assert tok2.vocab == tok.vocab
    assert all(tok2.encode(s) == tok.encode(s) for s in lines)


def test_unknown_chars_become_unk():
    tok = BPE.train(SENTENCES, "syl", vocab_size=300, alphabet_size=3000, log=lambda *_: None)
    assert UNK in tok.encode("뷁")          # 학습 데이터에 없는 음절
    jt = BPE.train(SENTENCES, "jamo", vocab_size=300, alphabet_size=3000, log=lambda *_: None)
    # 자모 모드는 처음 보는 음절도 자모가 다 아는 글자면 UNK 없이 표현 가능
    assert UNK not in jt.encode("낟")
