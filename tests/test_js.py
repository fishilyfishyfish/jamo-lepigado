"""웹 데모(JS) 구현이 파이썬 구현과 같은 결과를 내는지 확인. node가 없으면 건너뜀."""
import json
import os
import shutil
import subprocess

import pytest

from jamo.bpe import BPE
from jamo.data import clean
from tests.test_tokenizer import SENTENCES, fixture_lines

HERE = os.path.dirname(__file__)
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node 없음")


def run_js(*args):
    out = subprocess.run([NODE, os.path.join(HERE, "js_runner.mjs"), *map(str, args)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.parametrize("mode", ["jamo", "syl"])
def test_tokenizer_matches_python(mode, tmp_path):
    lines = fixture_lines()
    tok = BPE.train(lines, mode, vocab_size=4000, alphabet_size=3000, log=lambda *_: None)
    tok.save(tmp_path / "tok.json")
    probe = lines + ["처음 보는 뷁 같은 글자", "  공백이   여러 개  ", "English words 123", "😀이모지"]
    (tmp_path / "lines.json").write_text(json.dumps(probe, ensure_ascii=False), encoding="utf-8")
    js = run_js("tok", tmp_path / "tok.json", tmp_path / "lines.json")
    for s, r in zip(probe, js):
        assert r["ids"] == tok.encode(s), s
        assert r["dec"] == tok.decode(tok.encode(s)), s
        assert r["clean"] == clean(s), s
        assert r["rt"]


def test_gpt_matches_python(tmp_path):
    torch = pytest.importorskip("torch")
    from jamo.export import export_web
    from jamo.model import GPT

    torch.manual_seed(0)
    tok = BPE.train(SENTENCES * 3, "jamo", vocab_size=300, alphabet_size=3000, log=lambda *_: None)
    tok.save(tmp_path / "tok.json")
    cfg = dict(vocab=len(tok), block=32, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    model = GPT(cfg).eval()
    for p in model.parameters():  # 초기값(std 0.02)은 너무 작아서 차이가 안 드러나니 키움
        p.data.normal_(0, 0.3)
    deq, _ = export_web(model, tmp_path / "tok.json", tmp_path / "web")
    qm = GPT(cfg).eval()
    qm.load_state_dict({k: torch.from_numpy(v) for k, v in deq.items()})
    ids = [0] + tok.encode("낮과 낯은 소리가 같다")
    with torch.no_grad():
        ref = qm(torch.tensor([ids]))[0]
        lp = torch.log_softmax(ref, -1)
        ref_logprob = sum(lp[i, ids[i + 1]].item() for i in range(len(ids) - 1))
    (tmp_path / "ids.json").write_text(json.dumps(ids))
    js = run_js("gpt", tmp_path / "web", tmp_path / "ids.json")
    got = torch.tensor(js["logits"])
    assert got.shape == ref.shape
    assert (got - ref).abs().max().item() < 1e-3
    assert abs(js["logprob"] - ref_logprob) < 1e-2
