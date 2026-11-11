"""작은 GPT (nanoGPT 계열). 웹 데모의 docs/js/gpt.js가 이 구조를 그대로 따라 구현함."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.h = c["n_head"]
        s.qkv = nn.Linear(c["n_embd"], 3 * c["n_embd"])
        s.proj = nn.Linear(c["n_embd"], c["n_embd"])

    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(x).view(B, T, 3, s.h, C // s.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        e = c["n_embd"]
        s.ln1, s.ln2 = nn.LayerNorm(e), nn.LayerNorm(e)
        s.attn = Attn(c)
        s.mlp = nn.Sequential(nn.Linear(e, 4 * e), nn.GELU(), nn.Linear(4 * e, e))
        s.drop = nn.Dropout(c["dropout"])

    def forward(s, x):
        x = x + s.drop(s.attn(s.ln1(x)))
        return x + s.drop(s.mlp(s.ln2(x)))


class GPT(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c = c
        s.tok = nn.Embedding(c["vocab"], c["n_embd"])
        s.pos = nn.Embedding(c["block"], c["n_embd"])
        s.drop = nn.Dropout(c["dropout"])
        s.blocks = nn.ModuleList(Block(c) for _ in range(c["n_layer"]))
        s.ln = nn.LayerNorm(c["n_embd"])
        s.head = nn.Linear(c["n_embd"], c["vocab"], bias=False)
        s.head.weight = s.tok.weight  # 입출력 임베딩 공유
        s.apply(s._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(s, idx):
        T = idx.shape[1]
        x = s.drop(s.tok(idx) + s.pos(torch.arange(T, device=idx.device)))
        for b in s.blocks:
            x = b(x)
        return s.head(s.ln(x))


def device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(path, dev):
    ck = torch.load(path, map_location=dev)
    m = GPT(ck["cfg"]).to(dev)
    m.load_state_dict(ck["model"])
    return m.eval()
