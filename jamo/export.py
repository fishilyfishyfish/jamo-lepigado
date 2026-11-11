"""웹 데모용 내보내기: 행렬은 행 단위 int8 양자화, 벡터(LayerNorm, bias)는 float32.

docs/models/<mode>/
    model.json      설정 + 텐서 목록(이름, 모양, 종류, weights.bin 안의 바이트 위치)
    weights.bin     텐서 데이터 (리틀 엔디언)
    tokenizer.json  BPE 토크나이저
"""
import json
import os
import shutil

import numpy as np


def export_web(model, tok_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()
          if k != "head.weight"}  # head는 tok 임베딩과 공유라 따로 안 씀
    buf = bytearray()
    tensors, deq = [], {}

    def align4():
        buf.extend(b"\0" * (-len(buf) % 4))

    for name, arr in sd.items():
        if arr.ndim == 2:
            scale = np.abs(arr).max(axis=1) / 127.0
            scale[scale == 0] = 1.0
            q = np.clip(np.round(arr / scale[:, None]), -127, 127).astype(np.int8)
            off = len(buf)
            buf.extend(q.tobytes())
            align4()
            soff = len(buf)
            buf.extend(scale.astype("<f4").tobytes())
            tensors.append({"name": name, "shape": list(arr.shape), "kind": "q8",
                            "offset": off, "scale_offset": soff})
            deq[name] = q.astype(np.float32) * scale[:, None].astype(np.float32)
        else:
            align4()
            off = len(buf)
            buf.extend(arr.astype("<f4").tobytes())
            tensors.append({"name": name, "shape": list(arr.shape), "kind": "f32", "offset": off})
            deq[name] = arr.astype(np.float32)
    with open(os.path.join(out_dir, "weights.bin"), "wb") as f:
        f.write(buf)
    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump({"config": model.c, "tensors": tensors}, f, indent=1)
    shutil.copy(tok_path, os.path.join(out_dir, "tokenizer.json"))
    deq["head.weight"] = deq["tok.weight"]
    return deq, len(buf)
