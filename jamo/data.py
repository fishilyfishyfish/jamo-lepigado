"""코퍼스 받기, 정리, 학습/검증 분할."""
import os
import re
import unicodedata

# 한글 음절, 호환 자모(ㅋㅋ 등), ASCII만 남기고 나머지(한자, 이모지 등)는 지움.
# 두 모드가 완전히 같은 텍스트를 보게 하려고 여기서 한 번에 정리함.
_DROP = re.compile(r"[^ -~가-힣ㄱ-ㆎ]")
_SPACES = re.compile(r" {2,}")


def clean(line):
    line = _DROP.sub("", unicodedata.normalize("NFC", line))
    return _SPACES.sub(" ", line).strip()


def read_split(path, val_every=100):
    """줄 단위로 읽어서 val_every줄마다 1줄을 검증용으로 뗌 (두 모드 공통)."""
    train, val = [], []
    with open(path, encoding="utf-8") as f:
        for i, raw in enumerate(f):
            s = clean(raw)
            if s:
                (val if i % val_every == 0 else train).append(s)
    return train, val


def download_wiki(path, max_chars):
    """한국어 위키백과를 스트리밍으로 받아 max_chars자까지 저장."""
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in ds:
            for line in ex["text"].split("\n"):
                line = line.strip()
                if len(line) < 20:  # 제목, 목록 조각 등은 버림
                    continue
                f.write(line + "\n")
                n += len(line)
            if n >= max_chars:
                break
    print(f"{path}: {n:,}자", flush=True)
    # 스트리밍이 백그라운드에서 계속 받아오다가 종료 시점에 재시도 루프에 걸려
    # 프로세스가 안 끝나는 문제가 있어서, 파일을 닫은 뒤 바로 강제 종료함.
    os._exit(0)
