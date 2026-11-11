"""
한글 음절 <-> 자모 분해/조합. unicodedata 안 쓰고 유니코드 산술로 직접 함.

현대 한글 음절 11,172자(가~힣)는 이 공식으로 배치돼 있음:
    음절 = 0xAC00 + (초성 * 21 + 중성) * 28 + 종성
    초성 19개, 중성 21개, 종성 28개(0은 받침 없음)

분해 결과는 "첫가끝" 조합형 자모(U+1100~)로 냄. 이 영역은 초성 ㄴ(U+1102)과
종성 ㄴ(U+11AB)이 서로 다른 글자라서, 자모열만 보고도 음절 경계가 정확히 복원됨.
"""

S_BASE, L_BASE, V_BASE, T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
L_COUNT, V_COUNT, T_COUNT = 19, 21, 28
N_COUNT = V_COUNT * T_COUNT          # 588
S_COUNT = L_COUNT * N_COUNT          # 11172


def is_syllable(ch):
    return 0 <= ord(ch) - S_BASE < S_COUNT


def is_lead(ch):
    return 0 <= ord(ch) - L_BASE < L_COUNT


def is_vowel(ch):
    return 0 <= ord(ch) - V_BASE < V_COUNT


def is_tail(ch):
    return 0 < ord(ch) - T_BASE < T_COUNT


def split_syllable(ch):
    """'낮' -> (2, 0, 22): 초성/중성/종성 인덱스."""
    s = ord(ch) - S_BASE
    return s // N_COUNT, (s % N_COUNT) // T_COUNT, s % T_COUNT


def has_batchim(word):
    """마지막 글자가 받침 있는 한글 음절인지."""
    return bool(word) and is_syllable(word[-1]) and split_syllable(word[-1])[2] != 0


def decompose(text):
    """'낮' -> 'ᄂ' 'ᅡ' 'ᆽ' (3글자). 한글 음절 말고는 그대로 둠."""
    out = []
    for ch in text:
        if is_syllable(ch):
            l, v, t = split_syllable(ch)
            out.append(chr(L_BASE + l))
            out.append(chr(V_BASE + v))
            if t:
                out.append(chr(T_BASE + t))
        else:
            out.append(ch)
    return "".join(out)


def compose(text):
    """decompose의 역. 초성+중성(+종성)이 나란히 있으면 음절로 합침.
    모델이 초성만 내놓고 끝내는 식의 깨진 자모열이 와도 터지지 않고 남은 건 그대로 둠."""
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if is_lead(ch) and i + 1 < n and is_vowel(text[i + 1]):
            s = (ord(ch) - L_BASE) * N_COUNT + (ord(text[i + 1]) - V_BASE) * T_COUNT
            i += 2
            if i < n and is_tail(text[i]):
                s += ord(text[i]) - T_BASE
                i += 1
            out.append(chr(S_BASE + s))
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# 화면 표시용: 조합형 자모는 붙어 있으면 다시 음절처럼 렌더링돼서, 낱자로 보여줄 땐 호환 자모로 바꿈
_COMPAT_L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_COMPAT_V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_COMPAT_T = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def to_compat(text):
    out = []
    for ch in text:
        if is_lead(ch):
            out.append(_COMPAT_L[ord(ch) - L_BASE])
        elif is_vowel(ch):
            out.append(_COMPAT_V[ord(ch) - V_BASE])
        elif is_tail(ch):
            out.append(_COMPAT_T[ord(ch) - T_BASE])
        else:
            out.append(ch)
    return "".join(out)
