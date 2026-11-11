// 한글 음절 <-> 자모 분해/조합. jamo/hangul.py와 똑같이 동작해야 함.
// 음절 = 0xAC00 + (초성 * 21 + 중성) * 28 + 종성

const S_BASE = 0xac00, L_BASE = 0x1100, V_BASE = 0x1161, T_BASE = 0x11a7;
const L_COUNT = 19, V_COUNT = 21, T_COUNT = 28;
const N_COUNT = V_COUNT * T_COUNT, S_COUNT = L_COUNT * N_COUNT;

const cp = (ch) => ch.codePointAt(0);
export const isSyllable = (ch) => { const s = cp(ch) - S_BASE; return s >= 0 && s < S_COUNT; };
export const isLead = (ch) => { const s = cp(ch) - L_BASE; return s >= 0 && s < L_COUNT; };
export const isVowel = (ch) => { const s = cp(ch) - V_BASE; return s >= 0 && s < V_COUNT; };
export const isTail = (ch) => { const s = cp(ch) - T_BASE; return s > 0 && s < T_COUNT; };

export function splitSyllable(ch) {
  const s = cp(ch) - S_BASE;
  return [Math.floor(s / N_COUNT), Math.floor((s % N_COUNT) / T_COUNT), s % T_COUNT];
}

export function hasBatchim(word) {
  const last = Array.from(word).pop();
  return !!last && isSyllable(last) && splitSyllable(last)[2] !== 0;
}

export const T_RIEUL = 8;
export function tailIndex(word) {
  const last = Array.from(word).pop();
  return last && isSyllable(last) ? splitSyllable(last)[2] : 0;
}

export function decompose(text) {
  let out = "";
  for (const ch of text) {
    if (isSyllable(ch)) {
      const [l, v, t] = splitSyllable(ch);
      out += String.fromCharCode(L_BASE + l, V_BASE + v);
      if (t) out += String.fromCharCode(T_BASE + t);
    } else out += ch;
  }
  return out;
}

export function compose(text) {
  const c = Array.from(text);
  let out = "", i = 0;
  while (i < c.length) {
    if (isLead(c[i]) && i + 1 < c.length && isVowel(c[i + 1])) {
      let s = (cp(c[i]) - L_BASE) * N_COUNT + (cp(c[i + 1]) - V_BASE) * T_COUNT;
      i += 2;
      if (i < c.length && isTail(c[i])) { s += cp(c[i]) - T_BASE; i++; }
      out += String.fromCharCode(S_BASE + s);
    } else { out += c[i]; i++; }
  }
  return out;
}

const COMPAT_L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ";
const COMPAT_V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ";
const COMPAT_T = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ";

// 화면 표시용: 조합형 자모는 붙어 있으면 음절처럼 렌더링되니까 낱자(호환 자모)로 바꿈
export function toCompat(text) {
  let out = "";
  for (const ch of text) {
    if (isLead(ch)) out += COMPAT_L[cp(ch) - L_BASE];
    else if (isVowel(ch)) out += COMPAT_V[cp(ch) - V_BASE];
    else if (isTail(ch)) out += COMPAT_T[cp(ch) - T_BASE];
    else out += ch;
  }
  return out;
}

// 한글 자판으로 치는 도중의 상태를 자모 앞부분으로 바꿈.
// 자판은 받침을 일단 앞 음절에 붙였다가 모음이 오면 뒤로 넘기니까('서울'을 치면 서 → 성 → 서우),
// 마지막 음절에 받침이 있으면 "그 받침이 다음 음절 초성으로 넘어갈 수도 있는" 경우를 하나 더 만든다.
const DOUBLE_TAIL = { 3: [1, "ㅅ"], 5: [4, "ㅈ"], 6: [4, "ㅎ"], 9: [8, "ㄱ"], 10: [8, "ㅁ"], 11: [8, "ㅂ"],
  12: [8, "ㅅ"], 13: [8, "ㅌ"], 14: [8, "ㅍ"], 15: [8, "ㅎ"], 18: [17, "ㅅ"] };
export function typingVariants(word) {
  const chars = Array.from(word);
  if (!chars.length) return [""];
  const head = decompose(chars.slice(0, -1).join(""));
  const last = chars[chars.length - 1];
  const li = COMPAT_L.indexOf(last);
  if (li >= 0) return [head + String.fromCharCode(L_BASE + li)];  // 자음만 친 상태 (ㅅ)
  if (!isSyllable(last)) return [decompose(word)];
  const [l, v, t] = splitSyllable(last);
  const out = [decompose(word)];
  if (t) {
    const [rest, moved] = DOUBLE_TAIL[t] || [0, COMPAT_T[t]];
    const lead = COMPAT_L.indexOf(moved);
    if (lead >= 0) {
      out.push(head + String.fromCharCode(L_BASE + l, V_BASE + v) +
        (rest ? String.fromCharCode(T_BASE + rest) : "") + String.fromCharCode(L_BASE + lead));
    }
  }
  return out;
}

// 학습 데이터 정리(jamo/data.py clean)와 같은 규칙
const DROP = /[^ -~가-힣ㄱ-ㆎ]/g;
export function clean(text) {
  return text.normalize("NFC").replace(DROP, "").replace(/ {2,}/g, " ").trim();
}
