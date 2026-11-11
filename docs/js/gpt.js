// 브라우저에서 도는 GPT 추론. jamo/model.py 구조를 그대로 따라감.
// 토큰을 하나씩 넣으면서 K/V를 캐시에 쌓는 방식이라 생성할 때 이전 토큰을 다시 계산하지 않음.

function erf(x) {
  // Abramowitz & Stegun 7.1.26 (최대 오차 1.5e-7)
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}

export class GPT {
  constructor(meta, buffer) {
    const c = (this.cfg = meta.config);
    this.W = {};
    for (const t of meta.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      if (t.kind === "q8") {
        const q = new Int8Array(buffer, t.offset, n);
        const rows = t.shape[0], cols = t.shape[1];
        const sc = new Float32Array(buffer, t.scale_offset, rows);
        const w = new Float32Array(n);
        for (let r = 0; r < rows; r++) {
          const s = sc[r], o = r * cols;
          for (let j = 0; j < cols; j++) w[o + j] = q[o + j] * s;
        }
        this.W[t.name] = w;
      } else {
        this.W[t.name] = new Float32Array(buffer.slice(t.offset, t.offset + n * 4));
      }
    }
    const C = c.n_embd;
    this.x = new Float32Array(C);
    this.h = new Float32Array(C);
    this.qkv = new Float32Array(3 * C);
    this.att = new Float32Array(C);
    this.tmp = new Float32Array(C);
    this.mlp = new Float32Array(4 * C);
    this.scores = new Float32Array(c.block);
    this.logits = new Float32Array(c.vocab);
  }

  static async load(dir, onProgress) {
    const meta = await (await fetch(`${dir}/model.json`)).json();
    const res = await fetch(`${dir}/weights.bin`);
    const total = +res.headers.get("content-length") || 0;
    if (!res.body || !total || !onProgress) return new GPT(meta, await res.arrayBuffer());
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      onProgress(Math.min(1, got / total));  // 압축 전송이면 total이 실제보다 작을 수 있음
    }
    const buf = new Uint8Array(got);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    return new GPT(meta, buf.buffer);
  }

  newCache() {
    const { n_layer, block, n_embd } = this.cfg;
    return {
      len: 0,
      k: Array.from({ length: n_layer }, () => new Float32Array(block * n_embd)),
      v: Array.from({ length: n_layer }, () => new Float32Array(block * n_embd)),
    };
  }

  // 캐시 복사본. 쓰인 부분만 복사하고 extra 토큰만큼 자리를 더 잡음 (후보를 여러 갈래로 펼칠 때 씀)
  cloneCache(cache, extra) {
    const C = this.cfg.n_embd, used = cache.len * C, size = (cache.len + extra) * C;
    const cp = (arr) => { const a = new Float32Array(size); a.set(arr.subarray(0, used)); return a; };
    return { len: cache.len, k: cache.k.map(cp), v: cache.v.map(cp) };
  }

  // out = W x + b, W는 (rows, cols) 행 우선
  static linear(out, W, b, x, rows, cols) {
    for (let r = 0; r < rows; r++) {
      let s = b ? b[r] : 0;
      const o = r * cols;
      for (let j = 0; j < cols; j++) s += W[o + j] * x[j];
      out[r] = s;
    }
  }

  static layerNorm(out, x, g, b) {
    const n = x.length;
    let m = 0;
    for (let i = 0; i < n; i++) m += x[i];
    m /= n;
    let v = 0;
    for (let i = 0; i < n; i++) v += (x[i] - m) ** 2;
    const inv = 1 / Math.sqrt(v / n + 1e-5);
    for (let i = 0; i < n; i++) out[i] = (x[i] - m) * inv * g[i] + b[i];
  }

  // 토큰 하나를 위치 cache.len에 넣고 다음 토큰 logits를 돌려줌 (내부 버퍼라 필요하면 복사해서 쓸 것)
  step(token, cache) {
    const { n_layer, n_head, n_embd: C, block, vocab } = this.cfg;
    const W = this.W, p = cache.len;
    if (p >= block || (p + 1) * C > cache.k[0].length) throw new Error("context full");
    const hd = C / n_head, sc = 1 / Math.sqrt(hd);
    const x = this.x, h = this.h, qkv = this.qkv, att = this.att, tmp = this.tmp, mlp = this.mlp;
    const te = W["tok.weight"], pe = W["pos.weight"];
    for (let i = 0; i < C; i++) x[i] = te[token * C + i] + pe[p * C + i];

    for (let l = 0; l < n_layer; l++) {
      const P = `blocks.${l}.`;
      GPT.layerNorm(h, x, W[P + "ln1.weight"], W[P + "ln1.bias"]);
      GPT.linear(qkv, W[P + "attn.qkv.weight"], W[P + "attn.qkv.bias"], h, 3 * C, C);
      const K = cache.k[l], V = cache.v[l];
      K.set(qkv.subarray(C, 2 * C), p * C);
      V.set(qkv.subarray(2 * C, 3 * C), p * C);
      for (let hh = 0; hh < n_head; hh++) {
        const o = hh * hd;
        let mx = -Infinity;
        for (let j = 0; j <= p; j++) {
          let s = 0;
          const kj = j * C + o;
          for (let d = 0; d < hd; d++) s += qkv[o + d] * K[kj + d];
          s *= sc;
          this.scores[j] = s;
          if (s > mx) mx = s;
        }
        let z = 0;
        for (let j = 0; j <= p; j++) { this.scores[j] = Math.exp(this.scores[j] - mx); z += this.scores[j]; }
        for (let d = 0; d < hd; d++) att[o + d] = 0;
        for (let j = 0; j <= p; j++) {
          const a = this.scores[j] / z, vj = j * C + o;
          for (let d = 0; d < hd; d++) att[o + d] += a * V[vj + d];
        }
      }
      GPT.linear(tmp, W[P + "attn.proj.weight"], W[P + "attn.proj.bias"], att, C, C);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
      GPT.layerNorm(h, x, W[P + "ln2.weight"], W[P + "ln2.bias"]);
      GPT.linear(mlp, W[P + "mlp.0.weight"], W[P + "mlp.0.bias"], h, 4 * C, C);
      for (let i = 0; i < 4 * C; i++) mlp[i] = 0.5 * mlp[i] * (1 + erf(mlp[i] / Math.SQRT2));
      GPT.linear(tmp, W[P + "mlp.2.weight"], W[P + "mlp.2.bias"], mlp, C, 4 * C);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
    }
    GPT.layerNorm(h, x, W["ln.weight"], W["ln.bias"]);
    GPT.linear(this.logits, te, null, h, vocab, C);  // 출력층은 토큰 임베딩과 공유
    cache.len = p + 1;
    return this.logits;
  }

  // ids 전체의 로그 확률 합 (ids[0]은 조건으로만 쓰고 확률은 ids[1:]에 대해 계산)
  logProb(ids) {
    const cache = this.newCache();
    const tail = ids.slice(-this.cfg.block);
    let total = 0;
    for (let i = 0; i + 1 < tail.length; i++) {
      const lg = this.step(tail[i], cache);
      let mx = -Infinity;
      for (let j = 0; j < lg.length; j++) if (lg[j] > mx) mx = lg[j];
      let z = 0;
      for (let j = 0; j < lg.length; j++) z += Math.exp(lg[j] - mx);
      total += lg[tail[i + 1]] - mx - Math.log(z);
    }
    return total;
  }
}

// 간단한 시드 고정 난수 (데모 재현용)
export function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// temperature + top-k 샘플링
export function sample(logits, temp, topk, rand) {
  const n = logits.length, k = Math.min(topk, n);
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => logits[b] - logits[a]);
  const top = idx.slice(0, k);
  const mx = logits[top[0]];
  const w = top.map((i) => Math.exp((logits[i] - mx) / temp));
  const z = w.reduce((a, b) => a + b, 0);
  let r = rand() * z;
  for (let i = 0; i < k; i++) { r -= w[i]; if (r <= 0) return top[i]; }
  return top[k - 1];
}
