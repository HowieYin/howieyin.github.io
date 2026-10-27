// Mathematical teaching model. Numbers are capacities / states, never GPU timings.
export function mergeBlock(old, scores, values) {
  const mt = Math.max(...scores);
  if (!Number.isFinite(mt)) return { ...old, alpha: 1, beta: 0, mt, lt: 0 };
  const weights = scores.map(x => Number.isFinite(x) ? Math.exp(x - mt) : 0);
  const lt = weights.reduce((a, b) => a + b, 0);
  const ut = weights.reduce((a, b, i) => a + b * values[i], 0);
  const m = Math.max(old.m, mt);
  const alpha = old.l ? Math.exp(old.m - m) : 0;
  const beta = Math.exp(mt - m);
  const l = alpha * old.l + beta * lt;
  const u = alpha * old.u + beta * ut;
  return { m, l, u, o: l ? u / l : 0, alpha, beta, mt, lt };
}

export function costs({ n, d, batch, heads, layers, bytes, sramKB }) {
  const capacityWords = sramKB * 1024 / bytes;
  const M = Math.min(n * d, Math.max(d, capacityWords));
  return {
    singleQuadraticBytes: batch * heads * n * n * bytes,
    qkvoBytes: 4 * batch * heads * n * d * bytes,
    kvBytes: 2 * batch * layers * n * heads * d * bytes,
    statsBytes: 2 * batch * heads * n * 4,
    M, capacityWords,
    ioWords: n * n * d * d / M,
    denseIoWords: n * d + n * n,
  };
}

export function attentionBackward(Q, K, V, dO, mask, dropout, scale) {
  const N = Q.length, d = Q[0].length;
  const zero = () => Array.from({ length: N }, () => Array(d).fill(0));
  const dQ = zero(), dK = zero(), dV = zero();
  for (let i = 0; i < N; i++) {
    const scores = K.map((k, j) => mask[i][j] ? scale * Q[i].reduce((s, q, r) => s + q * k[r], 0) : -Infinity);
    const max = Math.max(...scores);
    if (!Number.isFinite(max)) continue;
    const w = scores.map(s => Math.exp(s - max));
    const l = w.reduce((a, b) => a + b, 0);
    const p = w.map(x => x / l);
    const dP = V.map((v, j) => dropout[i][j] * v.reduce((s, x, r) => s + x * dO[i][r], 0));
    const D = p.reduce((s, x, j) => s + x * dP[j], 0);
    for (let j = 0; j < N; j++) {
      const dS = p[j] * (dP[j] - D);
      for (let r = 0; r < d; r++) {
        dV[j][r] += p[j] * dropout[i][j] * dO[i][r];
        dQ[i][r] += scale * dS * K[j][r];
        dK[j][r] += scale * dS * Q[i][r];
      }
    }
  }
  return { dQ, dK, dV };
}
