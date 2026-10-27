// Pure arithmetic shared by the reading and its independent checks.
export function attention(Q, K, V, { causal = false, scaled = true } = {}) {
  const divisor = scaled ? Math.sqrt(Q[0].length) : 1;
  const weights = Q.map((query, i) => {
    const scores = K.map((key, j) => causal && j > i
      ? -Infinity
      : query.reduce((sum, value, d) => sum + value * key[d], 0) / divisor);
    const maximum = Math.max(...scores);
    const exp = scores.map(value => Math.exp(value - maximum));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(value => value / sum);
  });
  const output = weights.map(row => V[0].map((_, d) =>
    row.reduce((sum, value, j) => sum + value * V[j][d], 0)));
  return { weights, output };
}

export function kvBytes({ layers, batch, tokens, heads, dim, bytes }) {
  return 2 * layers * batch * tokens * heads * dim * bytes;
}

export function blockFlops(n, D, dff) {
  return { projection: 8 * n * D ** 2, mixing: 4 * n ** 2 * D, ffn: 4 * n * D * dff };
}
