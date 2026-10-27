(function (root) {
  "use strict";
  const X = [[1, -16, 2, 6], [-2, 8, -1, -9]];
  const W = [[2, 1, -2], [1, -1, -1], [2, -1, -2], [-1, -1, 1]];
  const multiply = (a, b) => a.map(row => b[0].map((_, o) => row.reduce((v, x, j) => v + x * b[j][o], 0)));
  const maxError = (a, b) => Math.max(...a.flatMap((row, t) => row.map((x, o) => Math.abs(x - b[t][o]))));
  const quantize = (a, bits) => {
    const qmax = 2 ** (bits - 1) - 1;
    const delta = Math.max(...a.flat().map(Math.abs)) / qmax || 1;
    return a.map(row => row.map(x => Math.sign(x) * Math.floor(Math.abs(x / delta) + 0.5) * delta));
  };
  function smooth(alpha, bits) {
    const a = [2, 16, 2, 9], w = [2, 1, 2, 1];
    const s = a.map((x, j) => x ** alpha / w[j] ** (1 - alpha));
    const xhat = X.map(row => row.map((x, j) => x / s[j]));
    const what = W.map((row, j) => row.map(x => x * s[j]));
    const y = multiply(X, W), transformed = multiply(xhat, what);
    const quantized = multiply(quantize(xhat, bits), quantize(what, bits));
    return {alpha, bits, s, a: a.map((x, j) => x / s[j]), w: w.map((x, j) => x * s[j]),
      y, transformed, quantized, floatError: maxError(y, transformed), quantError: maxError(y, quantized)};
  }
  function memory(p) {
    const weights = p.params * 1e9 * p.wbits / 8;
    const kv = 2 * p.batch * p.length * p.layers * p.heads * p.dim * p.kvbits / 8;
    return {...p, weights, kv, total: weights + kv};
  }
  const latency = [
    [152.6, 237.1, 124.5, 120.5, 112.1],
    [296.3, 371.5, 243.3, 235.1, 223.1],
    [343.0, 387.9, 246.7, 240.2, 227.6],
    [659.9, 654.9, 490.7, 478.3, 458.4]
  ];
  root.SQMath = {smooth, memory, latency};
})(typeof window !== "undefined" ? window : globalThis);
