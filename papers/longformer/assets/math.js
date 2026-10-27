"use strict";
globalThis.LongformerMath = {
  pattern(n, radius, dilation, globals, causal) {
    const hubs = new Set(globals);
    return Array.from({length: n}, (_, q) => Array.from({length: n}, (_, k) => {
      if (causal && k > q) return false;
      return hubs.has(q) || hubs.has(k) || (Math.abs(q - k) <= radius * dilation && (q - k) % dilation === 0);
    }));
  },
  reachable(matrix, source, layers) {
    let affected = new Set([source]);
    for (let layer = 0; layer < layers; layer++) {
      const previous = affected;
      affected = new Set();
      matrix.forEach((row, q) => {
        if (row.some((allowed, k) => allowed && previous.has(k))) affected.add(q);
      });
    }
    return [...affected].sort((a, b) => a - b);
  },
  costs(n, w, g, m, D, layers, heads, bytes) {
    // Globals occupy the first g positions; count the exact union, including edges.
    const r = Math.min(Math.floor(w / 2), n - 1);
    const localEntries = n * (2 * r + 1) - r * (r + 1);
    let ordinaryLocal = localEntries;
    let ordinaryGlobal = 0;
    for (let q = 0; q < g; q++) ordinaryLocal -= Math.min(n - 1, q + r) - Math.max(0, q - r) + 1;
    for (let q = g; q < n; q++) {
      const overlap = Math.max(0, Math.min(g - 1, q + r) - Math.max(0, q - r) + 1);
      ordinaryGlobal += g - overlap;
    }
    const sparseEntries = ordinaryLocal + ordinaryGlobal + g * n;
    return {
      localEntries, sparseEntries, denseEntries: n * n,
      sparseScoreBytes: sparseEntries * heads * bytes,
      denseScoreBytes: n * n * heads * bytes,
      crossEntries: n * m, decoderEntries: m * m,
      crossKV: 2 * layers * n * D * bytes,
      selfKV: 2 * layers * m * D * bytes
    };
  }
};
