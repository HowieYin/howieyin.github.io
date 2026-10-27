(() => {
  "use strict";
  function norm(a, mode = "rms", p = 1, eps = 0) {
    if (!a.length || !a.every(Number.isFinite) || p <= 0 || p > 1 || eps < 0) throw new RangeError("Invalid normalization input");
    const mean = a.reduce((s, x) => s + x, 0) / a.length;
    const variance = a.reduce((s, x) => s + (x - mean) ** 2, 0) / a.length;
    const k = mode === "ln" ? a.length : Math.max(1, Math.ceil(a.length * p));
    const second = a.slice(0, k).reduce((s, x) => s + x * x, 0) / k;
    const den = Math.sqrt((mode === "ln" ? variance : second) + eps);
    const out = a.map(x => den === 0 ? NaN : (mode === "ln" ? x - mean : x) / den);
    return { out, den, mean, variance, k, rms: Math.sqrt(a.reduce((s, x) => s + x * x, 0) / a.length) };
  }
  function costs(M, D, s, f, k) {
    return { bytes: 2 * M * D * s + D * s, statisticBytes: 4 * M, speedup: 1 / (1 - f + f / k) };
  }
  globalThis.RMSModel = { norm, costs };
})();
