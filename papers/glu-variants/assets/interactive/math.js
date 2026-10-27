(function (scope) {
  "use strict";
  const sigmoid = x => 1 / (1 + Math.exp(-x));
  // Abramowitz-Stegun approximation; maximum erf error about 1.5e-7.
  function erf(x) {
    const sign = x < 0 ? -1 : 1, z = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * z);
    const p = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return sign * (1 - p * Math.exp(-z * z));
  }
  function activation(kind, x) {
    if (kind === "GLU") return sigmoid(x);
    if (kind === "Bilinear") return x;
    if (kind === "ReGLU") return Math.max(0, x);
    if (kind === "GEGLU") return x * (1 + erf(x / Math.SQRT2)) / 2;
    return x * sigmoid(x);
  }
  function scalar(kind, a, up) {
    const gate = activation(kind, a);
    return {kind, a, up, gate, product: gate * up};
  }
  function budget(D, H0, M, bytes, mode, multiple = 0) {
    const ideal = mode === "fixed" ? H0 : 2 * H0 / 3;
    const H = multiple ? Math.ceil(ideal / multiple) * multiple : Math.round(ideal);
    const baseParams = 2 * D * H0, gateParams = 3 * D * H;
    return {
      D, H0, H, ideal, M, bytes, mode, multiple, baseParams, gateParams,
      ratio: gateParams / baseParams,
      baseFlops: 4 * M * D * H0, gateFlops: 6 * M * D * H,
      baseWeightBytes: baseParams * bytes, gateWeightBytes: gateParams * bytes,
      baseHidden: M * H0 * bytes, gatePair: 2 * M * H * bytes,
      gatePairAndProduct: 3 * M * H * bytes,
      baseTraffic: 4 * M * H0 * bytes, baseFusedTraffic: 2 * M * H0 * bytes,
      gateFusedProductTraffic: 6 * M * H * bytes, gateSeparateTraffic: 8 * M * H * bytes
    };
  }
  scope.GLUMath = {sigmoid, erf, activation, scalar, budget};
  if (typeof module !== "undefined") module.exports = scope.GLUMath;
})(typeof window !== "undefined" ? window : globalThis);
