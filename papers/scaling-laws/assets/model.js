(function (root) {
  "use strict";
  function positive(x, name) {
    if (!Number.isFinite(x) || x <= 0) throw new RangeError(name + " must be positive");
  }
  function exponents(alphaN = .076, alphaS = .76, alphaB = .21) {
    [alphaN, alphaS, alphaB].forEach(x => positive(x, "exponent"));
    const loss = 1 / (1 / alphaN + 1 / alphaS + 1 / alphaB);
    return {loss, n: loss / alphaN, b: loss / alphaB, s: loss / alphaS};
  }
  function batch(r) {
    positive(r, "B/Bcrit");
    return {steps: 1 + 1 / r, examples: 1 + r, minComputeFraction: 1 / (1 + r)};
  }
  function allocation(decades, policy) {
    if (!Number.isFinite(decades)) throw new RangeError("budget must be finite");
    const p = policy === "kaplan" ? {n: .73, b: .24, s: .03, loss: .050}
      : policy === "theory" ? exponents()
      : policy === "chinchilla" ? {n: .5, loss: null}
      : null;
    if (!p) throw new RangeError("unknown policy");
    const power = x => 10 ** (decades * x);
    return {n: power(p.n), tokens: power(1 - p.n),
      b: p.b === undefined ? null : power(p.b),
      s: p.s === undefined ? null : power(p.s),
      loss: p.loss === null ? null : power(-p.loss), exponents: p};
  }
  function suboptimal(r, alphaN = .076, alphaS = .76) {
    positive(r, "N/Neff");
    const bracket = 1 + (alphaS / alphaN) * (1 - r ** -alphaN);
    if (bracket <= 0) return {valid: false, bracket, steps: null, compute: null};
    const steps = bracket ** (-1 / alphaS);
    return {valid: true, bracket, steps, compute: r * steps};
  }
  const api = {exponents, batch, allocation, suboptimal};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.ScalingMath = api;
})(typeof window === "undefined" ? globalThis : window);
