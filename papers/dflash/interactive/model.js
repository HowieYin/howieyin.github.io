(function () {
  'use strict';
  window.DFlashModel = Object.freeze({
    prefix(draft, target) {
      let n = 0;
      while (n < Math.min(draft.length, target.length) && draft[n] === target[n]) n++;
      return n;
    },
    expectedLength(conditionalAcceptance) {
      let survival = 1, result = 1;
      for (const a of conditionalAcceptance) { survival *= a; result += survival; }
      return result;
    },
    lossWeight(k, decay) { return Math.exp(-(k - 1) / decay); },
    memory({ B, S, D, F, L, G, d, b }) {
      return {
        projection: F * D * D * b,
        input: B * S * F * D * b,
        output: B * S * D * b,
        kv: 2 * L * B * S * G * d * b
      };
    },
    speedup(targetLatency, draftLatency, verifyLatency, tau) {
      return targetLatency * tau / (draftLatency + verifyLatency);
    }
  });
})();
