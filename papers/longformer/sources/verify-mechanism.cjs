const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const context = {};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), context);
const {pattern, reachable, costs} = context.LongformerMath;
// Independent specification: local radius r excludes no self connection.
for (let n = 2; n <= 64; n += 2) {
  for (let r = 1; r <= 8; r++) {
    for (const globals of [[], [0], [0, n - 1]]) {
      const matrix = pattern(n, r, 1, globals, false);
      let expected = 0;
      for (let q = 0; q < n; q++) {
        for (let k = 0; k < n; k++) {
          const allowed = Math.abs(q - k) <= r || globals.includes(q) || globals.includes(k);
          assert.equal(matrix[q][k], allowed);
          expected += Number(allowed);
        }
      }
      assert.equal(matrix.flat().filter(Boolean).length, expected);
    }
  }
}
const local = pattern(16, 1, 1, [], false);
assert.deepEqual(Array.from(reachable(local, 8, 2)), [6, 7, 8, 9, 10]);
const global = pattern(16, 1, 1, [0], false);
assert.equal(reachable(global, 15, 1).includes(8), false);
assert.equal(reachable(global, 15, 2).includes(8), true);
const dilated = pattern(16, 2, 3, [], false);
assert.equal(dilated[8][9], false);
assert.equal(dilated[8][11], true);
assert.equal(dilated[8][14], true);
const causal = pattern(8, 2, 1, [], true);
assert.equal(causal[3][4], false);
assert.equal(causal[3][1], true);
const c = costs(16384, 1024, 1, 256, 1024, 12, 16, 2);
assert.equal(c.crossKV, 805306368);
assert.equal(c.selfKV, 12582912);
assert.equal(c.denseEntries, 268435456);
assert.equal(c.localEntries, 16384 * 1025 - 512 * 513);
assert.equal(c.crossEntries, 4194304);
console.log("PASS: 768 exhaustive graph configurations (n=2..64); synchronous relay, dilation, causal mask and LED byte arithmetic.");
