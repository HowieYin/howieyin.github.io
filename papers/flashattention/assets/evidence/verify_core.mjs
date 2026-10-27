const modulePath = import.meta.url.includes('/sources/') ? '../interactive/mechanism.mjs' : '../mechanism.mjs';
const { mergeBlock, costs, attentionBackward } = await import(modulePath);
const forward = [];
for (const scores of [[1, 2, 3, 4], [1001, 1002, 1003, 1004], [-1000, -998, -999, -997], [-Infinity, 2, -Infinity, 4], [-Infinity, -Infinity, -Infinity, -Infinity]]) {
  for (const block of [1, 2, 3, 4]) {
    const values = [10, 20, 30, 40];
    let state = { m: -Infinity, l: 0, u: 0, o: 0 };
    for (let i = 0; i < scores.length; i += block) {
      state = mergeBlock(state, scores.slice(i, i + block), values.slice(i, i + block));
    }
    forward.push({ scores: scores.map(x => Number.isFinite(x) ? x : '-Infinity'), values, block, output: state.o });
  }
}
const Q = [[.2, -.3], [.4, .1], [-.2, .5]];
const K = [[.1, .3], [-.5, .2], [.6, -.1]];
const V = [[.3, .8], [-.2, .4], [.7, -.6]];
const dO = [[.5, -.1], [.2, .4], [-.3, .2]];
const mask = [[true, true, false], [true, true, true], [false, false, false]];
const dropout = [[1.25, 0, 1.25], [1.25, 1.25, 0], [0, 1.25, 1.25]];
const scale = 1 / Math.sqrt(2);
const gradients = attentionBackward(Q, K, V, dO, mask, dropout, scale);
const cost = costs({ n: 4096, d: 64, batch: 16, heads: 8, layers: 32, bytes: 2, sramKB: 128 });
const c2 = costs({ n: 8192, d: 64, batch: 16, heads: 8, layers: 32, bytes: 2, sramKB: 128 });
const toyCorrect = forward[1].output;
console.log(JSON.stringify({ forward, Q, K, V, dO, mask, dropout, scale, ...gradients, cost, ioDoublingRatio: c2.ioWords / cost.ioWords, toyCorrect }));
