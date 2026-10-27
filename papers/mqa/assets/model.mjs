export function cacheModel({ heads, kvHeads, width, length, batch, layers, bytes, window = 0, step = 0 }) {
  if (heads % kvHeads !== 0) throw new RangeError("Query heads must be divisible by KV heads");
  const totalLength = length + step;
  const activeLength = window ? Math.min(totalLength, window) : totalLength;
  const capacity = 2 * batch * layers * kvHeads * width * activeLength * bytes;
  const mhaCapacity = 2 * batch * layers * heads * width * activeLength * bytes;
  // Logical one-pass cache reads; real traffic depends on kernel reuse and tiling.
  const flops = 4 * batch * layers * heads * width * activeLength;
  return { length: totalLength, activeLength, capacity, mhaCapacity, flops, ratio: heads / kvHeads, intensity: capacity ? flops / capacity : 0 };
}

export function sharedAttention(scale = 2) {
  const keys = [[1, 0], [0, 1], [-1, 0]];
  const values = [[1, 0], [0, 1], [-1, 0]];
  const queries = [[scale, 0], [-scale, 0]];
  const logits = queries.map(q => keys.map(k => q[0] * k[0] + q[1] * k[1]));
  const weights = logits.map(row => {
    const maximum = Math.max(...row);
    const exps = row.map(x => Math.exp(x - maximum));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
  });
  const outputs = weights.map(row => [0, 1].map(d => row.reduce((sum, w, j) => sum + w * values[j][d], 0)));
  return { keys, values, queries, logits, weights, outputs };
}
