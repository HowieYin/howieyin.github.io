# Expert prefetch correctness properties

Run with Python 3 and NumPy:

```bash
python3 prefetch_properties.py
```

This runs actual CPU numerical MoE execution using seeded synthetic float64
weights, injects incorrect predicted expert sets, and checks demand fallback
against an independently calculated reference. It also executes small scheduling
and stale-generation state checks. The inputs are independent vectors, not
autoregressive text or routing traces from a published language model.

`results.json` is the saved actual output. Weight copies and cache counts are
logical CPU operations. They are not PCIe traffic measurements or GPU speedups.
No pretrained-model benchmark has been reproduced in this task.
