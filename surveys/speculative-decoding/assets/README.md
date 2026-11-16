# Exact rejection-sampling teaching experiment

```bash
python3 rejection_identity.py --output results.json
```

Python standard library only. Executed on this Mac with Python 3.12.2 on 2026-11-16. All arithmetic uses `fractions.Fraction`; the equality check is exact rather than a Monte Carlo tolerance.

- Single-step identity: all 441 pairs of three-token distributions whose probabilities are multiples of 1/5, including zero probabilities and disjoint support.
- Conditional sequence identity: exhaustive probability distributions of all 81 length-four strings from a three-state Markov target, with draft lengths 1, 2 and 3. Correct speculative sampling has total variation distance 0 in every case.
- Deliberate fault one: on rejection, sample directly from the target instead of the positive residual. Single-step TV is 3/20; the length-four sequence TV is 18271857/100000000.
- Deliberate fault two: later verification rows use the pre-block history rather than the exact drafted prefix. Length-four TV is 1687/5000.

This is an educational finite-vocabulary experiment. It does not establish GPU latency, LLM quality, numerical equivalence of a serving engine, phone power, or the correctness of every tree/parallel proposal implementation.
