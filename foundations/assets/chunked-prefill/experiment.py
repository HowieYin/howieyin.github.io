"""Chunked causal forward correctness and token-budget arithmetic, not GPU timing."""
import json
import platform
import sys
from pathlib import Path

import numpy as np


def weights(rng, width=8, layers=2):
    scale = 1 / np.sqrt(width)
    return [
        [rng.normal(size=(width, width)) * scale for _ in range(4)]
        + [rng.normal(size=(width, 16)) * scale,
           rng.normal(size=(16, width)) / 4]
        for _ in range(layers)
    ]


def forward(x, ws, chunk, isolated=False):
    n, width = x.shape
    heads, dim = 2, width // 2
    cache = [[np.empty((heads, 0, dim)), np.empty((heads, 0, dim))] for _ in ws]
    outputs = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        z = x[start:stop].copy()
        for layer, (wq, wk, wv, wo, w1, w2) in enumerate(ws):
            q, k, v = [(z @ w).reshape(-1, heads, dim).transpose(1, 0, 2)
                       for w in (wq, wk, wv)]
            keys = k if isolated else np.concatenate([cache[layer][0], k], axis=1)
            vals = v if isolated else np.concatenate([cache[layer][1], v], axis=1)
            key_positions = np.arange(start, stop) if isolated else np.arange(stop)
            query_positions = np.arange(start, stop)
            scores = q @ keys.transpose(0, 2, 1) / np.sqrt(dim)
            scores = np.where(key_positions[None, :] > query_positions[:, None],
                              -np.inf, scores)
            scores -= scores.max(axis=-1, keepdims=True)
            p = np.exp(scores)
            p /= p.sum(axis=-1, keepdims=True)
            attn = (p @ vals).transpose(1, 0, 2).reshape(-1, width)
            z = z + attn @ wo
            z = z + np.tanh(z @ w1) @ w2
            cache[layer][0] = np.concatenate([cache[layer][0], k], axis=1)
            cache[layer][1] = np.concatenate([cache[layer][1], v], axis=1)
        outputs.append(z)
    return np.concatenate(outputs), sum(a.nbytes for pair in cache for a in pair)


def budget_trace(budget, decodes=2, prompt=12):
    assert budget > decodes
    remaining, trace = prompt, []
    while remaining:
        prefill = min(budget - decodes, remaining)
        remaining -= prefill
        trace.append({"iteration": len(trace) + 1, "decode_tokens": decodes,
                      "prefill_tokens": prefill, "total_tokens": prefill + decodes,
                      "prompt_remaining": remaining})
    assert sum(r["prefill_tokens"] for r in trace) == prompt
    return trace


def run():
    rng = np.random.default_rng(20261116)
    x = rng.normal(size=(12, 8))
    ws = weights(rng)
    reference, reference_bytes = forward(x, ws, len(x))
    correctness = []
    for chunk in (1, 2, 3, 4, 5, 12):
        output, cache_bytes = forward(x, ws, chunk)
        error = float(np.max(np.abs(output - reference)))
        assert error < 1e-12 and cache_bytes == reference_bytes
        # Count visible causal pairs rather than the masked rectangle.
        pairs = sum(sum(range(start + 1, min(start + chunk, 12) + 1))
                    for start in range(0, 12, chunk))
        assert pairs == 78
        correctness.append({"chunk_size": chunk, "max_abs_error": error,
                            "final_kv_bytes": cache_bytes, "causal_pairs_per_head": pairs,
                            "iterations": (12 + chunk - 1) // chunk})
    wrong, _ = forward(x, ws, 4, isolated=True)
    wrong_error = float(np.max(np.abs(wrong - reference)))
    assert wrong_error > 1e-3
    assert np.max(np.abs(wrong[:4] - reference[:4])) < 1e-12
    return {
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.system(), "dtype": "float64"},
        "seed": 20261116, "tokens": 12, "hidden": 8, "layers": 2, "heads": 2,
        "architecture": "Simplified two-block causal attention, residual, tanh FFN; no normalization or RoPE",
        "correctness": correctness, "isolated_chunk4_max_abs": wrong_error,
        "budget_traces": [{"token_budget": b, "trace": budget_trace(b)} for b in (5, 8, 16)],
        "scope": "CPU numerical equivalence and analytic scheduled token counts, no hardware latency measurements",
    }


if __name__ == "__main__":
    result = run()
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("results.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
