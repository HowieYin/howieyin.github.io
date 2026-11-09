"""NumPy TP algebra checks and a ring send-byte ledger; no GPU timing."""
import json
import platform
import sys
from pathlib import Path

import numpy as np


def run():
    rng = np.random.default_rng(20261109)
    x = rng.normal(size=(6, 8))
    w1 = rng.normal(size=(8, 16))
    w2 = rng.normal(size=(16, 8))
    b1 = rng.normal(size=16)
    b2 = rng.normal(size=8)
    dy = rng.normal(size=(6, 8))
    a = x @ w1 + b1
    z = np.tanh(a)
    y = z @ w2 + b2
    da = (dy @ w2.T) * (1 - z * z)
    dx = da @ w1.T
    dw1 = x.T @ da
    dw2 = z.T @ dy
    checks = []
    for p in (1, 2, 4):
        partials, dx_parts, gw1, gw2 = [], [], [], []
        for i in range(p):
            s = slice(i * 16 // p, (i + 1) * 16 // p)
            zi = np.tanh(x @ w1[:, s] + b1[s])
            partials.append(zi @ w2[s])
            dai = (dy @ w2[s].T) * (1 - zi * zi)
            dx_parts.append(dai @ w1[:, s].T)
            gw1.append(x.T @ dai)
            gw2.append(zi.T @ dy)
        errors = {
            "forward_max_abs": float(np.max(np.abs(sum(partials) + b2 - y))),
            "dx_max_abs": float(np.max(np.abs(sum(dx_parts) - dx))),
            "dw1_max_abs": float(np.max(np.abs(np.concatenate(gw1, axis=1) - dw1))),
            "dw2_max_abs": float(np.max(np.abs(np.concatenate(gw2, axis=0) - dw2))),
        }
        assert max(errors.values()) < 1e-12
        checks.append({"tp": p, **errors})

    # Gate and up projections must be partitioned with matching indices.
    gate = rng.normal(size=(8, 16))
    g = x @ gate
    swiglu = (g / (1 + np.exp(-g))) * (x @ w1)
    full_swiglu = swiglu @ w2
    pieces = []
    for i in range(4):
        s = slice(i * 4, (i + 1) * 4)
        local_gate = x @ gate[:, s]
        local_up = x @ w1[:, s]
        local_z = (local_gate / (1 + np.exp(-local_gate))) * local_up
        pieces.append(local_z @ w2[s])
    swiglu_error = float(np.max(np.abs(sum(pieces) - full_swiglu)))
    assert swiglu_error < 1e-12

    # MHA: each rank owns complete heads and all visible KV positions.
    q, k, v = [rng.normal(size=(4, 5, 2)) for _ in range(3)]
    score = q @ k.transpose(0, 2, 1) / np.sqrt(2)
    score = np.where(np.triu(np.ones((5, 5), bool), 1), -np.inf, score)
    score -= score.max(axis=-1, keepdims=True)
    prob = np.exp(score)
    prob /= prob.sum(axis=-1, keepdims=True)
    heads = prob @ v
    wo = rng.normal(size=(8, 8))
    full_attention = heads.transpose(1, 0, 2).reshape(5, 8) @ wo
    shard_attention = sum(heads[i] @ wo[2 * i:2 * i + 2] for i in range(4))
    attention_error = float(np.max(np.abs(full_attention - shard_attention)))
    assert attention_error < 1e-12
    # Independently check one input derivative of the full MLP loss.
    eps = 1e-6
    def loss(inp):
        return np.sum((np.tanh(inp @ w1 + b1) @ w2 + b2) * dy)
    xp, xm = x.copy(), x.copy()
    xp[0, 0] += eps
    xm[0, 0] -= eps
    finite_difference_error = float(abs((loss(xp) - loss(xm)) / (2 * eps) - dx[0, 0]))
    assert finite_difference_error < 1e-7

    ledger = []
    for tokens in (1, 32, 4096):
        m = tokens * 4096 * 2
        for p in (1, 2, 4, 8):
            # Per rank sent bytes, excluding receive counting and protocol overhead.
            sent = 0 if p == 1 else 2 * (p - 1) * m // p
            ledger.append({
                "tokens": tokens, "hidden": 4096, "bytes_per_element": 2,
                "tp": p, "payload_bytes": m,
                "sent_bytes_per_allreduce": sent,
                "sent_bytes_per_layer_forward": 2 * sent,
                "sent_bytes_32_layers_forward": 64 * sent,
                "ring_steps_per_allreduce": 0 if p == 1 else 2 * (p - 1),
            })
    return {
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.system(), "dtype": "float64"},
        "seed": 20261109, "mlp_shape": {"tokens": 6, "hidden": 8, "intermediate": 16},
        "mlp": checks, "swiglu_tp4_max_abs": swiglu_error,
        "attention_tp4_max_abs": attention_error,
        "dx00_finite_difference_abs": finite_difference_error, "ledger": ledger,
        "scope": "CPU algebra verification; analytical ring send bytes; no hardware latency benchmark",
    }


if __name__ == "__main__":
    result = run()
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("results.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
