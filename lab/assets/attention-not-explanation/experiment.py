"""Construct different simplex attention vectors sharing a fixed AV output."""
import json
import platform
import sys
from pathlib import Path

import numpy as np


def metrics(a, b, values):
    m = (a + b) / 2
    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log(x[mask] / y[mask])))
    return {
        "l1": float(np.sum(np.abs(a - b))),
        "tv": float(np.sum(np.abs(a - b)) / 2),
        "jsd_nats": (kl(a, m) + kl(b, m)) / 2,
        "output_a": (a @ values).tolist(), "output_b": (b @ values).tolist(),
        "output_max_abs": float(np.max(np.abs((a - b) @ values))),
    }


def equivalent_attention(a, values):
    # A row-vector delta must satisfy delta V = 0 and delta 1 = 0.
    c = np.column_stack([values, np.ones(len(a))])
    _, s, vh = np.linalg.svd(c.T, full_matrices=True)
    tol = max(c.shape) * np.finfo(float).eps * s[0]
    rank = int(np.sum(s > tol))
    if rank == len(a):
        return None, 0
    delta = vh[rank]
    if delta[np.argmax(np.abs(delta))] < 0:
        delta = -delta
    negative = delta < 0
    t_max = float(np.min(a[negative] / -delta[negative]))
    b = a + 0.95 * t_max * delta
    assert np.all(b > 0) and abs(b.sum() - 1) < 1e-12
    assert np.max(np.abs((b - a) @ values)) < 1e-12
    return b, len(a) - rank


def run():
    a = np.array([.49, .01, .01, .49])
    b = np.array([.01, .49, .49, .01])
    values = np.array([[0., 1.], [1., 0.], [1., 2.], [2., 1.]])
    exact = metrics(a, b, values)
    assert exact["output_max_abs"] < 1e-12
    assert np.allclose(exact["output_a"], [1., 1.])
    logits = np.log(b)
    softmax = np.exp(logits - logits.max())
    softmax /= softmax.sum()
    assert np.max(np.abs(softmax - b)) < 1e-12
    shuffled = a[[1, 0, 2, 3]]
    permutation = metrics(a, shuffled, values)
    assert permutation["output_max_abs"] > .4
    perturbations = []
    for eps in (0., .001, .01, .1):
        perturbed = values.copy()
        perturbed[0, 0] += eps
        m = metrics(a, b, perturbed)
        assert abs(m["output_max_abs"] - .48 * eps) < 1e-12
        perturbations.append({"epsilon": eps, **m})
    full_rank = metrics(a, b, np.eye(4))
    candidate, dimension = equivalent_attention(a, np.eye(4))
    assert candidate is None and dimension == 0
    rng = np.random.default_rng(20261109)
    random_values = rng.normal(size=(12, 3))
    random_a = rng.dirichlet(np.ones(12))
    random_b, nullity = equivalent_attention(random_a, random_values)
    random_case = metrics(random_a, random_b, random_values)
    return {
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.system(), "dtype": "float64"},
        "seed": 20261109, "values": values.tolist(),
        "attention_a": a.tolist(), "attention_b": b.tolist(),
        "exact": exact, "arbitrary_permutation": permutation,
        "perturbations": perturbations, "full_rank_identity": full_rank,
        "full_rank_nullity": dimension,
        "random_svd": {"n": 12, "dv": 3, "nullity": nullity,
                       "attention_a": random_a.tolist(), "attention_b": random_b.tolist(),
                       **random_case},
        "scope": "Fixed-value single-head interventions, no trained LLM or VLM evaluation",
    }


if __name__ == "__main__":
    result = run()
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("results.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
