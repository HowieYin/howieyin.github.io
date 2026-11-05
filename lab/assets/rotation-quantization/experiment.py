#!/usr/bin/env python3
"""CPU/NumPy teaching experiment, not a model or accelerator benchmark.

Run: python3 experiment.py --output results.json
All math uses float64. Quantization is simulated by round/dequantize; no INT4
packing, integer GEMM, clipping search, GPTQ, or learned rotation is performed.
"""
import argparse
import json
import platform
from pathlib import Path

import numpy as np


def hadamard(n):
    if n < 1 or n & (n - 1):
        raise ValueError("n must be a power of two")
    h = np.ones((1, 1), dtype=np.float64)
    while len(h) < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def quantize(x, axis=-1):
    """Symmetric [-7, 7] simulated INT4; scale shared along `axis`.

    The representable -8 code is unused. Zero vectors use scale 1.
    np.rint uses round-to-nearest, ties-to-even.
    """
    scale = np.max(np.abs(x), axis=axis, keepdims=True) / 7
    scale = np.where(scale == 0, 1.0, scale)
    return np.clip(np.rint(x / scale), -7, 7) * scale


def rel_l2(actual, reference):
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def rmsnorm(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def rope(n, position):
    """Row-vector convention: x @ P(position), adjacent coordinate pairs."""
    p = np.zeros((n, n), dtype=np.float64)
    for j in range(n // 2):
        angle = position * 10000.0 ** (-2.0 * j / n)
        c, s = np.cos(angle), np.sin(angle)
        p[2*j:2*j+2, 2*j:2*j+2] = [[c, -s], [s, c]]
    return p


def measure_case(x, w, r):
    y = x @ w
    xr, wr = x @ r, r.T @ w
    rows = {}
    for label, a, b, back in [
        ("original", x, w, np.eye(x.shape[-1])),
        ("hadamard", xr, wr, r.T),
    ]:
        aq = quantize(a)
        # Each output column has a separate weight scale.
        bq = quantize(b, axis=0)
        rows[label] = {
            "mean_row_max_abs": float(np.max(np.abs(a), axis=-1).mean()),
            "mean_peak_to_rms": float(
                (np.max(np.abs(a), axis=-1) /
                 np.sqrt(np.mean(a*a, axis=-1))).mean()),
            "input_reconstruction_rel_l2": rel_l2(aq @ back, x),
            "activation_only_output_rel_l2": rel_l2(aq @ b, y),
            "weight_and_activation_output_rel_l2": rel_l2(aq @ bq, y),
        }
    rows["full_precision_compensation_max_abs_error"] = float(
        np.max(np.abs(xr @ wr - y)))
    return rows


def run():
    rng = np.random.default_rng(20261105)
    n = 64
    r = hadamard(n)
    w = rng.normal(size=(n, 24)) / np.sqrt(n)
    outlier = rng.normal(scale=0.3, size=(8, n))
    outlier[:, 0] = 20.0
    # These rows lie exactly on the original symmetric INT4 grid.
    aligned = rng.integers(-7, 8, size=(8, n)).astype(np.float64)
    aligned[:, 0] = 7.0
    q = rng.normal(size=(3, n))
    k = rng.normal(size=(5, n))
    pm, pn = rope(n, 11), rope(n, 37)
    original_score = (q @ pm) @ (k @ pn).T
    post_score = (q @ pm @ r) @ (k @ pn @ r).T
    misplaced_score = (q @ r @ pm) @ (k @ r @ pn).T
    gamma = np.linspace(0.5, 1.5, n)
    result = {
        "scope": "Synthetic float64 CPU experiment; no full-model or speed claims",
        "seed": 20261105, "dimension": n,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "platform": platform.platform()},
        "quantizer": "symmetric [-7,7], per-row activation, per-column weight; ties-to-even; no clipping search",
        "invariance": {
            "orthogonality_max_abs_error": float(np.max(np.abs(r.T @ r - np.eye(n)))),
            "joint_qk_max_abs_error": float(np.max(np.abs((q @ r) @ (k @ r).T - q @ k.T))),
            "rmsnorm_without_gamma_max_abs_error": float(
                np.max(np.abs(rmsnorm(q @ r) - rmsnorm(q) @ r))),
            "rmsnorm_with_unfolded_gamma_rel_l2": rel_l2(
                rmsnorm(q @ r) * gamma, (rmsnorm(q) * gamma) @ r),
            "relu_wrong_commutation_rel_l2": rel_l2(
                np.maximum(q @ r, 0), np.maximum(q, 0) @ r),
            "post_rope_joint_rotation_max_abs_error": float(
                np.max(np.abs(post_score - original_score))),
            "pre_rope_unchanged_rope_wrong_rel_l2": rel_l2(
                misplaced_score, original_score),
        },
        "outlier_case": measure_case(outlier, w, r),
        "grid_aligned_counterexample": measure_case(aligned, w, r),
    }
    # Meaningful correctness conditions for this teaching construction.
    assert result["invariance"]["orthogonality_max_abs_error"] < 1e-12
    assert result["invariance"]["joint_qk_max_abs_error"] < 1e-12
    assert result["invariance"]["post_rope_joint_rotation_max_abs_error"] < 1e-12
    assert result["outlier_case"]["hadamard"]["activation_only_output_rel_l2"] < result["outlier_case"]["original"]["activation_only_output_rel_l2"]
    assert result["grid_aligned_counterexample"]["original"]["activation_only_output_rel_l2"] == 0
    assert result["grid_aligned_counterexample"]["hadamard"]["activation_only_output_rel_l2"] > 0
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
