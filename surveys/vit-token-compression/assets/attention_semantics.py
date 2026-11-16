"""CPU/float64 teaching experiment; no accuracy or hardware speed benchmark.

Run: python3 attention_semantics.py
The script writes results.json beside itself. Inputs, grouping and all outputs
are preserved so the limited mathematical claims can be independently checked.
"""
from pathlib import Path
import hashlib
import json
import math
import platform
import sys


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def attention(queries, keys, values, sizes=None):
    d = len(keys[0])
    sizes = sizes if sizes is not None else [1.0] * len(keys)
    output = []
    for q in queries:
        logits = [dot(q, k) / math.sqrt(d) + math.log(s)
                  for k, s in zip(keys, sizes)]
        peak = max(logits)
        mass = [math.exp(z - peak) for z in logits]
        probs = [m / sum(mass) for m in mass]
        output.append([sum(p * v[c] for p, v in zip(probs, values))
                       for c in range(len(values[0]))])
    return output


def average(rows):
    return [sum(row[c] for row in rows) / len(rows)
            for c in range(len(rows[0]))]


def merge(keys, values, groups):
    return ([average([keys[i] for i in group]) for group in groups],
            [average([values[i] for i in group]) for group in groups],
            [len(group) for group in groups])


def max_error(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def run():
    q = [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]
    k = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    v = [[2.0, 0.0], [2.0, 1.0], [2.0, -1.0], [0.0, 2.0]]
    groups = [[0, 1, 2], [3]]
    base = attention(q, k, v)
    mk, mv, size = merge(k, v, groups)
    corrected = attention(q, mk, mv, size)
    uncorrected = attention(q, mk, mv)
    exact_error = max_error(base, corrected)
    missing_mass_error = max_error(base, uncorrected)
    assert exact_error < 1e-12
    assert missing_mass_error > 0.1
    sweep = []
    for delta in [0.0, 0.1, 0.5, 1.0, 2.0]:
        perturbed = [row[:] for row in k]
        perturbed[1][1] = delta
        full = attention(q, perturbed, v)
        pk, pv, ps = merge(perturbed, v, groups)
        shortened = attention(q, pk, pv, ps)
        sweep.append({"key_delta": delta, "full_output": full,
                      "merged_output": shortened,
                      "max_abs_error": max_error(full, shortened)})
    assert sweep[-1]["max_abs_error"] > 0.1
    original = [2.0, 4.0, 8.0]
    first = (original[0] + original[1]) / 2
    hierarchical_unweighted = (first + original[2]) / 2
    hierarchical_size_weighted = (2 * first + original[2]) / 3
    true_mean = sum(original) / len(original)
    assert abs(hierarchical_size_weighted - true_mean) < 1e-12
    assert abs(hierarchical_unweighted - true_mean) > 0.1
    d = 768
    cost = []
    for n in [197, 577, 101, 50]:
        projection = 4 * n * d * d
        pairwise = 2 * n * n * d
        mlp = 8 * n * d * d
        cost.append({"tokens_including_cls": n, "width": d,
                     "projection_MACs": projection, "pairwise_MACs": pairwise,
                     "MLP_MACs": mlp, "total_MACs_per_block": projection + pairwise + mlp,
                     "pairwise_fraction": pairwise / (projection + pairwise + mlp)})
    result = {
        "identity": "Executed local CPU teaching experiment, not paper reproduction",
        "date": "2026-11-16", "python": sys.version, "platform": platform.platform(),
        "numeric_type": "Python double precision float; CPU",
        "limitations": ["Fixed queries and groups; one attention operator only",
                        "No positional update, layer norm, MLP, training or model accuracy",
                        "Analytical MAC ledger excludes matching/re-layout/runtime overhead",
                        "No GPU/NPU latency, TTFT or TPOT was measured"],
        "input": {"queries": q, "keys": k, "values": v, "groups": groups},
        "exact_duplicate_keys": {"full_output": base, "merged_keys": mk,
                                 "merged_values": mv, "sizes": size,
                                 "size_corrected_output": corrected,
                                 "uncorrected_output": uncorrected,
                                 "corrected_max_abs_error": exact_error,
                                 "uncorrected_max_abs_error": missing_mass_error},
        "perturbation_sweep": sweep,
        "hierarchical_feature_mean": {"original": original, "true_mean": true_mean,
                                     "without_sizes": hierarchical_unweighted,
                                     "with_sizes": hierarchical_size_weighted},
        "analytic_cost_ledger": cost,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assertions_passed": 5
    }
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"exact_error": exact_error, "missing_mass_error": missing_mass_error,
                      "key_perturbation_errors": [(x["key_delta"], x["max_abs_error"]) for x in sweep],
                      "hierarchical_mean": result["hierarchical_feature_mean"],
                      "assertions": "5 passed", "results": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
