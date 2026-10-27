"""Independent FP64 checks of the actual browser teaching model, not a GPU benchmark."""
from pathlib import Path
import json
import subprocess
import numpy as np

directory = Path(__file__).resolve().parent
root = directory.parent
result = subprocess.run(
    ["node", str(directory / "verify_core.mjs")],
    capture_output=True, text=True, check=True
)
actual = json.loads(result.stdout)
errors = []
for case in actual["forward"]:
    scores = np.array(case["scores"], dtype=float)
    values = np.array(case["values"], dtype=float)
    finite = np.isfinite(scores)
    if finite.any():
        weight = np.exp(scores[finite] - scores[finite].max())
        expected = float(weight @ values[finite] / weight.sum())
    else:
        expected = 0.0
    error = abs(case["output"] - expected)
    errors.append(error)
    assert error < 1e-10, (case, expected)

Q, K, V, dO = [np.array(actual[name]) for name in ["Q", "K", "V", "dO"]]
mask = np.array(actual["mask"])
dropout = np.array(actual["dropout"])
scale = actual["scale"]

def loss(q, k, v):
    scores = scale * q @ k.T
    scores = np.where(mask, scores, -np.inf)
    rowmax = np.max(scores, axis=1, keepdims=True)
    safe = np.where(np.isfinite(rowmax), rowmax, 0)
    weights = np.exp(scores - safe)
    denominator = weights.sum(axis=1, keepdims=True)
    probability = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator != 0)
    return float(((probability * dropout) @ v * dO).sum())

gradient_errors = {}
for name, tensor in [("Q", Q), ("K", K), ("V", V)]:
    reference = np.zeros_like(tensor)
    for index in np.ndindex(tensor.shape):
        old = tensor[index]
        tensor[index] = old + 1e-5
        upper = loss(Q, K, V)
        tensor[index] = old - 1e-5
        lower = loss(Q, K, V)
        tensor[index] = old
        reference[index] = (upper - lower) / 2e-5
    error = float(np.max(np.abs(reference - np.array(actual["d" + name]))))
    assert error < 1e-8, (name, error)
    gradient_errors[name] = error

assert actual["cost"]["singleQuadraticBytes"] == 16 * 8 * 4096 ** 2 * 2
assert actual["cost"]["qkvoBytes"] == 4 * 16 * 8 * 4096 * 64 * 2
assert actual["cost"]["kvBytes"] == 2 * 16 * 32 * 4096 * 8 * 64 * 2
assert actual["ioDoublingRatio"] == 4
assert abs(actual["toyCorrect"] - 34.92652734585769) < 1e-10
report = {
    "date": "2026-10-27",
    "scope": "Actual JavaScript teaching functions; independent NumPy stable-softmax and finite-difference references",
    "forward_cases": len(errors),
    "max_forward_abs_error": max(errors),
    "gradient_max_abs_error": gradient_errors,
    "gradient_conditions": "scaled scores, a partially masked row, an all-masked row, fixed replayed dropout factors",
    "storage_formula_checks": "passed",
    "fixed_hardware_io_doubling_ratio": actual["ioDoublingRatio"],
    "tolerances": {"forward": 1e-10, "gradient": 1e-8},
    "not_verified": "CUDA speed, mixed-precision kernel accuracy, training convergence, production backend support",
}
destination = root / "notes/mechanism-results.json" if directory.name == "sources" else Path.cwd() / "mechanism-results.json"
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
