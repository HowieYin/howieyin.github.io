import json
import platform

import numpy as np


def stable_softmax(values):
    values = np.asarray(values, dtype=np.float64)
    exponentials = np.exp(values - values.max())
    return exponentials / exponentials.sum()


def online_state(values):
    maximum = -np.inf
    denominator = 0.0
    for value in values:
        updated_maximum = max(maximum, value)
        denominator = (
            denominator * np.exp(maximum - updated_maximum)
            + np.exp(value - updated_maximum)
        )
        maximum = updated_maximum
    return maximum, denominator


def merge_states(left, right):
    maximum = max(left[0], right[0])
    denominator = (
        left[1] * np.exp(left[0] - maximum)
        + right[1] * np.exp(right[0] - maximum)
    )
    return maximum, denominator


logits = np.array([0.0, 0.25, 0.5], dtype=np.float64)
reference = stable_softmax(logits)
shifts = []
for offset in [0, 256, 512, 1024]:
    rounded = (logits + offset).astype(np.float16).astype(np.float64)
    probabilities = stable_softmax(rounded)
    centered_before_cast = (logits + offset - (logits + offset).max()).astype(
        np.float16
    )
    restored = stable_softmax(centered_before_cast)
    shifts.append(
        {
            "offset": offset,
            "rounded_logits": rounded.tolist(),
            "probabilities": probabilities.tolist(),
            "maximum_probability_error": float(np.max(np.abs(probabilities - reference))),
            "argmax": int(probabilities.argmax()),
            "center_before_cast_error": float(np.max(np.abs(restored - reference))),
            "float64_shift_error": float(np.max(np.abs(stable_softmax(logits + offset) - reference))),
        }
    )
assert shifts[0]["argmax"] == 2
assert shifts[-1]["argmax"] == 0
assert all(row["center_before_cast_error"] == 0 for row in shifts)
assert all(row["float64_shift_error"] == 0 for row in shifts)

large = np.float16(2048)
one = np.float16(1)
left_sum = np.float16(np.float16(large + one) + one)
right_sum = np.float16(large + np.float16(one + one))
assert left_sum == 2048
assert right_sum == 2050

generator = np.random.default_rng(20261026)
online_checks = []
for length in [3, 17, 257, 1024]:
    values = generator.normal(0, 4, length)
    maximum, denominator = online_state(values)
    online = np.exp(values - maximum) / denominator
    split = length // 2
    merged = merge_states(online_state(values[:split]), online_state(values[split:]))
    merged_probabilities = np.exp(values - merged[0]) / merged[1]
    error = float(np.max(np.abs(online - stable_softmax(values))))
    merge_error = float(np.max(np.abs(merged_probabilities - stable_softmax(values))))
    assert error < 1e-12
    assert merge_error < 1e-12
    online_checks.append({"length": length, "online_error": error, "merge_error": merge_error})

result = {
    "environment": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "hardware_scope": "CPU only; no LLM or GPU benchmark",
    },
    "reference": reference.tolist(),
    "shifts": shifts,
    "non_associativity": {
        "(2048+1)+1": float(left_sum),
        "2048+(1+1)": float(right_sum),
    },
    "online_checks": online_checks,
}
print(json.dumps(result, indent=2, ensure_ascii=False))
