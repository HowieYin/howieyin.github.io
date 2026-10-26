import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
ROOT = SCRIPT_DIRECTORY.parent
official_path = ROOT / "sources" / "official-vrope.py"
if not official_path.exists():
    official_path = SCRIPT_DIRECTORY / "official-vrope.py"
specification = importlib.util.spec_from_file_location("official_vrope", official_path)
official = importlib.util.module_from_spec(specification)
specification.loader.exec_module(official)


def paper_positions(frames, height, width, start, pairs):
    coordinates = []
    stride = height + width - 1
    for frame in range(frames):
        for row in range(height):
            for column in range(width):
                directions = [
                    column + row,
                    column - row + height - 1,
                    -column - row + height + width - 2,
                    -column + row + width - 1,
                ]
                coordinates.append([directions[pair % 4] + start + frame * stride for pair in range(pairs)])
    return np.array(coordinates)


def rotate(values, positions, frequencies):
    complex_values = values[0::2] + 1j * values[1::2]
    return complex_values * np.exp(1j * positions * frequencies)


reports = []
for frames, height, width in [(2, 3, 4), (16, 8, 8), (1, 1, 5), (2, 4, 1), (1, 1, 1)]:
    start = 7
    pairs = 8
    implementation = official.VRoPE(pairs, 2)
    implementation.generate_1d_positions(start)
    actual = implementation.generate_nd_positions((frames, height, width)).numpy()
    expected = paper_positions(frames, height, width, start, pairs)
    permutation = [0, 3, 1, 2, 4, 7, 5, 6]
    assert np.array_equal(actual, expected[:, permutation])
    stride = height + width - 1
    assert actual.min() == start
    assert actual.max() == start + frames * stride - 1
    assert implementation.current_index == start + frames * stride
    assert implementation.total_token_len == start + frames * height * width
    following = implementation.generate_1d_positions(3).numpy()
    assert np.array_equal(following[:, 0], np.arange(start + frames * stride, start + frames * stride + 3))
    assert np.all(following == following[:, :1])
    for frame in range(frames):
        for row in range(height):
            for column in range(width):
                direct = expected[frame * height * width + row * width + column]
                opposite = expected[frame * height * width + (height - 1 - row) * width + width - 1 - column]
                assert direct[0] == opposite[2]
                assert direct[1] == opposite[3]
    reports.append({
        "shape": [frames, height, width], "tokens": frames * height * width,
        "position_span": frames * stride, "range": [int(actual.min()), int(actual.max())],
        "next_text": int(following[0, 0]), "matches_paper_with_permutation": True,
        "half_time_dim_received": 2, "half_time_dim_effective": implementation.half_time_dim,
    })

random = np.random.default_rng(250211664)
pairs = 8
query = random.normal(size=pairs * 2)
key = random.normal(size=pairs * 2)
frequencies = 10000 ** (-np.arange(pairs) / pairs)
query_position = np.arange(pairs) + 19
key_position = np.arange(pairs) + 3
direct_score = np.real(np.sum(rotate(query, query_position, frequencies) * np.conj(rotate(key, key_position, frequencies))))
relative_score = np.real(np.sum(
    (query[0::2] + 1j * query[1::2]) * np.conj(key[0::2] + 1j * key[1::2])
    * np.exp(1j * (query_position - key_position) * frequencies)
))
assert np.isclose(direct_score, relative_score, atol=1e-12)
assert np.isclose(np.linalg.norm(rotate(query, query_position, frequencies)), np.linalg.norm(query), atol=1e-12)

table2 = {
    "Vicuna7B": [[38.5, 47, 43.90, 41.66, 53.10, 35.92], [38.9, 47.37, 45.18, 40.69, 54.05, 40.71]],
    "Qwen2_1.5B": [[39, 51.15, 51.15, 46.63, 56.96, 48.50], [42.4, 51.76, 50.78, 47.79, 57.15, 49.90]],
    "Qwen2_7B": [[50.1, 54.87, 54.33, 49.36, 63.73, 57.14], [50.6, 57.81, 54.70, 50.48, 65.88, 58.60]],
    "Qwen3_8B_Table9": [[61, 64.96, 59.68, 60.81, 68.67, 56.41], [62.56, 65.36, 59.23, 61.48, 68.67, 57.07]],
}
statistics = {
    model: {"baseline_mean": np.mean(values[0]), "vrope_mean": np.mean(values[1]),
            "mean_gain": np.mean(np.array(values[1]) - values[0]),
            "declining_task_indices": np.flatnonzero(np.array(values[1]) < values[0]).tolist()}
    for model, values in table2.items()
}
result = {
    "environment": {"torch": torch.__version__, "device": "CPU", "scope": "mechanism and arithmetic only; not model accuracy or GPU benchmark"},
    "position_cases": reports, "rotation_relative_identity_error": abs(direct_score - relative_score),
    "single_frequency_nonmonotonic_counterexample": {"cos_pi": math.cos(math.pi), "cos_2pi": math.cos(2 * math.pi)},
    "table_means": statistics,
    "retrieval_gain_pp": 87.03 - 54.84, "retrieval_gain_relative_percent": (87.03 - 54.84) / 54.84 * 100,
    "text_to_video_ratio_vrope_vs_3d": 3.70e-4 / 5.12e-5,
}
output_directory = ROOT / "notes" if (ROOT / "sources").exists() else SCRIPT_DIRECTORY
output_directory.mkdir(exist_ok=True)
(output_directory / "mechanism-checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
