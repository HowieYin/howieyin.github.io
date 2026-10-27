from pathlib import Path
import json
import numpy as np

def rotate(projected, positions, base=10000.0):
    dimension = projected.shape[-1]
    if dimension % 2:
        raise ValueError("rotary dimension must be even")
    frequencies = base ** (-2 * np.arange(dimension // 2) / dimension)
    phase = np.asarray(positions)[..., None] * frequencies
    cosine, sine = np.cos(phase), np.sin(phase)
    first, second = projected[..., 0::2], projected[..., 1::2]
    result = np.empty_like(projected)
    result[..., 0::2] = first * cosine - second * sine
    result[..., 1::2] = first * sine + second * cosine
    return result

generator = np.random.default_rng(20261027)
errors = {"norm": 0.0, "relative_identity": 0.0, "shared_shift": 0.0, "prefill_decode": 0.0, "complex": 0.0}
for dimension in [2, 8, 64, 128]:
    query = generator.normal(size=(100, dimension))
    key = generator.normal(size=(100, dimension))
    query_position = generator.integers(0, 10000, size=100)
    key_position = generator.integers(0, 10000, size=100)
    shift = generator.integers(0, 10000, size=100)
    rotated_query = rotate(query, query_position)
    rotated_key = rotate(key, key_position)
    score = np.sum(rotated_query * rotated_key, axis=-1)
    errors["norm"] = max(errors["norm"], float(np.max(np.abs(np.linalg.norm(query, axis=-1) - np.linalg.norm(rotated_query, axis=-1)))))
    relative = np.sum(query * rotate(key, key_position - query_position), axis=-1)
    errors["relative_identity"] = max(errors["relative_identity"], float(np.max(np.abs(score - relative))))
    shifted = np.sum(rotate(query, query_position + shift) * rotate(key, key_position + shift), axis=-1)
    errors["shared_shift"] = max(errors["shared_shift"], float(np.max(np.abs(score - shifted))))
    complex_query = query[:, 0::2] + 1j * query[:, 1::2]
    complex_key = key[:, 0::2] + 1j * key[:, 1::2]
    frequencies = 10000 ** (-2 * np.arange(dimension // 2) / dimension)
    complex_score = np.real(np.sum(complex_query * complex_key.conj() * np.exp(1j * (query_position - key_position)[:, None] * frequencies), axis=-1))
    errors["complex"] = max(errors["complex"], float(np.max(np.abs(score - complex_score))))
    sequential_cache = np.stack([rotate(key[index], key_position[index]) for index in range(100)])
    errors["prefill_decode"] = max(errors["prefill_decode"], float(np.max(np.abs(sequential_cache - rotated_key))))
assert max(errors.values()) < 1e-9, errors
steps = np.array([200000, 12500, 120000, 80000, 10000, 30000])
lengths = np.array([512, 1536, 256, 128, 1536, 512])
batches = np.array([256, 256, 256, 512, 256, 512])
result = {"float64_reference_max_errors": errors, "random_cases": 400, "single_frequency_scores": {str(distance): float(np.cos(distance)) for distance in [3, 6]}, "curriculum_steps": int(steps.sum()), "max_position_slots": int((steps * lengths * batches).sum()), "kv_bytes_example": 2 * 32 * 8 * 4096 * 128 * 2, "scope": "Mechanism tests only; not paper training reproduction"}
output = Path(__file__).resolve().parents[1] / "notes"
output.mkdir(exist_ok=True)
(output / "mechanism-results.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
