#!/usr/bin/env python3
"""Analytical KV storage ledger; no allocation or hardware measurement.

Assumptions: dense L=32,Hkv=8,D=128,B=1; FP16 scale and real offset per
group (4 bytes), no padding/allocator/temporary buffers, G=32,R=128.
K flushes a full R-token buffer; V retains a rolling R-token window.
"""
import argparse
import json
from pathlib import Path


def ledger(tokens, bits=2):
    per_token = 32 * 8 * 128
    group, residual = 32, 128
    k_fp = tokens % residual
    v_fp = min(tokens, residual)
    k_old, v_old = tokens - k_fp, tokens - v_fp
    fp16 = 2 * tokens * per_token * 2
    payload = (k_old + v_old) * per_token * bits // 8
    metadata = ((k_old // group) * per_token +
                v_old * (per_token // group)) * 4
    fp_window = (k_fp + v_fp) * per_token * 2
    total = payload + metadata + fp_window
    return {
        "tokens": tokens, "payload_bits": bits,
        "key_fp_tokens": k_fp, "value_fp_tokens": v_fp,
        "fp16_baseline_bytes": fp16, "quantized_payload_bytes": payload,
        "scale_offset_bytes": metadata, "fp_window_bytes": fp_window,
        "modeled_total_bytes": total,
        "modeled_total_mib": total / 2**20,
        "baseline_mib": fp16 / 2**20,
        "modeled_compression_ratio": fp16 / total,
        "effective_bits_per_cached_scalar": 8 * total / (2 * tokens * per_token),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("memory_results.json"))
    args = parser.parse_args()
    results = {
        "scope": "Analytical byte counts; not process memory or measured speed",
        "assumptions": {"batch": 1, "layers": 32, "kv_heads": 8,
                        "head_dim": 128, "group": 32, "residual": 128,
                        "scale_bytes": 2, "offset_bytes": 2},
        "states": [ledger(s, b) for s in [64, 128, 8192, 8256] for b in [2, 4]],
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
