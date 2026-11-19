# Actual Pythia-70M QAT diagnostic

This is an actual pretrained checkpoint and WikiText2 experiment, not a reproduction of a named QAT paper and not low-bit GEMM performance testing.

Model: `EleutherAI/pythia-70m-deduped`, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`. Dataset: `Salesforce/wikitext`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`, config `wikitext-2-raw-v1`. `model-files.json` and `dataset-files.json` bind downloads to the exact SHA256 inputs.

Only 24 Transformer Linear matrices, 18,874,368 of 70,426,624 parameters, are quantized. Other parameters, activations and KV remain float32. Group128 runs along each output row; W3 uses symmetric narrow-range codes −3…3, ties-to-even, absmax/3. Scales are detached/recomputed in full and merged-weight QAT; scale-only uses positive learned scales and fixed initial codes. LoRA rank8 is merged before Q/DQ on every forward. No weight-only or activation kernel is actually low bit.

Train: first 16,384 train tokens, two independently shuffled epochs of 64 non-overlapping 256-token windows, batch1, 128 optimizer updates. Test: first 4,096 test tokens, 16 windows, 4,080 scored targets. Join original text rows with two newlines, no extra cleaning. Model dropout disabled. CE; AdamW, weight decay0, clip norm1.0; full/float LR2e-5, scale/LoRA LR1e-3. Learning rates differ: this is a mechanism diagnostic, not a fully tuned fair method ranking.

The original runs used Python3.12.2, PyTorch2.9.1, Transformers4.57.6, float32/eager attention and MPS on macOS. CPU replay is supported but may differ numerically and be slow. Three seeds 20261119/20/21 completed. Means and **sample SD of PPL** are not confidence intervals. FP32 and RTN baselines are deterministic repeated readouts, not three independent runs.

All three QAT variants and all three seeds passed sampled Q/DQ-export logits checks (max absolute difference0). Additional actual packing was added only for seeds20261120/21: each variant packed codes derived from trained effective weights, loaded FP32 scales and decoded with zero code mismatch. Full-weight/rank8 weight errors are0; scale-only max absolute weight error is1.1920928955078125e-7 on both seeds. `pack_check` recomputes absmax scale and Q/DQ on CPU effective weights instead of directly serializing learned log_scale, adding a scale reconstruction/rounding step. No unpacked-model logits/PPL was measured; the zero-logits Q/DQ check freezes effective weights separately. The total for all24 quantized matrices is7,077,888 payload bytes +589,824 scales =3.25bits per quantized weight. This excludes unquantized weights, tensor indices and file container. No TTFT/TPOT is measured. Requantizing already-exported full-QAT effective g128 weights to group256 worsens PPL on all three seeds; this does not start from unrounded shadow weights.

Run from a directory with Python3.12 and the requirements installed:

```bash
python prepare_inputs.py --data-dir /absolute/path/qat-inputs
python experiment.py 20261120 --data-dir /absolute/path/qat-inputs --output-dir /absolute/path/qat-results
```

To match the published extended packing checks, use seed20261120 or20261121. To reproduce all results, run each listed seed separately. `executed-experiment.py` preserves the extended source used for the later two seeds; the first seed ran before packing checks were added, and its exact earlier source was not separately frozen. `experiment.py` only replaces hard-coded input/output paths with argparse; training and scoring are unchanged. `run-provenance.json` records hashes of the extended and portable scripts. Replaying the current script for the first seed will also perform packing, but those new checks are not part of the published first-seed evidence. The original filenames and execution timestamps remain in `results-<seed>.json` and logs. Publication literature cutoff2026-11-19 is distinct from individual execution-log timestamps.

Input preparation uses the existing cache when its hash matches. Downloads require access to Hugging Face. It does not publish model weights, dataset contents or credentials. Results are diagnostic evidence for the stated slice, not full WikiText2 benchmark results.
