# Actual pretrained Pythia PTQ diagnostic

This is a real pretrained-model numerical experiment, not a benchmark reproduction of GPTQ, AWQ, or any low-bit inference kernel.

The model is `EleutherAI/pythia-70m-deduped` revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`. The data is `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Frozen original model/data file identities are in `model-files.json` and `dataset-meta.json`.

- Model parameters: 70,426,624; quantized block Linear weights: 18,874,368 across24 modules.
- Other parameters, activations and cache remain FP32. Every quantized matrix is dequantized to FP32 before inference. `use_cache=False`, eager attention.
- Original text rows are joined with two newlines. Calibration uses first2048 train tokens, eight disjoint256-token windows. Evaluation uses first4096 test tokens, sixteen disjoint256-token windows,4080 next-token targets.
- Each window resets positions and has its own visible prefix; no extra BOS token is inserted.
- Symmetric narrow-range codes: -7..7 for4bit, -127..127 for8bit; ties-to-even rounding, no affine zero point. Scales use maxima per tensor, output row or128-input group.
- Calibrated clipping selects each row/group ratio in `{1,.99,.97,.95,.90,.85,.80}` using a diagonal activation-energy weighted weight-MSE proxy. It is not AWQ scaling or GPTQ covariance compensation.
- Environment: Python3.12.2, PyTorch2.9.1, Transformers4.57.6, macOS arm64, MPS FP32; seed20261119. CPU spot check independently tests only510 targets.
- Results are in `results.json`, with layer-specific error/zero fractions, NLL, PPL, token hashes, restoration checks and real-weight/real-activation format faults.

Download all nine companion files from the article's reproduction links into one directory. `prepare_inputs.py` downloads only the frozen public model and raw splits and checks each SHA256 in `model-files.json` and `dataset-files.json`. The numerical experiment reads local inputs and writes `experiment-results.json` to the selected data directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_inputs.py --data-dir ./inputs
python experiment.py --data-dir ./inputs
```

The default data directory is `./inputs`; use `--data-dir` to select another directory. Dependencies and the exact MPS environment are recorded above; CPU runs are distinct environments and may differ in the last digits. `prepare_inputs.py` retrieves and verifies inputs.

The verified run shows: finer W4 grouping improves PPL substantially; W4g128 RTN PPL125.921 is better than W8per-tensor190.348 for this checkpoint/protocol. Clipping lowers its calibration proxy relative-L2 from0.066414 to0.050379 but worsens test PPL125.921→133.040. Restoring original weights exactly restores baseline NLL and sampled logits.

Format injection uses a real32×128 weight slice. Payload is2048bytes versus16384FP32bytes, with scales additional. Signed codes are stored as `q+8`; this offset is a storage encoding, not an affine quantizer zero point. Correct unpack has0 code mismatches. Offset7 or transposed reinterpretation corrupts real-activation output. No native packed-kernel speed, TTFT/TPOT, QNN/HTP or GPU memory benchmark was measured.
