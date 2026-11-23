# Pretrained CNN layout and runtime diagnostic

Actual execution on 2026-11-23: Apple M3 Pro, macOS arm64, PyTorch 2.15 / torchvision 0.29.2.
No ImageNet labels were evaluated. This is not an accuracy benchmark or a QNN/HTP measurement.

Run `python experiment.py` with torch, torchvision and Pillow, after downloading the original
ResNet18 IMAGENET1K_V1 and MobileNetV2 IMAGENET1K_V2 weights linked in `results.json`.
The script verifies their complete SHA256 against the official filename hash prefixes.
The local execution interpreter was `/tmp/llm-ptq-survey-work/model-venv/bin/python`.

Inputs: 8 deterministic DreamBench++ files from
https://huggingface.co/datasets/yuangpeng/dreambench_plus .
The script names animal/00, animal/01, human/00, human/01, object/00, object/01,
style/00 and style/01 under `/tmp/image-consistency-survey-work/data/images/`.
Adjust this data root and the weight root for another machine; verify the hashes
in `results.json` before treating results as the same experiment.

Each model uses its own torchvision weight preprocessing, eval/inference mode,
FP32 CPU and 4 threads. Correct NCHW and channels_last cases retain the same logical
input. The deliberately wrong NHWC permutation followed by NCHW reshape does not.
Correct outputs are compared on all 8 images. Timing uses batch 1, 10 warmups,
50 repetitions per layout, then a separate instrumented profiler run.

Conv MACs exclude all linear, normalization, activation, input processing and
postprocessing operations. Profiler self times are from one additional execution,
not the 50-repeat median. Interleaved layout testing, thermal control, QNN,
quantization, labelled dataset accuracy and task-head latency were not performed.
