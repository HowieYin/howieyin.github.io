"""Real checkpoint continued training: CE, fake quantization and export checks.

This is a controlled small-model diagnostic, not a reproduction of a named QAT paper.
All matrix multiplication is float32; elapsed training time is not low-bit inference speed.
"""
from pathlib import Path
import hashlib, json, math, platform, time, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("seed", nargs="?", type=int, default=20261119)
parser.add_argument("--data-dir", type=Path, default=Path("/tmp/llm-ptq-survey-work"))
parser.add_argument("--output-dir", type=Path, default=Path("/tmp/llm-qat-survey-work"))
args = parser.parse_args()
B = args.output_dir
OLD = args.data_dir
B.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BITS, GROUP, STEPS = 3, 128, 128
SEED = args.seed
OUTPUT = B / f"experiment-results-{SEED}.json"
tok = AutoTokenizer.from_pretrained(OLD / "model", local_files_only=True)

def corpus(split, count):
    text = "\n\n".join(pq.read_table(OLD / f"{split}.parquet").column("text").to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[0, :count]
    return ids.reshape(-1, 256)

train = corpus("train", 16384)
test = corpus("test", 4096)
validation = corpus("validation", 4096) if (OLD / "validation.parquet").exists() else None

def qdq(weight, bits=BITS, group=GROUP):
    z = weight.reshape(weight.shape[0], -1, group)
    scale = z.abs().amax(-1, keepdim=True).clamp(min=1e-12) / (2 ** (bits - 1) - 1)
    codes = (z / scale).round().clamp(-(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1)
    return (codes * scale).reshape_as(weight), codes, scale

class FakeLinear(nn.Module):
    def __init__(self, layer, mode):
        super().__init__()
        self.weight = nn.Parameter(layer.weight.detach().clone(), requires_grad=mode in {"float", "full"})
        self.bias = nn.Parameter(layer.bias.detach().clone(), requires_grad=False) if layer.bias is not None else None
        self.mode = mode
        self.in_features, self.out_features = layer.in_features, layer.out_features
        _, codes, scale = qdq(self.weight)
        if mode == "scale":
            self.register_buffer("codes", codes)
            self.log_scale = nn.Parameter(scale.log())
        elif mode == "lora":
            self.A = nn.Parameter(torch.empty(8, self.in_features, device=DEVICE))
            self.B = nn.Parameter(torch.zeros(self.out_features, 8, device=DEVICE))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def merged(self):
        return self.weight + self.B @ self.A if self.mode == "lora" else self.weight

    def effective(self):
        if self.mode == "scale":
            return (self.codes * self.log_scale.exp()).reshape_as(self.weight)
        w = self.merged()
        if self.mode == "float":
            return w
        rounded = qdq(w)[0]
        return w + (rounded - w).detach()  # identity STE; no scale derivative

    def forward(self, x):
        return F.linear(x, self.effective(), self.bias)

def load(mode=None):
    model = AutoModelForCausalLM.from_pretrained(
        OLD / "model", local_files_only=True, attn_implementation="eager", dtype=torch.float32
    ).to(DEVICE).eval().requires_grad_(False)
    layers = {}
    for name, layer in list(model.named_modules()):
        if isinstance(layer, nn.Linear) and name.startswith("gpt_neox.layers."):
            if mode is not None:
                parent_name, child = name.rsplit(".", 1)
                layer = FakeLinear(layer, mode)
                setattr(model.get_submodule(parent_name), child, layer)
            layers[name] = layer
    return model, layers

@torch.no_grad()
def score(model, data=test):
    model.eval()
    total, losses, logits = 0.0, [], None
    for i, x in enumerate(data):
        y = model(x[None].to(DEVICE), use_cache=False).logits[0, :-1].float()
        loss = float(F.cross_entropy(y, x[1:].to(DEVICE), reduction="sum").cpu())
        total += loss
        losses.append(loss / 255)
        if i == 0:
            logits = y[:32].cpu()
    nll = total / (len(data) * 255)
    return {"nll_nats": nll, "perplexity": math.exp(nll),
            "scored_tokens": len(data) * 255, "window_nll": losses}, logits

def save():
    OUTPUT.write_text(json.dumps(result, indent=2))

def pack_check(layers, mode):
    """Pack actual trained values into a 3-bit stream with float32 scale metadata."""
    payload, metadata, mismatches, max_error = 0, 0, 0, 0.0
    for layer in layers.values():
        weight = layer.effective().detach().cpu()
        _, codes, scale = qdq(weight)
        unsigned = (codes.reshape(-1).numpy().astype(np.int8) + 4).astype(np.uint8)
        bits = ((unsigned[:, None] >> np.arange(3, dtype=np.uint8)) & 1).reshape(-1)
        packed = np.packbits(bits, bitorder="little")
        unpacked_bits = np.unpackbits(packed, bitorder="little")[:bits.size].reshape(-1, 3)
        unsigned_again = unpacked_bits @ np.array([1, 2, 4], dtype=np.uint8)
        mismatches += int(np.count_nonzero(unsigned != unsigned_again))
        decoded = torch.from_numpy(unsigned_again.astype(np.int16) - 4).reshape_as(codes).float()
        recovered = (decoded * scale).reshape_as(weight)
        max_error = max(max_error, float((recovered - weight).abs().max()))
        payload += packed.nbytes
        metadata += scale.numel() * 4
    if mismatches or max_error > 1e-6:
        raise RuntimeError(f"Pack mismatch {mode}: {mismatches}, {max_error}")
    return {"encoding": "little-endian3-bit stream, signed code storage offset4; float32scale/group128",
            "actual_trained_values": True, "payload_bytes": payload, "scale_bytes": metadata,
            "code_mismatches": mismatches, "decoded_weight_max_abs": max_error,
            "bit_per_quantized_weight_with_scale": (payload + metadata) * 8 / result["quantized_parameters"]}

result = {
    "identity": "actual pretrained small-model diagnostic, not named-paper reproduction",
    "model": "EleutherAI/pythia-70m-deduped",
    "model_revision": json.loads((OLD / "model-meta.json").read_text())["sha"],
    "dataset": "Salesforce/wikitext, wikitext-2-raw-v1",
    "dataset_revision": json.loads((OLD / "dataset-meta.json").read_text())["sha"],
    "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                    "python": platform.python_version(), "platform": platform.platform(),
                    "device": DEVICE, "dtype": "float32", "threads": 4},
    "protocol": {
        "training": "first 16384 train tokens; 64 nonoverlapping length256 windows; 2 shuffled epochs",
        "testing": "first4096 test tokens; 16 nonoverlapping length256 windows; 4080 targets",
        "joining": "all original text rows joined by two newlines without other cleanup",
        "token_sha256": {s: hashlib.sha256(t.numpy().tobytes()).hexdigest()
                         for s, t in [("train", train), ("test", test)]},
        "bits": BITS, "group": GROUP, "rounding": "symmetric narrow range -3..3; ties to even",
        "trained": "only 24 Transformer linear weight matrices or corresponding adapters/scales",
        "excluded": "embedding, output head, norms, biases, activations and KV",
        "steps": STEPS, "batch_size": 1, "optimizer": "AdamW, betas=(0.9,0.999), eps1e-8",
        "learning_rates": {"float": 2e-5, "full": 2e-5, "scale": 1e-3, "lora": 1e-3},
        "weight_decay": 0.0, "gradient_clip": 1.0, "loss": "next-token cross entropy",
        "seed": SEED, "dropout": "model.eval throughout training to isolate deterministic quantization",
        "full_rule": "recompute absmax scale every forward; identity STE to shadow weight; scale gradients suppressed",
        "scale_rule": "freeze RTN codes; optimize positive per-group scales via log scale; codes never change",
        "lora_rule": "rank8 update is merged BEFORE Q/DQ; identity STE; multiplier1",
        "speed_boundary": "float32 Q/DQ and GEMM; neither packed low-bit training nor inference"
    }, "runs": []
}
model, layers = load()
result["parameters"] = sum(p.numel() for p in model.parameters())
result["quantized_parameters"] = sum(m.weight.numel() for m in layers.values())
fp, reference = score(model)
result["runs"].append({"mode": "FP32 checkpoint", **fp})
with torch.no_grad():
    for layer in layers.values():
        layer.weight.copy_(qdq(layer.weight)[0])
ptq, _ = score(model)
result["runs"].append({"mode": "W3 group128 RTN", **ptq})
del model, layers
save()
print("Baselines", fp["perplexity"], ptq["perplexity"], flush=True)

for mode in ["float", "full", "scale", "lora"]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model, layers = load(mode)
    params = [p for p in model.parameters() if p.requires_grad]
    rate = result["protocol"]["learning_rates"][mode]
    opt = torch.optim.AdamW(params, lr=rate, weight_decay=0)
    order = np.concatenate([np.random.permutation(len(train)) for _ in range(2)])
    history, started = [], time.monotonic()
    model.eval()
    for step, idx in enumerate(order):
        x = train[idx][None].to(DEVICE)
        opt.zero_grad(set_to_none=True)
        loss = model(x, labels=x, use_cache=False).loss
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).cpu())
        opt.step()
        if not math.isfinite(float(loss.detach().cpu())):
            raise RuntimeError(f"Nonfinite loss in {mode} step{step}")
        history.append({"step": step + 1, "window": int(idx),
                        "loss": float(loss.detach().cpu()), "grad_norm_before_clip": norm})
        if (step + 1) % 16 == 0:
            print(mode, step + 1, "loss", history[-1]["loss"],
                  "seconds", round(time.monotonic() - started, 1), flush=True)
    if DEVICE == "mps":
        torch.mps.synchronize()
    elapsed = time.monotonic() - started
    before, before_logits = score(model)
    if mode == "float":
        float_result = before
        with torch.no_grad():
            for layer in layers.values():
                layer.weight.copy_(qdq(layer.weight)[0])
        after, after_logits = score(model)
        export_error = None
    else:
        packed_result = pack_check(layers, mode)
        exported = {}
        with torch.no_grad():
            for name, layer in layers.items():
                exported[name] = layer.effective().detach().clone()
        for name, layer in layers.items():
            parent_name, child = name.rsplit(".", 1)
            plain = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None,
                              device=DEVICE, dtype=torch.float32)
            with torch.no_grad():
                plain.weight.copy_(exported[name])
                if layer.bias is not None:
                    plain.bias.copy_(layer.bias)
            setattr(model.get_submodule(parent_name), child, plain)
        after, after_logits = score(model)
        export_error = float((before_logits - after_logits).abs().max())
        if export_error > 1e-5:
            raise RuntimeError(f"Export changed QAT output: {export_error}")
    row = {"mode": {"float": "float fine-tune then W3 RTN", "full": "W3 full linear-weight QAT",
                    "scale": "W3 frozen-code scale adaptation",
                    "lora": "W3 rank8 merged-weight QAT"}[mode],
           "trainable_parameters": sum(p.numel() for p in params), "learning_rate": rate,
           "training_seconds": elapsed, "history": history, **after,
           "fake_to_export_logits_max_abs": export_error}
    if mode != "float":
        row["actual_weight_packing"] = packed_result
    if mode == "float":
        row["float_before_export"] = float_result
    # Deployment mismatch: same QAT learned weights re-quantized with a coarser group.
    if mode == "full":
        with torch.no_grad():
            for name in layers:
                layer = model.get_submodule(name)
                layer.weight.copy_(qdq(layer.weight, group=256)[0])
        mismatch, _ = score(model)
        row["requantized_group256"] = mismatch
    result["runs"].append(row)
    save()
    print({k: v for k, v in row.items() if k not in {"history", "window_nll"}}, flush=True)
    del model, layers, params, opt
    if DEVICE == "mps":
        torch.mps.empty_cache()
save()
