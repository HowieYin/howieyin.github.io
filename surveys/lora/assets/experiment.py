"""Real small-checkpoint LoRA diagnostic; not a reproduction of a named paper.

Run on frozen public Pythia-70M and WikiText2 inputs. All GEMM is float32.
"""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=20261119)
parser.add_argument("--steps", type=int, default=96)
parser.add_argument("--data-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(args.data_dir / "model", local_files_only=True)
model_revision = json.loads((args.data_dir / "model-meta.json").read_text())["sha"]

def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()

def corpus(split, count):
    text = "\n\n".join(pq.read_table(args.data_dir / f"{split}.parquet").column("text").to_pylist())
    tokens = tok(text, return_tensors="pt").input_ids[0, :count]
    return tokens.reshape(-1, 256)

train, test = corpus("train", 16384), corpus("test", 4096)

class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, fixed_a=False):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.multiplier = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device),
                              requires_grad=not fixed_a)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.multiplier * F.linear(F.linear(x, self.A), self.B)

    def delta(self):
        return self.multiplier * (self.B @ self.A)

    def merged_copy(self, times=1.0):
        layer = nn.Linear(self.base.in_features, self.base.out_features,
                          bias=self.base.bias is not None, device=self.base.weight.device,
                          dtype=self.base.weight.dtype)
        with torch.no_grad():
            layer.weight.copy_(self.base.weight + times * self.delta())
            if self.base.bias is not None:
                layer.bias.copy_(self.base.bias)
        return layer.requires_grad_(False)

def load(mode=None, rank=0):
    model = AutoModelForCausalLM.from_pretrained(
        args.data_dir / "model", local_files_only=True,
        attn_implementation="eager", dtype=torch.float32,
    ).to(DEVICE).eval().requires_grad_(False)
    targets = {}
    for name, layer in list(model.named_modules()):
        if isinstance(layer, nn.Linear) and name.startswith("gpt_neox.layers."):
            if mode == "full":
                layer.weight.requires_grad_(True)
            elif mode in {"lora", "fixed_a"}:
                parent, child = name.rsplit(".", 1)
                layer = LoRALinear(layer, rank, alpha=rank, fixed_a=mode == "fixed_a")
                setattr(model.get_submodule(parent), child, layer)
            targets[name] = layer
    return model, targets

@torch.no_grad()
def score(model):
    total = 0.0
    windows = []
    model.eval()
    for x in test:
        logits = model(x[None].to(DEVICE), use_cache=False).logits[0, :-1].float()
        loss = float(F.cross_entropy(logits, x[1:].to(DEVICE), reduction="sum").cpu())
        total += loss
        windows.append(loss / 255)
    nll = total / (len(test) * 255)
    return {"nll_nats": nll, "ppl": math.exp(nll), "targets": len(test) * 255,
            "window_nll": windows}

@torch.no_grad()
def probe(model):
    return model(test[0:1, :64].to(DEVICE), use_cache=False).logits.float().cpu()

def differences(a, b):
    return {"max_abs_logits": float((a - b).abs().max()),
            "mean_abs_logits": float((a - b).abs().mean()),
            "argmax_mismatches": int((a.argmax(-1) != b.argmax(-1)).sum()),
            "argmax_positions": a.shape[0] * a.shape[1]}

@torch.no_grad()
def timing(model):
    """Synchronized local prefill + first selection, and eight cached decode steps."""
    prompt = test[0:1, :64].to(DEVICE)
    samples = []
    for iteration in range(7):
        sync()
        start = time.perf_counter()
        out = model(prompt, use_cache=True)
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        sync()
        first = time.perf_counter()
        cache = out.past_key_values
        for _ in range(8):
            out = model(token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            token = out.logits[:, -1].argmax(-1, keepdim=True)
        sync()
        end = time.perf_counter()
        if iteration >= 2:
            samples.append({"prefill_first_selection_ms": (first - start) * 1000,
                            "decode_ms_per_step": (end - first) * 1000 / 8})
    return {"samples": samples,
            "mean_prefill_first_selection_ms": float(np.mean([x["prefill_first_selection_ms"] for x in samples])),
            "mean_decode_ms_per_step": float(np.mean([x["decode_ms_per_step"] for x in samples])),
            "boundary": "single-process MPS/CPU float32; batch1 prompt64; two warmups, five samples; no tokenizer/network/queue"}

result = {
    "identity": "Actual public checkpoint diagnostic; not named-method full reproduction",
    "model": "EleutherAI/pythia-70m-deduped", "model_revision": model_revision,
    "dataset": "Salesforce/wikitext, wikitext-2-raw-v1",
    "dataset_revision": json.loads((args.data_dir / "dataset-meta.json").read_text())["sha"],
    "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                    "python": platform.python_version(), "platform": platform.platform(),
                    "device": DEVICE, "dtype": "float32", "threads": 4},
    "protocol": {
        "seed": args.seed, "steps": args.steps, "batch": 1, "sequence_length": 256,
        "training": "first16384 train tokens, 64 disjoint windows; reshuffled order each pass",
        "training_order": "independent CPU torch.Generator seeded once per mode; identical window order across modes at each seed",
        "testing": "first4096 test tokens,16 disjoint windows,4080 next-token targets",
        "joining": "all source rows joined by two newlines; no other cleanup",
        "token_sha256": {name: hashlib.sha256(data.numpy().tobytes()).hexdigest()
                         for name, data in [("train", train), ("test", test)]},
        "target_scope": "24 Transformer Linear weights; no embeddings, head, biases or norms",
        "optimizer": "AdamW beta(0.9,0.999), eps1e-8, weight_decay0, clip1",
        "learning_rate": {"full": 2e-5, "lora": 1e-3, "fixed_a": 1e-3},
        "scaling": "alpha=rank, multiplier=1 at all ranks; not fixed-alpha rank sweep",
        "dropout": "model.eval throughout to isolate numerical behavior; gradients remain enabled",
        "comparison_limit": "one prescribed LR per family; no equal-budget hyperparameter search",
        "fixed_a_limit": "frozen A + ordinary AdamW; NOT the LoRA-FA paper's specialized optimizer",
        "speed_limit": "float32 GEMM; not quantized inference or multi-tenant serving",
        "merge_acceptance": "finite logits and test mean NLL difference <=1e-3 nats; report actual logits/argmax differences; float64 identity diagnostic separate",
    }, "runs": [],
}
output = args.output_dir / f"results-{args.seed}.json"
def save():
    output.write_text(json.dumps(result, indent=2))

model, targets = load()
result["parameters"] = sum(p.numel() for p in model.parameters())
result["target_parameters"] = sum(m.weight.numel() for m in targets.values())
result["target_names"] = list(targets)
reference = probe(model)
result["baseline"] = score(model)
del model, targets
save()
print("Baseline", args.seed, result["baseline"]["ppl"], flush=True)

for mode, rank in [("full", 0), ("lora", 4), ("lora", 16), ("lora", 64), ("fixed_a", 16)]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, targets = load(mode, rank)
    params = [p for p in model.parameters() if p.requires_grad]
    run = {"mode": mode, "rank": rank, "alpha": rank if rank else None,
           "trainable_parameters": sum(p.numel() for p in params),
           "initial_identity": differences(reference, probe(model))}
    if run["initial_identity"]["max_abs_logits"] > 1e-6:
        raise AssertionError("Fresh adapter changed base function")
    opt = torch.optim.AdamW(params, lr=result["protocol"]["learning_rate"][mode], weight_decay=0.0)
    loss_log = []
    order = []
    order_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    window_indices = []
    start = time.perf_counter()
    for step in range(args.steps):
        if not order:
            order = torch.randperm(len(train), generator=order_generator).tolist()
        window_index = order.pop()
        window_indices.append(window_index)
        x = train[window_index][None].to(DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(x, use_cache=False).logits[:, :-1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        loss.backward()
        if step == 0 and mode != "full":
            first = next(iter(targets.values()))
            run["first_step_gradients"] = {
                "A_norm": None if first.A.grad is None else float(first.A.grad.norm().cpu()),
                "B_norm": float(first.B.grad.norm().cpu()),
                "base_grad_is_none": first.base.weight.grad is None,
            }
            if mode == "lora" and run["first_step_gradients"]["A_norm"] != 0:
                raise AssertionError("B=0 should make initial A gradient zero")
            if run["first_step_gradients"]["B_norm"] == 0:
                raise AssertionError("Adapter never received gradient")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        loss_log.append(float(loss.detach().cpu()))
        if (step + 1) % 24 == 0:
            print(args.seed, mode, rank, step + 1, loss_log[-1], flush=True)
    sync()
    run["training_elapsed_seconds"] = time.perf_counter() - start
    run["training_losses"] = loss_log
    run["train_window_indices"] = window_indices
    run.update(score(model))
    trained = probe(model)
    if mode != "full":
        payload = {
            "metadata": {"model_revision": model_revision, "rank": rank, "alpha": rank,
                         "targets": list(targets), "dtype": "float32", "layout": "B(out,r) A(r,in)"},
            "adapters": {name: {"A": m.A.detach().cpu(), "B": m.B.detach().cpu()}
                         for name, m in targets.items()},
        }
        artifact = args.output_dir / f"adapter-{args.seed}-{mode}-{rank}.pt"
        torch.save(payload, artifact)
        decoded = torch.load(artifact, map_location="cpu", weights_only=True)
        run["serialized"] = {"bytes": artifact.stat().st_size,
                             "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
        reloaded, fresh = load(mode, rank)
        if decoded["metadata"]["model_revision"] != model_revision:
            raise AssertionError("Base revision mismatch")
        with torch.no_grad():
            for name, m in fresh.items():
                m.A.copy_(decoded["adapters"][name]["A"].to(DEVICE))
                m.B.copy_(decoded["adapters"][name]["B"].to(DEVICE))
        run["reload"] = differences(trained, probe(reloaded))
        if run["reload"]["max_abs_logits"] != 0:
            raise AssertionError("Serialized adapter changed function")
        del reloaded, fresh
        # Merge actual trained adapters, then inject a double merge into the same base.
        for name, m in targets.items():
            parent, child = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), child, m.merged_copy())
        merged = probe(model)
        run["merged"] = differences(trained, merged)
        run["merged_score"] = score(model)
        run["merged_nll_difference"] = run["merged_score"]["nll_nats"] - run["nll_nats"]
        if not torch.isfinite(merged).all() or abs(run["merged_nll_difference"]) > 1e-3:
            (args.output_dir / f"merge-failure-{args.seed}-{mode}-{rank}.json").write_text(json.dumps(run, indent=2))
            raise AssertionError("Merged task NLL exceeds tolerance; inspect saved raw evidence")
        for name, m in targets.items():
            model.get_submodule(name).weight.add_(m.delta().detach())
        run["double_merge_injection"] = differences(trained, probe(model))
        # Restore the single merge exactly, recovering output logits rather than only not crashing.
        for name, m in targets.items():
            model.get_submodule(name).weight.copy_(m.merged_copy().weight)
        run["restore_after_injection"] = differences(merged, probe(model))
        if run["restore_after_injection"]["max_abs_logits"] != 0:
            raise AssertionError("Restoration did not recover the original merged behavior")
        first = next(iter(targets.values()))
        singular = torch.linalg.svdvals(first.delta().detach().cpu())
        p = singular / singular.sum()
        positive = p[p > 0]
        run["first_layer_delta_spectrum"] = {
            "layer": next(iter(targets)), "nominal_rank": rank,
            "entropy_effective_rank": float(torch.exp(-(positive * positive.log()).sum())),
            "stable_rank": float(singular.square().sum() / singular[0].square()),
            "top_singular_values": singular[:rank + 4].tolist(),
            "boundary": "final delta of one selected layer; not per-step tangent effective rank",
        }
        if mode == "lora" and rank == 16 and args.seed == 20261119:
            run["local_timing_merged"] = timing(model)
            for name, m in targets.items():
                parent, child = name.rsplit(".", 1)
                setattr(model.get_submodule(parent), child, m)
            run["local_timing_unmerged"] = timing(model)
            # Wrong alpha affects actual live adapter function.
            for m in targets.values():
                m.multiplier *= 2
            run["wrong_alpha_injection"] = differences(trained, probe(model))
            for m in targets.values():
                m.multiplier /= 2
            run["restore_alpha"] = differences(trained, probe(model))
            if run["restore_alpha"]["max_abs_logits"] != 0:
                raise AssertionError("Alpha restoration changed output")
    result["runs"].append(run)
    save()
    print("Completed", args.seed, mode, rank, run["ppl"], flush=True)
    del opt, params, model, targets
    if DEVICE == "mps":
        torch.mps.empty_cache()
print("Finished", output, flush=True)
