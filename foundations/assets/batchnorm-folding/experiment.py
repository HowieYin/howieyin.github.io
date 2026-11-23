"""Train a real digits Conv-BN classifier, then verify frozen-stat folding.

Run: python3 experiment.py --out results
Requires numpy, scikit-learn, torch. CPU only; no latency claims.
"""
import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils import fuse_conv_bn_eval


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.head = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return self.head(nn.functional.max_pool2d(x, 2).flatten(1))


def manual_fold(conv, bn, eps=None):
    assert not conv.training and not bn.training
    assert bn.running_mean is not None and bn.running_var is not None
    fused = copy.deepcopy(conv)
    s = bn.weight / torch.sqrt(bn.running_var + (bn.eps if eps is None else eps))
    bias = torch.zeros_like(s) if conv.bias is None else conv.bias
    fused.weight = nn.Parameter(conv.weight * s[:, None, None, None])
    fused.bias = nn.Parameter(bn.bias + s * (bias - bn.running_mean))
    return fused


def folded_model(model, official=False, eps=None):
    result = copy.deepcopy(model).eval()
    for i in (1, 2):
        conv, bn = getattr(result, f"conv{i}"), getattr(result, f"bn{i}")
        fused = fuse_conv_bn_eval(conv, bn) if official else manual_fold(conv, bn, eps)
        setattr(result, f"conv{i}", fused)
        setattr(result, f"bn{i}", nn.Identity())
    return result


def metrics(y, pred):
    cm = confusion_matrix(y, pred, labels=np.arange(10))
    per_class = []
    for c in range(10):
        tp = int(cm[c, c])
        fn = int(cm[c].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        per_class.append(dict(label=c, TP=tp, FP=fp, TN=tn, FN=fn,
                              precision=p, recall=r, f1=2*p*r/max(p+r, 1e-30)))
    return dict(correct=int(np.trace(cm)), total=int(cm.sum()),
                accuracy=float(np.trace(cm)/cm.sum()), confusion_matrix=cm.tolist(),
                per_class=per_class)


def evaluate(base, candidate, x, y):
    with torch.no_grad():
        a, b = base(x), candidate(x)
    report = metrics(y.numpy(), b.argmax(1).numpy())
    report.update(max_abs_logit_error=float((a-b).abs().max()),
                  prediction_changes=int((a.argmax(1) != b.argmax(1)).sum()))
    return report


def run(out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)
    data = load_digits()
    raw = data.images.astype(np.float32)
    x = torch.tensor(raw[:, None] / 16)
    y = torch.tensor(data.target, dtype=torch.long)
    train, test = train_test_split(np.arange(len(y)), test_size=0.25,
                                  random_state=42, stratify=data.target)
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    history = []
    for epoch in range(25):
        model.train()
        perm = torch.randperm(len(train))
        total = 0.0
        for batch in perm.split(64):
            ids = train[batch.numpy()]
            loss = nn.functional.cross_entropy(model(x[ids]), y[ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        history.append(total / len(train))
    model.eval()
    xt, yt = x[test], y[test]
    torch.save(model.state_dict(), out / "model.pt")
    reports = {}
    for dtype in (torch.float32, torch.float64):
        base = copy.deepcopy(model).to(dtype).eval()
        manual = folded_model(base)
        official = folded_model(base, official=True)
        reports[str(dtype)] = evaluate(base, manual, xt.to(dtype), yt)
        with torch.no_grad():
            reports[str(dtype)]["manual_official_max_param_error"] = max(
                float((a-b).abs().max()) for a, b in zip(manual.parameters(), official.parameters()))
        assert reports[str(dtype)]["prediction_changes"] == 0
        assert reports[str(dtype)]["max_abs_logit_error"] < (1e-4 if dtype == torch.float32 else 1e-10)
    wrong_eps = folded_model(model, eps=1e-3)
    reports["wrong_eps_1e-3"] = evaluate(model, wrong_eps, xt, yt)
    wrong_stats = copy.deepcopy(model)
    for bn in (wrong_stats.bn1, wrong_stats.bn2):
        bn.running_mean.zero_()
        bn.running_var.fill_(1)
    reports["reset_running_stats"] = evaluate(model, wrong_stats, xt, yt)
    # Deliberately exercise zero and negative gamma using a trained convolution.
    special = copy.deepcopy(model)
    with torch.no_grad():
        special.bn1.weight[0] = 0
        special.bn1.weight[1] = -special.bn1.weight[1].abs()
    reports["zero_negative_gamma"] = evaluate(special, folded_model(special), xt, yt)
    assert reports["zero_negative_gamma"]["max_abs_logit_error"] < 1e-4
    # First test digit 0 remains the target; companions are other real test images.
    target_id = int(test[np.flatnonzero(data.target[test] == 0)[0]])
    companions = {}
    for name, classes in (("0_1", (0, 1)), ("8_9", (8, 9))):
        candidates = [int(i) for i in test if data.target[i] in classes and i != target_id][:31]
        companions[name] = torch.cat([x[target_id:target_id+1], x[candidates]])
    dependency = {}
    for training in (True, False):
        outputs = []
        for batch in companions.values():
            clone = copy.deepcopy(model)
            clone.train(training)
            with torch.no_grad():
                z = clone.bn1(clone.conv1(batch))[0]
                logits = clone(batch)[0]
            outputs.append((z, logits))
        dependency["train" if training else "eval"] = dict(
            first_bn_target_max_difference=float((outputs[0][0]-outputs[1][0]).abs().max()),
            target_logit_max_difference=float((outputs[0][1]-outputs[1][1]).abs().max()),
            target_predictions=[int(item[1].argmax()) for item in outputs])
    # Quantize folded convolution weights only: float activations and float biases.
    quantized = {}
    fused = folded_model(model)
    for grouping in ("per_tensor", "per_output_channel"):
        candidate = copy.deepcopy(fused)
        ranges = []
        for conv in (candidate.conv1, candidate.conv2):
            w = conv.weight.detach()
            dims = () if grouping == "per_tensor" else (1, 2, 3)
            scale = (w.abs().max() if not dims else w.abs().amax(dim=dims, keepdim=True))/127
            scale = scale.clamp_min(torch.finfo(w.dtype).tiny)
            conv.weight = nn.Parameter(torch.round(w / scale).clamp(-127, 127)*scale)
            ranges.append(dict(min_scale=float(scale.min()), max_scale=float(scale.max())))
        quantized[grouping] = evaluate(model, candidate, xt, yt)
        quantized[grouping]["weight_scales"] = ranges
    with torch.no_grad():
        baseline = metrics(yt.numpy(), model(xt).argmax(1).numpy())
    # Validate the article's training backward formula on actual trained activations.
    check = copy.deepcopy(model).train()
    z = check.conv1(x[train[:32]])
    normalized = check.bn1(z)
    hidden = torch.relu(check.bn2(check.conv2(torch.relu(normalized))))
    logits = check.head(nn.functional.max_pool2d(hidden, 2).flatten(1))
    loss = nn.functional.cross_entropy(logits, y[train[:32]])
    gy, gz = torch.autograd.grad(loss, (normalized, z))
    dims = (0, 2, 3)
    mean = z.mean(dims, keepdim=True)
    variance = z.var(dims, correction=0, keepdim=True)
    zhat = (z-mean) / torch.sqrt(variance+check.bn1.eps)
    expected = check.bn1.weight[None, :, None, None] / torch.sqrt(variance+check.bn1.eps) * (
        gy-gy.mean(dims, keepdim=True)-zhat*(gy*zhat).mean(dims, keepdim=True))
    backward_error = float((gz-expected).abs().max().detach())
    assert backward_error < 1e-6
    result = dict(
        config=dict(seed=17, split_seed=42, epochs=25, batch_size=64, lr=0.003,
                    train=len(train), test=len(test), dataset="sklearn load_digits; 8x8; raw/16",
                    input_sha256=hashlib.sha256(raw.tobytes()+data.target.tobytes()).hexdigest(),
                    torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__,
                    python=platform.python_version(), device="CPU", threads=1),
        baseline=baseline, train_loss=history, folding=reports, batch_dependency=dependency,
        backward_formula_max_abs_error=backward_error,
        weight_only_int8=quantized, target_id=target_id,
        bn1_gamma=model.bn1.weight.detach().tolist(),
        bn1_running_var=model.bn1.running_var.tolist(),
        bn1_fold_scale=(model.bn1.weight / torch.sqrt(model.bn1.running_var+model.bn1.eps)).tolist(),
        split=dict(train=train.tolist(), test=test.tolist()))
    (out/"results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("config", "batch_dependency")}, indent=2), flush=True)
    print("baseline", baseline["accuracy"], "fold errors",
          {k: v["max_abs_logit_error"] for k, v in reports.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.out)
