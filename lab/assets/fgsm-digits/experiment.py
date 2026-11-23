"""Real digits, independent MLPs: FGSM, random-sign control, PGD, transfer.

Run: python3 experiment.py --out results
Requires numpy, scikit-learn, torch. Perturbations are in raw/16 coordinates.
"""
import argparse
import csv
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


def net():
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 128), nn.ReLU(),
                         nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))


def predict(model, x):
    with torch.no_grad():
        return model(x).argmax(1)


def metrics(y, pred, clean):
    cm = confusion_matrix(y.numpy(), pred.numpy(), labels=np.arange(10))
    per_class = []
    for c in range(10):
        tp = int(cm[c, c])
        fn = int(cm[c].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        p, r = tp/max(tp+fp, 1), tp/max(tp+fn, 1)
        per_class.append(dict(label=c, TP=tp, FP=fp, TN=tn, FN=fn,
                              precision=p, recall=r, f1=2*p*r/max(p+r, 1e-30)))
    correct = clean == y
    success = int((correct & (pred != y)).sum())
    return dict(correct=int((pred == y).sum()), total=len(y),
                accuracy=float((pred == y).float().mean()),
                clean_correct=int(correct.sum()), attacked_clean_correct_failures=success,
                conditional_attack_success=success/max(int(correct.sum()), 1),
                recovered_clean_errors=int(((~correct) & (pred == y)).sum()),
                macro_precision=float(np.mean([r["precision"] for r in per_class])),
                macro_recall=float(np.mean([r["recall"] for r in per_class])),
                macro_f1=float(np.mean([r["f1"] for r in per_class])),
                confusion_matrix=cm.tolist(), per_class=per_class)


def gradient(model, x, y):
    z = x.detach().clone().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(z), y)
    return torch.autograd.grad(loss, z)[0].detach()


def pgd(model, x, y, eps=0.1, step=0.01, steps=20, restarts=5):
    generator = torch.Generator().manual_seed(999)
    best = x.clone()
    with torch.no_grad():
        best_loss = nn.functional.cross_entropy(model(x), y, reduction="none")
    for _ in range(restarts):
        noise = (torch.rand(x.shape, generator=generator)*2-1)*eps
        adv = (x+noise).clamp(0, 1)
        for _ in range(steps):
            adv = adv + step*gradient(model, adv, y).sign()
            adv = torch.max(torch.min(adv, x+eps), x-eps).clamp(0, 1).detach()
            with torch.no_grad():
                losses = nn.functional.cross_entropy(model(adv), y, reduction="none")
                improve = losses > best_loss
                best[improve] = adv[improve]
                best_loss[improve] = losses[improve]
    return best


def run(out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    d = load_digits()
    raw = d.images.astype(np.float32)
    x = torch.tensor(raw[:, None]/16)
    y = torch.tensor(d.target, dtype=torch.long)
    train, test = train_test_split(np.arange(len(y)), test_size=0.25,
                                  random_state=42, stratify=d.target)
    xt, yt = x[test], y[test]
    epsilons = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3]
    seeds = [7, 19, 31]
    models, clean_preds, grads = [], [], []
    records, full, histories = [], {}, {}
    random_sign = (torch.randint(0, 2, xt.shape, generator=torch.Generator().manual_seed(555))*2-1).float()
    sample_data = {}
    for seed in seeds:
        torch.manual_seed(seed)
        model = net()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        history = []
        for epoch in range(40):
            model.train()
            perm = torch.randperm(len(train))
            total_loss = 0.0
            for batch in perm.split(64):
                ids = train[batch.numpy()]
                loss = nn.functional.cross_entropy(model(x[ids]), y[ids])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())*len(ids)
            history.append(total_loss/len(train))
        model.eval()
        torch.save(model.state_dict(), out/f"model-{seed}.pt")
        histories[str(seed)] = history
        clean = predict(model, xt)
        grad = gradient(model, xt, yt)
        models.append(model)
        clean_preds.append(clean)
        grads.append(grad)
        full[str(seed)] = {}
        for epsilon in epsilons:
            for attack, direction in (("fgsm", grad.sign()), ("random_sign", random_sign)):
                adv = (xt+epsilon*direction).clamp(0, 1)
                delta = adv-xt
                assert float(delta.abs().max()) <= epsilon+1e-6
                assert float(adv.min()) >= 0 and float(adv.max()) <= 1
                pred = predict(model, adv)
                m = metrics(yt, pred, clean)
                m.update(linf_max=float(delta.abs().max()),
                         mean_abs_delta=float(delta.abs().mean()))
                full[str(seed)][f"{attack}_{epsilon}"] = m
                records.append(dict(seed=seed, attack=attack, epsilon=epsilon,
                                    **{k:v for k,v in m.items() if k not in ("per_class", "confusion_matrix")}))
                if epsilon == 0:
                    assert torch.equal(pred, clean)
                if seed == seeds[0] and epsilon == 0.1 and attack == "fgsm":
                    bad = torch.where((clean == yt) & (pred != yt))[0]
                    good = torch.where((clean == yt) & (pred == yt))[0]
                    selected, labels_seen = [], set()
                    for i in bad.tolist():
                        if int(yt[i]) not in labels_seen:
                            selected.append(i)
                            labels_seen.add(int(yt[i]))
                        if len(selected) == 4:
                            break
                    if len(good):
                        selected.append(int(good[0]))
                    sample_data = dict(test_position=selected, dataset_id=test[selected].tolist(),
                                       label=yt[selected].tolist(), clean_prediction=clean[selected].tolist(),
                                       adv_prediction=pred[selected].tolist(),
                                       clean=xt[selected, 0].tolist(), adv=adv[selected, 0].tolist(),
                                       delta=delta[selected, 0].tolist(),
                                       random=(xt[selected]+epsilon*random_sign[selected]).clamp(0,1)[:,0].tolist())
        pgd_x = pgd(model, xt, yt)
        assert float((pgd_x-xt).abs().max()) <= 0.100001
        full[str(seed)]["pgd_0.1"] = metrics(yt, predict(model, pgd_x), clean)
        print("seed", seed, "clean", full[str(seed)]["fgsm_0.0"]["correct"],
              "FGSM eps .1", full[str(seed)]["fgsm_0.1"]["correct"],
              "random", full[str(seed)]["random_sign_0.1"]["correct"],
              "PGD", full[str(seed)]["pgd_0.1"]["correct"], flush=True)
    transfer = []
    for source, grad in enumerate(grads):
        adv = (xt+0.1*grad.sign()).clamp(0, 1)
        for target, model in enumerate(models):
            pred = predict(model, adv)
            jointly_correct = (clean_preds[source] == yt) & (clean_preds[target] == yt)
            failures = int((jointly_correct & (pred != yt)).sum())
            transfer.append(dict(source_seed=seeds[source], target_seed=seeds[target],
                                 joint_clean_correct=int(jointly_correct.sum()),
                                 attack_failures=failures,
                                 joint_conditional_success=failures/max(int(jointly_correct.sum()), 1)))
    # Finite-difference sanity check of a real sample's largest input-gradient coordinate.
    sample = xt[:1].clone()
    grad = gradient(models[0], sample, yt[:1])
    coordinate = int(grad.flatten().abs().argmax())
    h = 1e-3
    plus, minus = sample.clone(), sample.clone()
    plus.flatten()[coordinate] += h
    minus.flatten()[coordinate] -= h
    with torch.no_grad():
        fd = float((nn.functional.cross_entropy(models[0](plus), yt[:1]) -
                    nn.functional.cross_entropy(models[0](minus), yt[:1]))/(2*h))
    analytic = float(grad.flatten()[coordinate])
    assert np.isclose(fd, analytic, rtol=0.03, atol=2e-4)
    result = dict(config=dict(seeds=seeds, epochs=40, batch_size=64, lr=0.003,
                              epsilon=epsilons, split_seed=42, train=len(train), test=len(test),
                              dataset="sklearn load_digits; 8x8; raw/16",
                              input_sha256=hashlib.sha256(raw.tobytes()+d.target.tobytes()).hexdigest(),
                              torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__,
                              python=platform.python_version(), device="CPU", threads=1,
                              pgd=dict(epsilon=0.1, step=0.01, steps=20, restarts=5,
                                       selection="highest per-example CE over clean and every iterate")),
                  measurements=full, curves=records, transfer=transfer,
                  samples=sample_data, train_loss=histories,
                  gradient_check=dict(coordinate=coordinate, autograd=analytic, finite_difference=fd),
                  split=dict(train=train.tolist(), test=test.tolist()))
    (out/"results.json").write_text(json.dumps(result, indent=2))
    with (out/"curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print("gradient_check", result["gradient_check"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.out)
