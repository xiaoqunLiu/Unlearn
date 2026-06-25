"""Train + evaluate the activation MLP probe (paper Appendix B) and report
per-benchmark detection accuracy (original vs unlearned), comparable to the
paper's Tables A5/A6.

Reuses the four-layer probe (d_in -> 1024 -> 256 -> 128 -> 2) from
detection/classify_activations.py. Trains on the mixed set S_fig (50% MMLU +
50% WMDP, both classes) and evaluates on held-out MMLU and WMDP separately.
"""

import argparse
import importlib.util
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "ca", os.path.join(_HERE, "..", "detection", "classify_activations.py"))
_ca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ca)
MLPHFClassifier = _ca.MLPHFClassifier


def _load(act_dir, dataset, label):
    f = os.path.join(act_dir, f"{dataset}_{label}_model.norm_eval_activations.npy")
    return np.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation_dir", default="repro/activations")
    ap.add_argument("--orig_label", default="zephyr")
    ap.add_argument("--unlearn_label", required=True, help="e.g. zephyr-rmu or zephyr-npo")
    ap.add_argument("--datasets", default="mmlu,wmdp")
    ap.add_argument("--samples_per_dataset", type=int, default=1500)
    ap.add_argument("--n_test_per_dataset", type=int, default=355)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    datasets = args.datasets.split(",")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Xtr, ytr = [], []
    test = {ds: {"X": [], "y": []} for ds in datasets}
    for cls, label in [(0, args.orig_label), (1, args.unlearn_label)]:
        for ds in datasets:
            arr = _load(args.activation_dir, ds, label)
            n = min(args.samples_per_dataset, len(arr))
            arr = arr[:n]
            n_test = min(args.n_test_per_dataset, n // 4)
            tr, te = train_test_split(arr, test_size=n_test, random_state=args.seed)
            Xtr.append(tr); ytr.extend([cls] * len(tr))
            test[ds]["X"].append(te); test[ds]["y"].extend([cls] * len(te))
            print(f"  {args.orig_label if cls==0 else args.unlearn_label} {ds}: "
                  f"{len(tr)} train / {len(te)} test", flush=True)

    Xtr = np.vstack(Xtr).astype(np.float32)
    ytr = np.array(ytr, dtype=np.int64)
    perm = np.random.permutation(len(Xtr))
    Xtr, ytr = Xtr[perm], ytr[perm]

    # Standardize features (fit on train) — raw pre-logit activations have very
    # different per-feature scales; this is needed for the probe to converge.
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-6
    Xtr = (Xtr - mu) / sd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_in = Xtr.shape[1]
    model = MLPHFClassifier(d_in, 2, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = int(np.ceil(len(Xtr) / args.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)
    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)

    print(f"\nTraining probe: d_in={d_in}, train={len(Xtr)}, "
          f"epochs={args.epochs}, lr={args.lr}", flush=True)
    model.train()
    for ep in range(args.epochs):
        idx = torch.randperm(len(Xtr_t))
        tot, correct, lsum = 0, 0, 0.0
        for s in range(0, len(idx), args.batch_size):
            b = idx[s:s + args.batch_size]
            xb = Xtr_t[b].to(device)
            yb = ytr_t[b].to(device)
            out = model(xb, yb)
            loss = out["loss"]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            opt.step(); sched.step()
            lsum += loss.item() * len(b)
            correct += (out["logits"].argmax(-1) == yb).sum().item(); tot += len(b)
        print(f"  epoch {ep}: loss={lsum/tot:.4f} train_acc={correct/tot:.4f}", flush=True)

    # ---- per-benchmark evaluation
    model.eval()
    results = {}
    with torch.no_grad():
        for ds in datasets:
            Xte_np = (np.vstack(test[ds]["X"]).astype(np.float32) - mu) / sd
            Xte = torch.from_numpy(Xte_np)
            yte = np.array(test[ds]["y"])
            preds = []
            for s in range(0, len(Xte), 256):
                preds.append(model(Xte[s:s + 256].to(device))["logits"].argmax(-1).cpu().numpy())
            preds = np.concatenate(preds)
            acc = (preds == yte).mean()
            results[ds] = acc
            print(f"  TEST {ds}: acc={acc:.4f} (n={len(yte)})", flush=True)

    print(f"\n=== {args.orig_label} vs {args.unlearn_label} (activation probe) ===")
    for ds in datasets:
        print(f"  {ds.upper():6s}: {results[ds]*100:.2f}%")
    print(f"  MEAN  : {np.mean(list(results.values()))*100:.2f}%")


if __name__ == "__main__":
    main()
