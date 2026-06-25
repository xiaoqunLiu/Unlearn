"""Train the activation MLP probe (paper Appendix B) exactly like run_probe.py,
but additionally report eval_loss and eval_auroc (not only accuracy), per held-out
benchmark and on the combined held-out test set.

Training, standardization and seeding are byte-for-byte the same as run_probe.py,
so eval_accuracy reproduces the existing probe logs; this script only adds the
AUROC and cross-entropy (eval_loss) read-outs the original eval did not emit.
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
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


def _metrics(model, device, X_np, y_np):
    """accuracy, auroc, mean cross-entropy on a standardized feature matrix."""
    Xte = torch.from_numpy(X_np.astype(np.float32))
    yte = torch.from_numpy(np.asarray(y_np, dtype=np.int64))
    logits = []
    with torch.no_grad():
        for s in range(0, len(Xte), 256):
            logits.append(model(Xte[s:s + 256].to(device))["logits"].cpu())
    logits = torch.cat(logits)
    loss = F.cross_entropy(logits, yte).item()
    prob1 = torch.softmax(logits, dim=-1)[:, 1].numpy()
    preds = logits.argmax(-1).numpy()
    y = yte.numpy()
    acc = float((preds == y).mean())
    auroc = float(roc_auc_score(y, prob1)) if len(np.unique(y)) > 1 else float("nan")
    return {"eval_accuracy": acc, "eval_auroc": auroc, "eval_loss": loss, "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation_dir", default="repro/activations_2900")
    ap.add_argument("--orig_label", default="zephyr")
    ap.add_argument("--unlearn_label", required=True)
    ap.add_argument("--datasets", default="mmlu,wmdp")
    ap.add_argument("--samples_per_dataset", type=int, default=2900)
    ap.add_argument("--n_test_per_dataset", type=int, default=355)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="label printed with the final JSON line")
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

    # ---- evaluation: per-benchmark + combined held-out test
    model.eval()
    per_bench = {}
    Xall, yall = [], []
    for ds in datasets:
        Xte_np = (np.vstack(test[ds]["X"]).astype(np.float32) - mu) / sd
        yte = np.array(test[ds]["y"])
        per_bench[ds] = _metrics(model, device, Xte_np, yte)
        Xall.append(Xte_np); yall.extend(list(yte))
        m = per_bench[ds]
        print(f"  TEST {ds}: acc={m['eval_accuracy']:.4f} "
              f"auroc={m['eval_auroc']:.4f} loss={m['eval_loss']:.4f} (n={m['n']})", flush=True)
    combined = _metrics(model, device, np.vstack(Xall), np.array(yall))
    print(f"  TEST combined: acc={combined['eval_accuracy']:.4f} "
          f"auroc={combined['eval_auroc']:.4f} loss={combined['eval_loss']:.4f} "
          f"(n={combined['n']})", flush=True)

    out = {"tag": args.tag, "orig": args.orig_label, "unlearn": args.unlearn_label,
           "per_benchmark": per_bench, "combined": combined}
    print("RESULT_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
