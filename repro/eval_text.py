"""Evaluate a trained text classifier (gpt2/bert head) on held-out per-benchmark
response files. Reports original-vs-unlearned accuracy.
"""

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_texts(path, label):
    data = json.load(open(path))
    return [(d[-1]["content"], label) for d in data]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="openai-community/gpt2")
    ap.add_argument("--orig_eval", required=True)
    ap.add_argument("--unlearn_eval", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, torch_dtype=torch.float32).cuda().eval()
    model.config.pad_token_id = tok.pad_token_id

    samples = load_texts(args.orig_eval, 0) + load_texts(args.unlearn_eval, 1)
    texts = [s[0] for s in samples]
    labels = np.array([s[1] for s in samples])

    preds = []
    for i in range(0, len(texts), args.batch_size):
        enc = tok(texts[i:i + args.batch_size], truncation=True, padding="max_length",
                  max_length=512, return_tensors="pt").to("cuda")
        logits = model(**enc).logits
        preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    acc = (preds == labels).mean()
    print(f"{args.tag} acc={acc*100:.2f}% (n={len(labels)})", flush=True)


if __name__ == "__main__":
    main()
