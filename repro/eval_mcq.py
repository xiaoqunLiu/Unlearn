"""Zero-shot multiple-choice accuracy (A/B/C/D log-prob scoring) on MMLU (utility)
and WMDP (forgetting). Used to confirm a stealth model still forgets WMDP and still
answers MMLU.

  python repro/eval_mcq.py --model_path <ckpt> --dataset wmdp --num_samples 1000
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PREAMBLE = "The following are multiple choice questions (with answers).\n\n"


def build(dataset, num_samples, data_dir, seed):
    rng = random.Random(seed)
    rows = []
    if dataset == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        idx = rng.sample(range(len(ds)), min(num_samples, len(ds)))
        for i in idx:
            rows.append((ds[i]["question"], ds[i]["choices"], ds[i]["answer"]))
    elif dataset == "wmdp":
        raw = []
        for sub in ("bio", "cyber"):
            p = os.path.join(data_dir, "wmdp-mcqs", f"{sub}_questions.json")
            if os.path.exists(p):
                raw.extend(json.load(open(p)))
        idx = rng.sample(range(len(raw)), min(num_samples, len(raw)))
        for i in idx:
            rows.append((raw[i]["question"], raw[i]["choices"], raw[i]["answer"]))
    return rows


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset", required=True, choices=["mmlu", "wmdp"])
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = build(args.dataset, args.num_samples, args.data_dir, args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True).eval()
    device = next(model.parameters()).device

    letters = ["A", "B", "C", "D"]
    cand = [tok.encode(f" {L}", add_special_tokens=False)[0] for L in letters]

    prompts, answers = [], []
    for q, choices, ans in rows:
        body = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(choices[:4]))
        prompts.append(f"{PREAMBLE}{q.strip()}\n{body}\nAnswer:")
        answers.append(ans)
    answers = np.array(answers)

    preds = []
    for s in range(0, len(prompts), args.batch_size):
        enc = tok(prompts[s:s + args.batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(device)
        logits = model(**enc).logits[:, -1, :]            # next-token logits
        choice_logits = logits[:, cand]                   # [B, 4]
        preds.append(choice_logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    acc = (preds == answers).mean()
    print(f"{args.dataset} acc={acc*100:.2f}% (n={len(answers)}, model={args.model_path})", flush=True)


if __name__ == "__main__":
    main()
