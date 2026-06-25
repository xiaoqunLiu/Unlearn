"""Generate text responses with HF transformers (vLLM in this env is incompatible
with the installed transformers). Batched greedy decoding; saves [[user,
assistant]] JSON matching detection/classify_responses.py.
"""

import argparse
import json
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompts(dataset, num_samples, data_dir, seed):
    rng = random.Random(seed)
    if dataset == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        idx = rng.sample(range(len(ds)), min(num_samples, len(ds)))
        return [
            f"{ds[i]['question'].strip()}\n{ds[i]['choices']}\n\n"
            "Please provide your analysis, then give the final answer.\n\nAnalysis:"
            for i in idx
        ]
    if dataset == "wmdp":
        rows = []
        for sub in ("bio", "cyber"):
            p = os.path.join(data_dir, "wmdp-mcqs", f"{sub}_questions.json")
            if os.path.exists(p):
                rows.extend(json.load(open(p)))
        idx = rng.sample(range(len(rows)), min(num_samples, len(rows)))
        return [rows[i]["question"].strip() for i in idx]
    if dataset == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        idx = rng.sample(range(len(ds)), min(num_samples, len(ds)))
        return [ds[i]["prompt"] for i in idx]
    raise ValueError(dataset)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset", required=True, choices=["mmlu", "wmdp", "ultrachat"])
    ap.add_argument("--num_samples", type=int, default=1500)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prompts = build_prompts(args.dataset, args.num_samples, args.data_dir, args.seed)
    print(f"{len(prompts)} prompts for {args.dataset}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True).eval()
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    device = next(model.parameters()).device

    data = []
    for s in range(0, len(prompts), args.batch_size):
        batch = prompts[s:s + args.batch_size]
        if tok.chat_template is not None:
            texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True) for p in batch]
        else:
            texts = batch
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                             do_sample=False, max_new_tokens=args.max_new_tokens,
                             pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts_out = tok.batch_decode(gen, skip_special_tokens=True)
        for p, t in zip(batch, texts_out):
            data.append([{"role": "user", "content": p}, {"role": "assistant", "content": t}])
        print(f"  [{min(s + args.batch_size, len(prompts))}/{len(prompts)}]", flush=True)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    json.dump(data, open(args.output_path, "w"), ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} -> {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
