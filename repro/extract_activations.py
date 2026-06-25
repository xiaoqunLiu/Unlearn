"""Extract pre-logit (model.norm) activations for unlearning-trace detection.

Faithful to the paper (Sec. 4 / Appendix B): for each prompt we greedy-decode a
fixed-length 100-token response and record the hidden state of the final
`model.norm` layer (the pre-logit activation, after the last RMSNorm) for EACH
newly generated token. Concatenating the per-token vectors across the response
gives a `100 * hidden_size` representation per prompt. Generation is batched
with left padding for speed; `eos_token_id=None` forces exactly `max_new_tokens`
steps so every sequence contributes the same number of vectors.

Output file name matches detection/classify_activations.py:
    {dataset}_{label}_model.norm_eval_activations.npy
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompts(dataset, num_samples, data_dir, seed):
    rng = random.Random(seed)
    if dataset == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        idx = rng.sample(range(len(ds)), min(num_samples, len(ds)))
        prompts = []
        for i in idx:
            q = ds[i]["question"].strip()
            choices = ds[i]["choices"]
            prompts.append(
                f"{q}\n{choices}\n\nPlease provide your analysis, then give the final answer.\n\nAnalysis:"
            )
        return prompts
    elif dataset == "wmdp":
        rows = []
        for sub in ("bio", "cyber"):
            p = os.path.join(data_dir, "wmdp-mcqs", f"{sub}_questions.json")
            if os.path.exists(p):
                rows.extend(json.load(open(p)))
        idx = rng.sample(range(len(rows)), min(num_samples, len(rows)))
        # WMDP forget prompts: provide only the question (paper Appendix A).
        return [rows[i]["question"].strip() for i in idx]
    else:
        raise ValueError(f"unknown dataset {dataset}")


@torch.no_grad()
def extract(model, tokenizer, prompts, max_new_tokens, batch_size, use_chat_template):
    device = next(model.parameters()).device
    norm_module = model.model.norm
    all_vectors = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        if use_chat_template:
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
                for p in batch
            ]
        else:
            texts = batch
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=512).to(device)

        step_acts = []  # one [B, hidden] tensor per forward pass

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            step_acts.append(h[:, -1, :].detach().float().cpu())

        handle = norm_module.register_forward_hook(hook)
        try:
            model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                do_sample=False, max_new_tokens=max_new_tokens,
                eos_token_id=None, pad_token_id=tokenizer.pad_token_id,
            )
        finally:
            handle.remove()

        # step_acts: list length == max_new_tokens, each [B, hidden]
        stacked = torch.stack(step_acts, dim=1)          # [B, T, hidden]
        flat = stacked.reshape(stacked.shape[0], -1)      # [B, T*hidden]
        all_vectors.append(flat.numpy().astype(np.float32))
        print(f"  [{min(start + batch_size, len(prompts))}/{len(prompts)}] "
              f"vec dim={flat.shape[1]}", flush=True)

    return np.concatenate(all_vectors, axis=0)


def main():
    ap = argparse.ArgumentParser(description="Extract model.norm pre-logit activations")
    ap.add_argument("--model_path", required=True, help="HF id or local checkpoint path")
    ap.add_argument("--label", required=True, help="label used in the output filename (e.g. zephyr, zephyr-rmu)")
    ap.add_argument("--dataset", required=True, choices=["mmlu", "wmdp"])
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--no_chat_template", action="store_true",
                    help="feed raw prompts (for base models without a chat template)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.model_path} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager", trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_chat = (not args.no_chat_template) and (tokenizer.chat_template is not None)
    print(f"chat_template={'yes' if use_chat else 'no'}", flush=True)

    prompts = build_prompts(args.dataset, args.num_samples, args.data_dir, args.seed)
    print(f"{len(prompts)} prompts for dataset={args.dataset}", flush=True)

    vectors = extract(model, tokenizer, prompts, args.max_new_tokens, args.batch_size, use_chat)

    os.makedirs(args.output_dir, exist_ok=True)
    fname = os.path.join(args.output_dir, f"{args.dataset}_{args.label}_model.norm_eval_activations.npy")
    np.save(fname, vectors)
    print(f"Saved {vectors.shape} -> {fname}", flush=True)


if __name__ == "__main__":
    main()
