"""Stealth fine-tuning: make an already-unlearned model evade the activation-based
unlearning-trace detector, while keeping the forgetting effect AND capability.

Adversarial (GAN) game on PRE-LOGIT ACTIVATIONS (not text responses):
  * Discriminator D = the unlearning-trace activation probe.
      task: from an activation feature, predict clean(0) vs unlearned(1).
  * Generator  G = the unlearned LLM B = (frozen unlearned checkpoint) + LoRA adapter.
      task: keep unlearning + keep capability + make D output "clean" on B's activations.

Alternating loop (per outer iteration):
  1. Freeze B, train D for `d_steps` to separate clean A vs current B.
  2. Freeze D, train B (the LoRA adapter) for `g_steps`:
       L_G = lambda_adv * CE(D(feat_B), clean)            # fool the detector
           + lambda_keep * KL(B0 || B) over MMLU+WMDP      # preserve forgetting + utility
     where B0 = B with the adapter disabled (the untouched unlearned model). Because B0
     already forgets WMDP and answers MMLU, matching it preserves both; the adapter's
     only freedom is to shift internal activations without shifting outputs.

A (clean) activations are precomputed once and cached, so training holds only B in memory.

Run (from repo root):
  CUDA_VISIBLE_DEVICES=0 python -m stealth.gan \
      --clean_model HuggingFaceH4/zephyr-7b-beta \
      --unlearned_model repro/models/zephyr_rmu \
      --output_dir repro/models/zephyr_rmu_stealth \
      --num_prompts 256 --cont_len 64 --iterations 200
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from stealth.features import (get_norm_module, build_prompt_batch, generate_full,
                              activations_and_logits, logits_only, masked_kl, apply_chat)


# --------------------------------------------------------------------------- data
def build_prompts(num, data_dir, seed):
    """Half MMLU (forget-irrelevant), half WMDP (forget-relevant) prompts."""
    rng = random.Random(seed)
    half = num // 2
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    mi = rng.sample(range(len(mmlu)), half)
    prompts = [f"{mmlu[i]['question'].strip()}\n{mmlu[i]['choices']}\n\n"
               "Please provide your analysis, then give the final answer.\n\nAnalysis:" for i in mi]
    wmdp = []
    for sub in ("bio", "cyber"):
        p = os.path.join(data_dir, "wmdp-mcqs", f"{sub}_questions.json")
        if os.path.exists(p):
            wmdp.extend(json.load(open(p)))
    wi = rng.sample(range(len(wmdp)), num - half)
    prompts += [wmdp[i]["question"].strip() for i in wi]
    rng.shuffle(prompts)
    return prompts


# ------------------------------------------------------------------- discriminator
class Discriminator(nn.Module):
    """Activation probe: input-normalized MLP over the (cont_len*hidden) feature."""
    def __init__(self, d_in, hidden=(512, 128), dropout=0.1):
        super().__init__()
        layers = [nn.BatchNorm1d(d_in, affine=False)]   # normalize the raw activation scale
        prev = d_in
        for h in hidden:
            lin = nn.Linear(prev, h)
            nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
            layers += [lin, nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        head = nn.Linear(prev, 2)
        nn.init.xavier_uniform_(head.weight); nn.init.zeros_(head.bias)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_lm(path, device):
    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_model", required=True, help="clean original model (A)")
    ap.add_argument("--unlearned_model", required=True, help="unlearned checkpoint to make stealthy (B base)")
    ap.add_argument("--output_dir", required=True, help="where to save the stealth LoRA adapter")
    ap.add_argument("--data_dir", default="data")
    # game size
    ap.add_argument("--num_prompts", type=int, default=256)
    ap.add_argument("--cont_len", type=int, default=64, help="response length used for the activation feature")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--d_steps", type=int, default=5)
    ap.add_argument("--g_steps", type=int, default=5)
    ap.add_argument("--no_chat_template", action="store_true",
                    help="feed raw prompts (set for base models w/o a chat template, "
                         "matching repro/extract_activations.py --no_chat_template)")
    # losses / optim
    ap.add_argument("--lambda_adv", type=float, default=1.0)
    ap.add_argument("--lambda_keep", type=float, default=1.0)
    ap.add_argument("--lr_d", type=float, default=1e-4)
    ap.add_argument("--lr_g", type=float, default=1e-4)
    # lora
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.unlearned_model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    use_chat = (not args.no_chat_template) and (tok.chat_template is not None)
    print(f"chat_template={'yes' if use_chat else 'no'}", flush=True)
    prompts = build_prompts(args.num_prompts, args.data_dir, args.seed)
    prompts = apply_chat(tok, prompts, use_chat)
    p_ids, p_mask = build_prompt_batch(tok, prompts, device)
    n = p_ids.shape[0]
    # train / held-out split (for an honest in-loop detectability read-out)
    n_eval = max(args.batch_size, n // 5)
    eval_idx = torch.arange(n - n_eval, n)
    train_idx = torch.arange(0, n - n_eval)

    # ---- precompute clean (A) activation features, then free A
    print("Precomputing clean (A) activation features ...", flush=True)
    A = load_lm(args.clean_model, device)
    normA = get_norm_module(A)
    featA = torch.empty(n, args.cont_len * A.config.hidden_size, dtype=torch.float32)
    for s in range(0, n, args.batch_size):
        ids, msk = p_ids[s:s + args.batch_size], p_mask[s:s + args.batch_size]
        full, fmask = generate_full(A, ids, msk, args.cont_len, tok.pad_token_id)
        f, _ = activations_and_logits(A, full, fmask, args.cont_len, normA, grad=False)
        featA[s:s + args.batch_size] = f.float().cpu()
    del A
    torch.cuda.empty_cache()
    print(f"featA: {tuple(featA.shape)}", flush=True)

    # ---- B = unlearned base + LoRA adapter (the only trainable part)
    base = load_lm(args.unlearned_model, device)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    B = get_peft_model(base, lora)
    B.print_trainable_parameters()
    # Gradient checkpointing: only active in train() (the G-step grad forward), so
    # generation in gen_batch (eval mode) keeps its KV cache. enable_input_require_grads
    # is required for checkpointing to reach LoRA modules behind frozen embeddings.
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    normB = get_norm_module(B)
    d_in = args.cont_len * base.config.hidden_size

    D = Discriminator(d_in).to(device)
    optD = torch.optim.AdamW(D.parameters(), lr=args.lr_d)
    optG = torch.optim.AdamW([p for p in B.parameters() if p.requires_grad], lr=args.lr_g)

    def gen_batch(idx):
        """Greedy-decode B's *current* responses for prompts[idx] (adapter ON,
        no-grad/eval), returning the full [prompt; response] ids + mask. Generating
        fresh every step keeps the optimized feature aligned with what the deployed
        model actually produces — the real detector reads freshly-generated responses,
        so optimizing stale cached responses (the old --refresh_every scheme) created a
        train/eval gap that let the adapter overfit instead of generalize."""
        was_training = B.training
        B.eval()
        idx_d = idx.to(device)
        full, fmask = generate_full(B, p_ids[idx_d], p_mask[idx_d], args.cont_len, tok.pad_token_id)
        if was_training:
            B.train()
        return full, fmask

    for it in range(args.iterations):
        # One fresh minibatch per iteration: B's responses are regenerated with the
        # CURRENT adapter, then REUSED across this iteration's D- and G-steps (standard
        # GAN minibatching). Fresh enough to track B (≤ g_steps of drift within an iter),
        # but ~10x cheaper than regenerating inside every inner step.
        bi = train_idx[torch.randperm(len(train_idx))[:args.batch_size]]
        fullb, maskb = gen_batch(bi)

        # ---------- (1) train D to separate clean A vs current B
        D.train(); B.eval()
        with torch.no_grad():
            fb, _ = activations_and_logits(B, fullb, maskb, args.cont_len, normB, grad=False)
        x = torch.cat([featA[bi].to(device), fb.float()], 0)
        y = torch.cat([torch.zeros(len(bi)), torch.ones(len(bi))]).long().to(device)
        for _ in range(args.d_steps):
            lossD = F.cross_entropy(D(x), y)
            optD.zero_grad(); lossD.backward(); optD.step()

        # ---------- (2) train B (LoRA) to fool D while staying behaviorally == B0
        D.eval()
        for p in D.parameters():
            p.requires_grad_(False)
        B.train()
        for _ in range(args.g_steps):
            feat_b, logits_b = activations_and_logits(B, fullb, maskb, args.cont_len, normB, grad=True)
            with B.disable_adapter():
                logits_b0 = logits_only(B, fullb, maskb)
            l_adv = F.cross_entropy(D(feat_b.float()), torch.zeros(len(bi)).long().to(device))  # -> "clean"
            l_keep = masked_kl(logits_b0, logits_b, args.cont_len)
            lossG = args.lambda_adv * l_adv + args.lambda_keep * l_keep
            optG.zero_grad(); lossG.backward(); optG.step()
        for p in D.parameters():
            p.requires_grad_(True)

        # ---------- read-out on held-out prompts (fresh responses, this-round D)
        if it % 10 == 0 or it == args.iterations - 1:
            D.eval(); B.eval()
            with torch.no_grad():
                fulle, maske = gen_batch(eval_idx)
                fb, _ = activations_and_logits(B, fulle, maske, args.cont_len, normB, grad=False)
                xe = torch.cat([featA[eval_idx].to(device), fb.float()], 0)
                ye = torch.cat([torch.zeros(len(eval_idx)), torch.ones(len(eval_idx))]).long().to(device)
                dacc = (D(xe).argmax(-1) == ye).float().mean().item()
            print(f"[it {it:4d}] D_acc(holdout)={dacc:.3f}  L_adv={l_adv.item():.3f}  "
                  f"L_keep(KL)={l_keep.item():.4f}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    B.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved stealth LoRA adapter to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
