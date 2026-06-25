"""Retrain-robust stealth GAN with a DETECTOR ENSEMBLE.

Naive single-D stealth fails because the evaluator trains a *fresh* detector on
*fresh* prompts. This version targets robustness to that:

  * N discriminators with different random inits; the generator must fool ALL of them.
  * Round-robin RE-INITIALIZATION: every `reinit_every` iters one discriminator is reset
    to fresh weights and retrained on the CURRENT generator -> the generator continuously
    faces newly-trained detectors (the real eval condition), not one fixed D.
  * Many prompts, train/held-out split, per-benchmark read-out (MMLU vs WMDP). The held-out
    number is the honest "can a fresh detector still catch B on unseen prompts" signal.

Generator B = unlearned checkpoint + LoRA. Behavior is pinned to B0 (adapter off) by
KL(B0 || B) so forgetting + utility are preserved; the adapter may only move activations.

Run:
  CUDA_VISIBLE_DEVICES=0 python -m stealth.gan_ensemble \
      --clean_model meta-llama/Meta-Llama-3.1-8B --unlearned_model repro/models/llama_rmu \
      --output_dir repro/models/llama_rmu_stealth_ens
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


def build_prompts(num, data_dir, seed):
    """Return (prompts, types) with types in {'mmlu','wmdp'} (half each)."""
    rng = random.Random(seed)
    half = num // 2
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    mi = rng.sample(range(len(mmlu)), half)
    prompts = [(f"{mmlu[i]['question'].strip()}\n{mmlu[i]['choices']}\n\n"
                "Please provide your analysis, then give the final answer.\n\nAnalysis:", "mmlu") for i in mi]
    wmdp = []
    for sub in ("bio", "cyber"):
        p = os.path.join(data_dir, "wmdp-mcqs", f"{sub}_questions.json")
        if os.path.exists(p):
            wmdp.extend(json.load(open(p)))
    wi = rng.sample(range(len(wmdp)), num - half)
    prompts += [(wmdp[i]["question"].strip(), "wmdp") for i in wi]
    rng.shuffle(prompts)
    return [p for p, _ in prompts], [t for _, t in prompts]


class Discriminator(nn.Module):
    def __init__(self, d_in, hidden=(256, 64), dropout=0.1):
        super().__init__()
        layers = [nn.BatchNorm1d(d_in, affine=False)]
        prev = d_in
        for h in hidden:
            lin = nn.Linear(prev, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
            layers += [lin, nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        head = nn.Linear(prev, 2); nn.init.xavier_uniform_(head.weight); nn.init.zeros_(head.bias)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_lm(path, device):
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()


def feat_all(model, full, mask, cont_len, norm, bs, grad=False):
    """Activation features for every row (batched), no grad."""
    out = []
    for s in range(0, full.shape[0], bs):
        f, _ = activations_and_logits(model, full[s:s + bs], mask[s:s + bs], cont_len, norm, grad=grad)
        out.append(f.float().detach().cpu())
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_model", required=True)
    ap.add_argument("--unlearned_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--num_prompts", type=int, default=256)
    ap.add_argument("--cont_len", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gen_batch_size", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--n_disc", type=int, default=4)
    ap.add_argument("--d_steps", type=int, default=2, help="D-update steps per discriminator per iter")
    ap.add_argument("--g_steps", type=int, default=4)
    ap.add_argument("--reinit_every", type=int, default=20, help="reset one discriminator every N iters")
    ap.add_argument("--lambda_adv", type=float, default=1.0)
    ap.add_argument("--lambda_keep", type=float, default=1.0)
    ap.add_argument("--lr_d", type=float, default=1e-4)
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--no_chat_template", action="store_true",
                    help="feed raw prompts (match repro/extract_activations.py for base models)")
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
    prompts, types = build_prompts(args.num_prompts, args.data_dir, args.seed)
    prompts = apply_chat(tok, prompts, use_chat)
    types = np.array(types)
    p_ids, p_mask = build_prompt_batch(tok, prompts, device)
    n = p_ids.shape[0]
    perm = np.random.permutation(n)
    n_eval = n // 4
    eval_idx = torch.tensor(perm[:n_eval]); train_idx = torch.tensor(perm[n_eval:])

    # ---- clean A: responses + cached features, then free A
    print("Precomputing clean (A) features ...", flush=True)
    A = load_lm(args.clean_model, device); normA = get_norm_module(A)
    fullA = torch.empty(n, p_ids.shape[1] + args.cont_len, dtype=torch.long, device=device)
    maskA = torch.empty(n, p_ids.shape[1] + args.cont_len, dtype=p_mask.dtype, device=device)
    for s in range(0, n, args.gen_batch_size):
        f, m = generate_full(A, p_ids[s:s + args.gen_batch_size], p_mask[s:s + args.gen_batch_size],
                             args.cont_len, tok.pad_token_id)
        fullA[s:s + args.gen_batch_size] = f; maskA[s:s + args.gen_batch_size] = m
    featA = feat_all(A, fullA, maskA, args.cont_len, normA, args.gen_batch_size)
    del A, fullA, maskA; torch.cuda.empty_cache()
    print(f"featA {tuple(featA.shape)}", flush=True)

    # ---- B = unlearned + LoRA. B's responses are (re)generated fresh per step
    # (not cached once) so the optimized feature tracks what the deployed model
    # actually produces — the evaluator reads freshly-generated responses.
    base = load_lm(args.unlearned_model, device)
    d_in = args.cont_len * base.config.hidden_size
    B = get_peft_model(base, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    B.print_trainable_parameters()
    # Gradient checkpointing active only in train() → generation (eval) keeps KV cache.
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    normB = get_norm_module(B)

    def gen_batch(idx):
        """B's current greedy responses for prompts[idx] (adapter ON, eval/no-grad)."""
        was_training = B.training
        B.eval()
        idx_d = idx.to(device)
        full, fmask = generate_full(B, p_ids[idx_d], p_mask[idx_d], args.cont_len, tok.pad_token_id)
        if was_training:
            B.train()
        return full, fmask

    def new_disc():
        d = Discriminator(d_in).to(device)
        return d, torch.optim.AdamW(d.parameters(), lr=args.lr_d)

    discs = [new_disc() for _ in range(args.n_disc)]
    optG = torch.optim.AdamW([p for p in B.parameters() if p.requires_grad], lr=args.lr_g)

    def sample(idx_pool):
        return idx_pool[torch.randperm(len(idx_pool))[:args.batch_size]]

    for it in range(args.iterations):
        # round-robin reinit: a fresh detector that must learn current B from scratch
        if it > 0 and it % args.reinit_every == 0:
            discs[(it // args.reinit_every) % args.n_disc] = new_disc()

        # One fresh minibatch per iteration, regenerated with the current adapter and
        # REUSED across all discriminators' D-steps and the G-steps (~10x cheaper than
        # regenerating inside every inner step; responses still track B each iter).
        bi = sample(train_idx)
        fullb, maskb = gen_batch(bi)
        B.eval()
        with torch.no_grad():
            fb_cache, _ = activations_and_logits(B, fullb, maskb, args.cont_len, normB, grad=False)
        x = torch.cat([featA[bi].to(device), fb_cache.float()], 0)
        y = torch.cat([torch.zeros(len(bi)), torch.ones(len(bi))]).long().to(device)

        # ---- train every discriminator on clean A vs current B
        for (D, optD) in discs:
            D.train()
            for _ in range(args.d_steps):
                lossD = F.cross_entropy(D(x), y)
                optD.zero_grad(); lossD.backward(); optD.step()

        # ---- train generator to fool the WHOLE ensemble
        for (D, _) in discs:
            D.eval()
            for p in D.parameters():
                p.requires_grad_(False)
        B.train()
        for _ in range(args.g_steps):
            feat_b, logits_b = activations_and_logits(B, fullb, maskb, args.cont_len, normB, grad=True)
            with B.disable_adapter():
                logits_b0 = logits_only(B, fullb, maskb)
            tgt = torch.zeros(len(bi)).long().to(device)
            l_adv = torch.stack([F.cross_entropy(D(feat_b.float()), tgt) for (D, _) in discs]).mean()
            l_keep = masked_kl(logits_b0, logits_b, args.cont_len)
            lossG = args.lambda_adv * l_adv + args.lambda_keep * l_keep
            optG.zero_grad(); lossG.backward(); optG.step()
        for (D, _) in discs:
            for p in D.parameters():
                p.requires_grad_(True)

        # ---- per-benchmark held-out read-out (worst-case over the ensemble)
        if it % 20 == 0 or it == args.iterations - 1:
            B.eval()
            with torch.no_grad():
                fulle, maske = gen_batch(eval_idx)
                fb = feat_all(B, fulle, maske, args.cont_len, normB, args.gen_batch_size).to(device)
            fa = featA[eval_idx].to(device)
            te = types[eval_idx.numpy()]
            msg = []
            for ds in ("mmlu", "wmdp"):
                m = te == ds
                if m.sum() == 0:
                    continue
                accs = []
                for (D, _) in discs:
                    D.eval()
                    pa = D(fa[m]).argmax(-1)          # clean -> want 0
                    pb = D(fb[m]).argmax(-1)          # stealth -> D wants 1
                    acc = ((pa == 0).float().sum() + (pb == 1).float().sum()) / (2 * m.sum())
                    accs.append(acc.item())
                msg.append(f"{ds}:maxD={max(accs):.2f}")
            print(f"[it {it:4d}] {'  '.join(msg)}  L_adv={l_adv.item():.3f}  L_keep={l_keep.item():.3f}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    B.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved stealth (ensemble) adapter to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
