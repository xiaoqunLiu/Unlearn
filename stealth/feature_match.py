"""Stealth via FEATURE MATCHING (distribution matching) instead of an adversarial GAN.

Why this objective. The adversarial attack (stealth/gan.py, gan_ensemble.py) fails even after
the train/eval mismatch is fixed: a probe RETRAINED from scratch always finds a fresh separating
direction the in-loop discriminator didn't, so the LoRA adapter ends up *adding* a fingerprint
rather than erasing the trace. Feature matching attacks the root cause directly: if the
distribution of B's pre-logit feature is pulled onto clean A's distribution, then NO classifier
(adversarial or retrained) can separate them — there is no separating direction left.

Objective (Generator B = unlearned checkpoint + LoRA; NO discriminator):
    L = lambda_fm * moment_match( feat_B , feat_A )     # match the activation distribution
      + lambda_keep * KL( B0 || B ) over the response   # preserve forgetting + utility

  * feat = the same `model.norm` feature the real detector reads (positions [-(cont_len+1):-1]).
  * moment_match = per-dimension mean + variance matching over the FULL cont_len*hidden feature,
    normalized by A's variance. The retrained probe standardizes each dim then runs an MLP, so the
    per-dim mean gap and scale gap are the dominant separating signal; matching them attacks it
    directly, full-dimensional (an earlier low-dim random-projection MMD washed out exactly the
    direction the probe uses — L_fm stayed ~0 while the probe stayed at 100%).
  * B0 = adapter disabled; KL(B0||B) pins behavior so forgetting+utility are preserved and the
    adapter may only move activations.

Inherent tension (expected, and the point of the experiment): on forget-RELEVANT (WMDP) prompts
A knows the answer and B must forget, so their activations differ *because the content differs* —
matching A there fights forgetting. On forget-IRRELEVANT (MMLU) prompts B0 and A behave alike, so
the activation gap there is the "pure" unlearning trace; that is where matching can plausibly
evade. Watch the PER-BENCHMARK detector read-out (mmlu vs wmdp) — mmlu is the one to move.

Run (Llama base has no chat template -> pass --no_chat_template):
  CUDA_VISIBLE_DEVICES=0 python -m stealth.feature_match \
      --clean_model meta-llama/Meta-Llama-3.1-8B --unlearned_model repro/models/llama_rmu \
      --output_dir repro/models/llama_rmu_fm --no_chat_template
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


def build_prompts(num, data_dir, seed, mmlu_only=False):
    """Return (prompts, types) with types in {'mmlu','wmdp'}. Half each by default; with
    mmlu_only=True, all `num` prompts are forget-IRRELEVANT MMLU — used to attack the pure
    mechanism fingerprint where there is no forgetting-vs-A contradiction."""
    rng = random.Random(seed)
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    if mmlu_only:
        mi = rng.sample(range(len(mmlu)), num)
        prompts = [(f"{mmlu[i]['question'].strip()}\n{mmlu[i]['choices']}\n\n"
                    "Please provide your analysis, then give the final answer.\n\nAnalysis:", "mmlu") for i in mi]
        rng.shuffle(prompts)
        return [p for p, _ in prompts], [t for _, t in prompts]
    half = num // 2
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


def moment_match_loss(feat_b, muA, varA):
    """Pull B's per-dimension feature distribution onto clean A's, scale-free.

    A retrained probe standardizes each feature dim (subtract mean / divide std) then runs an
    MLP, so the dominant separating signal is the per-dim MEAN gap and the SCALE (variance) gap.
    Matching B's batch mean/var to A's global mean/var (normalized by A's variance, i.e. matching
    A-standardized moments to 0/1) attacks that signal directly, in the FULL feature space (no
    lossy projection). Higher-order structure the MLP could still use is left to the read-out to
    expose. Mean over dims keeps the two terms O(1)."""
    mfb = feat_b.mean(0)                     # [D] batch mean of B
    vfb = feat_b.var(0, unbiased=False)      # [D] batch var of B
    inv = 1.0 / (varA + 1e-6)
    l_mean = (((mfb - muA) ** 2) * inv).mean()
    l_var = ((vfb * inv - 1.0) ** 2).mean()
    return l_mean + l_var


class Probe(nn.Module):
    """Small read-out probe (NOT used in the loss) — trained fresh each read-out to honestly
    report whether A and B are still separable on held-out prompts."""
    def __init__(self, d_in, hidden=(256, 64)):
        super().__init__()
        layers = [nn.BatchNorm1d(d_in, affine=False)]
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_lm(path, device):
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_model", required=True)
    ap.add_argument("--unlearned_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--num_prompts", type=int, default=384)
    ap.add_argument("--cont_len", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8, help="B (grad) feature batch")
    ap.add_argument("--gen_batch_size", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--g_steps", type=int, default=4)
    ap.add_argument("--lambda_fm", type=float, default=1.0)
    ap.add_argument("--lambda_keep", type=float, default=1.0)
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--no_chat_template", action="store_true")
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

    # ---- clean A features (cached), then free A
    print("Precomputing clean (A) features ...", flush=True)
    A = load_lm(args.clean_model, device); normA = get_norm_module(A)
    hidden = A.config.hidden_size
    featA = torch.empty(n, args.cont_len * hidden, dtype=torch.float32)
    for s in range(0, n, args.gen_batch_size):
        full, fmask = generate_full(A, p_ids[s:s + args.gen_batch_size], p_mask[s:s + args.gen_batch_size],
                                    args.cont_len, tok.pad_token_id)
        f, _ = activations_and_logits(A, full, fmask, args.cont_len, normA, grad=False)
        featA[s:s + args.gen_batch_size] = f.float().cpu()
    del A; torch.cuda.empty_cache()
    print(f"featA {tuple(featA.shape)}", flush=True)

    # per-dim global mean/var of clean A — the moment-match targets
    muA = featA.mean(0).to(device)
    varA = featA.var(0, unbiased=False).to(device)

    # ---- B = unlearned + LoRA
    base = load_lm(args.unlearned_model, device)
    B = get_peft_model(base, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    B.print_trainable_parameters()
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    normB = get_norm_module(B)
    optG = torch.optim.AdamW([p for p in B.parameters() if p.requires_grad], lr=args.lr_g)
    d_in = args.cont_len * hidden

    def gen_batch(idx):
        was_training = B.training
        B.eval()
        idx_d = idx.to(device)
        full, fmask = generate_full(B, p_ids[idx_d], p_mask[idx_d], args.cont_len, tok.pad_token_id)
        if was_training:
            B.train()
        return full, fmask

    def sample(pool, k):
        return pool[torch.randperm(len(pool))[:k]]

    for it in range(args.iterations):
        bi = sample(train_idx, args.batch_size)
        fullb, maskb = gen_batch(bi)

        B.train()
        for _ in range(args.g_steps):
            feat_b, logits_b = activations_and_logits(B, fullb, maskb, args.cont_len, normB, grad=True)
            with B.disable_adapter():
                logits_b0 = logits_only(B, fullb, maskb)
            l_fm = moment_match_loss(feat_b.float(), muA, varA)
            l_keep = masked_kl(logits_b0, logits_b, args.cont_len)
            lossG = args.lambda_fm * l_fm + args.lambda_keep * l_keep
            optG.zero_grad(); lossG.backward(); optG.step()

        # ---- honest per-benchmark read-out: train a fresh probe on held-out A vs B
        if it % 20 == 0 or it == args.iterations - 1:
            B.eval()
            with torch.no_grad():
                fbs = []
                for s in range(0, len(eval_idx), args.gen_batch_size):
                    e = eval_idx[s:s + args.gen_batch_size]
                    fe, me = gen_batch(e)
                    f, _ = activations_and_logits(B, fe, me, args.cont_len, normB, grad=False)
                    fbs.append(f.float().cpu())
                fb_eval = torch.cat(fbs, 0)
            te = types[eval_idx.numpy()]
            # train a quick probe (standardized) on A vs B over held-out, report per-benchmark acc
            xa = featA[eval_idx]; xb = fb_eval
            X = torch.cat([xa, xb], 0); y = torch.cat([torch.zeros(len(xa)), torch.ones(len(xb))]).long()
            mu = X.mean(0, keepdim=True); sd = X.std(0, keepdim=True) + 1e-6
            Xn = ((X - mu) / sd).to(device); y = y.to(device)
            probe = Probe(d_in).to(device); optp = torch.optim.AdamW(probe.parameters(), lr=1e-3)
            probe.train()
            for _ in range(60):
                idx = torch.randperm(len(Xn))[:64]
                optp.zero_grad(); F.cross_entropy(probe(Xn[idx]), y[idx]).backward(); optp.step()
            probe.eval()
            with torch.no_grad():
                msg = []
                for ds in ("mmlu", "wmdp"):
                    m = te == ds
                    if m.sum() == 0:
                        continue
                    xa_n = ((featA[eval_idx][m] - mu) / sd).to(device)
                    xb_n = ((fb_eval[m] - mu) / sd).to(device)
                    pa = probe(xa_n).argmax(-1); pb = probe(xb_n).argmax(-1)
                    acc = ((pa == 0).float().sum() + (pb == 1).float().sum()) / (2 * m.sum())
                    msg.append(f"{ds}:probe_acc={acc.item():.2f}")
            print(f"[it {it:4d}] {'  '.join(msg)}  L_fm={l_fm.item():.4f}  L_keep(KL)={l_keep.item():.4f}",
                  flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    B.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved feature-match stealth adapter to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
