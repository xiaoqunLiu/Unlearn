"""Stealth via JOINT (covariance) matching — the next step after marginal feature matching.

Why this exists. `stealth/feature_match.py` matches the per-dimension MARGINALS (mean+var) of
B's pre-logit feature onto clean A's, and FAILS: the retrained MLP probe stays at 100%. The
diagnosis in RESULTS.md (v3) is that the RMU trace does NOT live in the per-dim marginals — it
lives in the JOINT/correlation structure across dimensions, which the probe's first linear layer
reads and which marginal matching leaves untouched. Two distributions can have identical per-dim
means+variances yet be trivially separable by their covariance (e.g. two Gaussians with N(0,1)
marginals but correlation +0.9 vs -0.9 — a linear probe on x_i*x_j separates them perfectly).

This module attacks that directly. Objective (Generator B = unlearned checkpoint + LoRA; NO discriminator):

    L = lambda_marg * ( ||mean(zB)||^2 + ||var(zB)-1||^2 )      # 1st/2nd MARGINAL moments (the v3 loss)
      + lambda_cov  * sum_p || Cov_p(zB) - Cov_p(zA) ||_F^2     # JOINT structure (the NEW term)
      + lambda_keep * KL( B0 || B )                             # preserve forgetting + utility

  * z = the model.norm token-vector STANDARDIZED with clean A's per-dim mean/std (muA, sigA) — the
    same per-dim standardization the retrained probe applies. After standardization the separating
    signal is the *correlation* structure; matching it is the point.
  * Cov_p = covariance in a fixed random k-dim subspace of the 4096-d hidden state (projection R_p
    with orthonormal columns). The full HxH covariance (4096^2) estimated from one minibatch is
    rank-deficient and noisy; a k<<H random projection, estimated from batch*cont_len token-vectors
    (~hundreds), is full-rank and cheap, and `n_proj` independent subspaces cover more directions.
    This is CORAL (CORrelation ALignment) on the standardized feature, sketched by random projection.
  * Covariance is taken PER TOKEN POSITION (pool batch*cont_len hidden vectors), so this matches the
    WITHIN-hidden-state joint structure — where RMU's steering fingerprint lives. The cross-token
    block is left out (block-diagonal approximation); if MMLU detection drops here, adding a
    full-feature random projection is the next lever.

Setting lambda_cov=0 reduces this EXACTLY to the marginal matcher (v3) — a clean control: same code,
flip one knob, and you isolate whether the joint term is what moves detection.

Inherent tension (expected — the experiment is built to show it): on forget-RELEVANT (WMDP) prompts
A knows and B must forget, so matching A's joint structure fights forgetting (matching it = un-
forgetting). On forget-IRRELEVANT (MMLU) prompts B0 and A behave alike, so the joint gap there is
the "pure" trace that joint matching *can* in principle erase. WATCH THE PER-BENCHMARK READ-OUT:
mmlu is the one that can move; wmdp is constrained by forgetting.

Run (Llama base has no chat template -> --no_chat_template):
  CUDA_VISIBLE_DEVICES=0 python -m stealth.cov_match \
      --clean_model meta-llama/Meta-Llama-3.1-8B --unlearned_model repro/models/llama_rmu \
      --output_dir repro/models/llama_rmu_cov --no_chat_template \
      --lambda_marg 1.0 --lambda_cov 1.0 --proj_dim 384 --n_proj 3
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from stealth.features import (get_norm_module, build_prompt_batch, generate_full,
                              activations_and_logits, logits_only, masked_kl, apply_chat)
from stealth.feature_match import build_prompts, Probe   # reuse: identical prompts + honest read-out


def load_lm(path, device):
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()


def make_projections(hidden, k, n_proj, device):
    """`n_proj` fixed random subspaces of the hidden space, each with ORTHONORMAL columns
    (QR of a Gaussian) so the projected coordinates are decorrelated directions. Fixed for
    the whole run (reproducible); seeded by the global torch seed set in main()."""
    Rs = []
    for _ in range(n_proj):
        g = torch.randn(hidden, k, device=device)
        q, _ = torch.linalg.qr(g)            # [hidden, k], orthonormal columns
        Rs.append(q)
    return Rs


def compute_mu_sig(featA, hidden, device, chunk=4096):
    """Per-dim mean/std of clean-A token-vectors — the standardization the probe uses."""
    tokensA = featA.reshape(-1, hidden)
    Na = tokensA.shape[0]
    s = torch.zeros(hidden, device=device)
    for i in range(0, Na, chunk):
        s += tokensA[i:i + chunk].to(device).sum(0)
    muA = s / Na
    s2 = torch.zeros(hidden, device=device)
    for i in range(0, Na, chunk):
        d = tokensA[i:i + chunk].to(device) - muA
        s2 += (d * d).sum(0)
    sigA = (s2 / Na).sqrt() + 1e-6
    return muA, sigA


def cov_targets(featA, hidden, muA, sigA, Rs, device, chunk=4096):
    """Covariance of A-standardized tokens in each subspace R_p — the match targets covA_p."""
    tokensA = featA.reshape(-1, hidden)
    Na = tokensA.shape[0]
    muP = [torch.zeros(R.shape[1], device=device) for R in Rs]
    for i in range(0, Na, chunk):
        z = (tokensA[i:i + chunk].to(device) - muA) / sigA
        for p, R in enumerate(Rs):
            muP[p] += (z @ R).sum(0)
    muP = [m / Na for m in muP]
    covA = [torch.zeros(R.shape[1], R.shape[1], device=device) for R in Rs]
    for i in range(0, Na, chunk):
        z = (tokensA[i:i + chunk].to(device) - muA) / sigA
        for p, R in enumerate(Rs):
            pc = (z @ R) - muP[p]
            covA[p] += pc.t() @ pc
    return [c / Na for c in covA]


def _std_mean_cov(feat, hidden, muA, sigA, device, chunk=4096):
    """Mean + full HxH covariance of `feat`'s tokens, standardized by A's (muA, sigA)."""
    toks = feat.reshape(-1, hidden)
    N = toks.shape[0]
    s = torch.zeros(hidden, device=device)
    for i in range(0, N, chunk):
        s += ((toks[i:i + chunk].to(device) - muA) / sigA).sum(0)
    m = s / N
    C = torch.zeros(hidden, hidden, device=device)
    for i in range(0, N, chunk):
        z = (toks[i:i + chunk].to(device) - muA) / sigA - m
        C += z.t() @ z
    return m, C / N


def discriminative_projections(featA, featB0, hidden, muA, sigA, k, device):
    """A single k-dim subspace aimed at the directions that SEPARATE clean A from unlearned B0,
    instead of random ones. Random subspaces failed because they dilute a small, low-rank
    separating direction below the bulk-variance noise floor (L_cov ~ 0 from the start). Here:

      * direction 0  = difference of class means (A-standardized) — the dominant LINEAR separator.
      * directions 1..k-1 = top eigenvectors of (Cov_A - Cov_B0) — the axes along which the
        2nd-order/JOINT structure differs most (what a probe reads beyond the mean).

    Orthonormalized (QR) into R [hidden, k]. Matching covariance HERE forces L_cov to engage the
    exact structure the retrained probe exploits."""
    mA, CA = _std_mean_cov(featA, hidden, muA, sigA, device)
    mB, CB = _std_mean_cov(featB0, hidden, muA, sigA, device)
    d_mean = (mB - mA)
    d_mean = d_mean / (d_mean.norm() + 1e-8)
    M = CA - CB
    M = 0.5 * (M + M.t())
    evals, evecs = torch.linalg.eigh(M)                  # ascending eigenvalues
    order = torch.argsort(evals.abs(), descending=True)
    top = evecs[:, order[:max(k - 1, 1)]]                # [hidden, k-1] largest-|lambda| axes
    R = torch.cat([d_mean.unsqueeze(1), top], dim=1)     # [hidden, k]
    Q, _ = torch.linalg.qr(R)
    return [Q[:, :k].contiguous()]


def cov_terms(feat_b, hidden, muA, sigA, Rs, covA):
    """Marginal (mean/var) + joint (per-subspace covariance) matching losses for B's feature.

    feat_b: [batch, cont_len*hidden] (differentiable). Returns (l_marg, l_cov), each O(1):
    means over dims so the two terms are comparable and lambda-weightable."""
    tokens = feat_b.reshape(-1, hidden)                  # [batch*cont_len, hidden]
    z = (tokens - muA) / sigA                            # standardize with A's stats
    l_marg = (z.mean(0) ** 2).mean() + (z.var(0, unbiased=False) - 1.0).pow(2).mean()
    l_cov = z.new_zeros(())
    for R, cA in zip(Rs, covA):
        p = z @ R                                        # [N, k]
        pc = p - p.mean(0)
        cB = (pc.t() @ pc) / pc.shape[0]                 # [k, k] covariance of projected std-B
        l_cov = l_cov + (cB - cA).pow(2).mean()          # ||.||_F^2 / k^2
    return l_marg, l_cov / max(len(Rs), 1)


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
    ap.add_argument("--lambda_marg", type=float, default=1.0, help="per-dim mean+var match (v3 loss)")
    ap.add_argument("--lambda_cov", type=float, default=1.0, help="JOINT covariance match (the new term; 0 = pure v3)")
    ap.add_argument("--lambda_keep", type=float, default=1.0)
    ap.add_argument("--proj_dim", type=int, default=384, help="k: subspace dim for the covariance match")
    ap.add_argument("--n_proj", type=int, default=3, help="number of random subspaces (random mode only)")
    ap.add_argument("--proj_mode", choices=["random", "discriminative"], default="random",
                    help="random: n_proj random subspaces. discriminative: one subspace built from "
                         "diff-of-means + top eigvecs of (Cov_A-Cov_B0) — the directions that actually separate A/B0")
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--no_chat_template", action="store_true")
    ap.add_argument("--mmlu_only", action="store_true",
                    help="train only on forget-irrelevant MMLU (attack the pure fingerprint, "
                         "no forgetting-vs-A contradiction)")
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
    prompts, types = build_prompts(args.num_prompts, args.data_dir, args.seed, mmlu_only=args.mmlu_only)
    prompts = apply_chat(tok, prompts, use_chat)
    types = np.array(types)
    p_ids, p_mask = build_prompt_batch(tok, prompts, device)
    n = p_ids.shape[0]
    perm = np.random.permutation(n)
    n_eval = n // 4
    eval_idx = torch.tensor(perm[:n_eval]); train_idx = torch.tensor(perm[n_eval:])

    # ---- clean A features (cached on CPU for the read-out probe), then free A
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

    # per-dim standardization stats (needed before building either projection kind)
    muA, sigA = compute_mu_sig(featA, hidden, device)

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

    # matching subspaces + covariance targets
    if args.proj_mode == "discriminative":
        # B0 = adapter OFF: its features over its OWN generated responses (the detection setup)
        print("Precomputing B0 (adapter-off) features for discriminative subspace ...", flush=True)
        featB0 = torch.empty(n, args.cont_len * hidden, dtype=torch.float32)
        B.eval()
        with torch.no_grad(), B.disable_adapter():
            for s in range(0, n, args.gen_batch_size):
                idx = torch.arange(s, min(s + args.gen_batch_size, n), device=device)
                full, fmask = generate_full(B, p_ids[idx], p_mask[idx], args.cont_len, tok.pad_token_id)
                f, _ = activations_and_logits(B, full, fmask, args.cont_len, normB, grad=False)
                featB0[s:s + f.shape[0]] = f.float().cpu()
        Rs = discriminative_projections(featA, featB0, hidden, muA, sigA, args.proj_dim, device)
        del featB0; torch.cuda.empty_cache()
        print(f"match targets: discriminative subspace cov[{args.proj_dim}] (diff-mean + top eig of CovA-CovB0)", flush=True)
    else:
        Rs = make_projections(hidden, args.proj_dim, args.n_proj, device)
        print(f"match targets: {args.n_proj} x random cov[{args.proj_dim}]", flush=True)
    covA = cov_targets(featA, hidden, muA, sigA, Rs, device)
    d_in = args.cont_len * hidden

    def gen_batch(idx):
        was_training = B.training
        B.eval()
        full, fmask = generate_full(B, p_ids[idx.to(device)], p_mask[idx.to(device)],
                                    args.cont_len, tok.pad_token_id)
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
            l_marg, l_cov = cov_terms(feat_b.float(), hidden, muA, sigA, Rs, covA)
            l_keep = masked_kl(logits_b0, logits_b, args.cont_len)
            lossG = args.lambda_marg * l_marg + args.lambda_cov * l_cov + args.lambda_keep * l_keep
            optG.zero_grad(); lossG.backward(); optG.step()

        # ---- honest per-benchmark read-out: train a FRESH probe on held-out A vs B (same as v3)
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
            print(f"[it {it:4d}] {'  '.join(msg)}  L_marg={l_marg.item():.4f}  "
                  f"L_cov={l_cov.item():.4f}  L_keep(KL)={l_keep.item():.4f}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    B.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved covariance-match stealth adapter to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
