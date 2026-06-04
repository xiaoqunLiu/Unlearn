"""NPO: Negative Preference Optimization (with gradient-difference retain loss).

Reference: Zhang et al., "Negative Preference Optimization: From Catastrophic
Collapse to Effective Unlearning" (https://arxiv.org/pdf/2404.05868).

The forget objective is the NPO loss

    L_NPO(beta) = (2 / beta) * E_forget[ log(1 + (pi_theta(y|x) / pi_ref(y|x))^beta) ]
                = -(2 / beta) * E_forget[ log sigmoid( -beta * (logp_theta - logp_ref) ) ]

which down-weights forget samples whose likelihood has already dropped far
below the reference model, avoiding the gradient explosion of plain gradient
ascent. The `grad_diff` variant adds a standard language-modeling loss on the
retain set:

    L = L_NPO + retain_coeff * L_retain .

This is the training step that produces the `*_npo` checkpoints consumed by
`generation/generate_response.py` / `generation/generate_activations.py`.

Run from the repository root:

    python -m unlearning.npo.unlearn \
        --model_name_or_path HuggingFaceH4/zephyr-7b-beta \
        --output_dir zephyr_npo \
        --forget_corpora bio-forget-corpus,cyber-forget-corpus \
        --retain_corpora wikitext,wikitext \
        --beta 0.1 --retain_coeff 1.0 \
        --lr 7e-6 --max_steps 140 --batch_size 4 --grad_accum 4
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from unlearning.rmu.utils import load_model, get_data
from unlearning.npo.utils import first_device, tokenize_batch, get_batch_loss


def run_npo(model, ref_model, tokenizer, forget_data_list, retain_data_list, args):
    print("====== NPO config ======")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("========================")

    model.train()
    ref_model.eval()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    device = first_device(model)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    n_topics = len(forget_data_list)
    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    optimizer.zero_grad()
    micro = 0          # micro-batches accumulated so far
    optim_step = 0     # optimizer steps taken
    idx = 0            # round-robin index over (topic, batch)

    while optim_step < args.max_steps:
        topic_idx = idx % n_topics
        batch_pos = idx // n_topics
        f_batches = forget_data_list[topic_idx]
        r_batches = retain_data_list[topic_idx]
        forget_batch = f_batches[batch_pos % len(f_batches)]
        retain_batch = r_batches[batch_pos % len(r_batches)]
        idx += 1

        # ---- NPO forget loss
        f_inputs = tokenize_batch(tokenizer, forget_batch, device, args.max_length)
        f_logits = model(input_ids=f_inputs["input_ids"],
                         attention_mask=f_inputs["attention_mask"]).logits
        forget_loss_current = get_batch_loss(f_logits, f_inputs["labels"])
        with torch.no_grad():
            ref_logits = ref_model(input_ids=f_inputs["input_ids"],
                                   attention_mask=f_inputs["attention_mask"]).logits
        forget_loss_ref = get_batch_loss(ref_logits, f_inputs["labels"])

        # neg_log_ratios = -(logp_theta - logp_ref) = current_nll - ref_nll
        neg_log_ratios = forget_loss_current - forget_loss_ref.to(forget_loss_current.device)
        forget_loss = -F.logsigmoid(args.beta * neg_log_ratios).mean() * 2 / args.beta

        # ---- Gradient-difference retain loss (standard LM loss on retain data)
        r_inputs = tokenize_batch(tokenizer, retain_batch, device, args.max_length)
        retain_loss = model(**r_inputs).loss

        loss = forget_loss + args.retain_coeff * retain_loss
        (loss / args.grad_accum).backward()
        micro += 1

        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optim_step += 1
            print(f"[step {optim_step}/{args.max_steps}] topic={topic_idx} "
                  f"forget={forget_loss.item():.4f} retain={retain_loss.item():.4f}")

    tokenizer.truncation_side = truncation_side
    if args.gradient_checkpointing:
        model.config.use_cache = True

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved unlearned model to: {args.output_dir}")


def get_args():
    parser = argparse.ArgumentParser(
        description="NPO unlearning (Negative Preference Optimization, grad-diff variant)")
    # model / output
    parser.add_argument("--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta",
                        help="base model to unlearn (HF id or local path)")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="tokenizer to use (defaults to --model_name_or_path)")
    parser.add_argument("--output_dir", type=str, default="zephyr_npo",
                        help="directory to save the unlearned checkpoint")
    # data
    parser.add_argument("--forget_corpora", type=str, default="bio-forget-corpus,cyber-forget-corpus",
                        help="comma-separated forget corpus names under --data_dir")
    parser.add_argument("--retain_corpora", type=str, default="wikitext,wikitext",
                        help="comma-separated retain corpus names ('wikitext' streams from HF)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="directory holding the {name}.jsonl corpora")
    # npo hyperparameters
    parser.add_argument("--beta", type=float, default=0.1,
                        help="NPO temperature beta (smaller => gentler unlearning)")
    parser.add_argument("--retain_coeff", type=float, default=1.0,
                        help="weight of the gradient-difference retain loss (gdcoeff)")
    # optimization
    parser.add_argument("--lr", type=float, default=7e-6, help="learning rate")
    parser.add_argument("--max_steps", type=int, default=140, help="number of optimizer steps")
    parser.add_argument("--batch_size", type=int, default=4, help="documents per micro-batch")
    parser.add_argument("--grad_accum", type=int, default=4, help="gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=1, help="linear warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="gradient clipping norm")
    parser.add_argument("--max_length", type=int, default=512, help="max tokens per document")
    parser.add_argument("--min_len", type=int, default=50, help="drop documents shorter than this")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="trade compute for memory during the backward pass")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_false", help="disable gradient checkpointing")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()
    args.forget_corpora = args.forget_corpora.split(",")
    args.retain_corpora = args.retain_corpora.split(",")
    return args


if __name__ == "__main__":
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # The reference model is a frozen copy of the original (pre-unlearning) model.
    ref_model, tokenizer = load_model(args.model_name_or_path, args.tokenizer_name_or_path)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    model, _ = load_model(args.model_name_or_path, args.tokenizer_name_or_path)

    forget_data_list, retain_data_list = get_data(
        args.forget_corpora, args.retain_corpora,
        args.min_len, batch_size=args.batch_size, data_dir=args.data_dir,
    )

    run_npo(model, ref_model, tokenizer, forget_data_list, retain_data_list, args)
