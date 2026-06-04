"""RMU: Representation Misdirection for Unlearning.

Trains an unlearned model by (1) steering the hidden activations of a chosen
layer toward a random "control" direction on the forget corpora, while (2)
keeping the activations on retain data close to a frozen copy of the original
model. Only a small subset of decoder-layer MLP parameters is updated.

This is the training step that produces the `*_rmu` checkpoints consumed by
`generation/generate_response.py` / `generation/generate_activations.py`.

Run from the repository root:

    python -m unlearning.rmu.unlearn \
        --model_name_or_path HuggingFaceH4/zephyr-7b-beta \
        --output_dir zephyr_rmu \
        --forget_corpora bio-forget-corpus,cyber-forget-corpus \
        --retain_corpora wikitext,wikitext \
        --steering_coeffs 6.5,6.5 --alphas 1200,1200 \
        --layer_id 7 --layer_ids 5,6,7 --param_ids 6
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.optim import AdamW

try:  # `python -m unlearning.rmu.unlearn` from the repo root
    from unlearning.rmu.utils import load_model, get_params, forward_with_cache, get_data
except ModuleNotFoundError:  # pragma: no cover  (direct-file run fallback)
    from utils import load_model, get_params, forward_with_cache, get_data


def run_rmu(updated_model, frozen_model, tokenizer,
            forget_data_list, retain_data_list, args):
    print("====== RMU config ======")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("========================")

    updated_model.train()
    params = get_params(updated_model, args.layer_ids, args.param_ids)
    optimizer = AdamW(params, lr=args.lr)

    module_str = "{model_name}.model.layers[{layer_id}]"
    frozen_module = eval(module_str.format(model_name="frozen_model", layer_id=args.layer_id))
    updated_module = eval(module_str.format(model_name="updated_model", layer_id=args.layer_id))

    # One fixed random unit control vector per forget topic, scaled by its coeff.
    control_vectors_list = []
    for i in range(len(forget_data_list)):
        random_vector = torch.rand(
            1, 1, updated_model.config.hidden_size,
            dtype=updated_model.dtype, device=updated_model.device,
        )
        control_vec = random_vector / torch.norm(random_vector) * args.steering_coeffs[i]
        control_vectors_list.append(control_vec)

    num_batches = min(
        args.max_num_batches,
        min(len(f) for f in forget_data_list),
        min(len(r) for r in retain_data_list),
    )

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    for epoch in range(args.epochs):
        print(f"===== epoch {epoch} =====")
        for idx in range(num_batches):
            topic_idx = idx % len(forget_data_list)
            batch_idx = idx // len(forget_data_list)
            control_vec = control_vectors_list[topic_idx]
            unlearn_batch = forget_data_list[topic_idx][batch_idx]
            retain_batch = retain_data_list[topic_idx][batch_idx]

            # ---- Unlearning (forget) loss: push activations to the control vector.
            max_length = 512 if topic_idx == 0 else 768
            unlearn_inputs = tokenizer(
                unlearn_batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(updated_model.device)
            updated_forget_act = forward_with_cache(
                updated_model, unlearn_inputs, module=updated_module, no_grad=False
            ).to(updated_model.device)
            unlearn_loss = torch.nn.functional.mse_loss(
                updated_forget_act, control_vec.to(updated_forget_act.dtype)
            )

            # ---- Retain loss: keep retain activations close to the frozen model.
            retain_inputs = tokenizer(
                retain_batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(updated_model.device)
            updated_retain_act = forward_with_cache(
                updated_model, retain_inputs, module=updated_module, no_grad=False
            ).to(updated_model.device)
            frozen_retain_act = forward_with_cache(
                frozen_model, retain_inputs, module=frozen_module, no_grad=True
            ).to(updated_model.device)
            retain_loss = torch.nn.functional.mse_loss(
                updated_retain_act, frozen_retain_act
            ) * args.alphas[topic_idx]

            loss = unlearn_loss + retain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"[{idx}/{num_batches}] topic={topic_idx} "
                  f"unlearn={unlearn_loss.item():.4f} retain={retain_loss.item():.4f}")

    tokenizer.truncation_side = truncation_side

    os.makedirs(args.output_dir, exist_ok=True)
    updated_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved unlearned model to: {args.output_dir}")


def get_args():
    parser = argparse.ArgumentParser(
        description="RMU unlearning (Representation Misdirection for Unlearning)")
    # model / output
    parser.add_argument("--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta",
                        help="base model to unlearn (HF id or local path)")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="tokenizer to use (defaults to --model_name_or_path)")
    parser.add_argument("--output_dir", type=str, default="zephyr_rmu",
                        help="directory to save the unlearned checkpoint")
    # data
    parser.add_argument("--forget_corpora", type=str, default="bio-forget-corpus,cyber-forget-corpus",
                        help="comma-separated forget corpus names under --data_dir")
    parser.add_argument("--retain_corpora", type=str, default="wikitext,wikitext",
                        help="comma-separated retain corpus names ('wikitext' streams from HF)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="directory holding the {name}.jsonl corpora")
    # rmu hyperparameters
    parser.add_argument("--steering_coeffs", type=str, default="6.5,6.5",
                        help="comma-separated steering coefficient per forget corpus")
    parser.add_argument("--alphas", type=str, default="1200,1200",
                        help="comma-separated retain-loss weight per forget corpus")
    parser.add_argument("--layer_id", type=int, default=7,
                        help="layer whose activations are steered/cached")
    parser.add_argument("--layer_ids", type=str, default="5,6,7",
                        help="comma-separated layers whose parameters are updated")
    parser.add_argument("--param_ids", type=str, default="6",
                        help="comma-separated parameter indices within each layer to update")
    # optimization
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="number of passes over the batches")
    parser.add_argument("--max_num_batches", type=int, default=150, help="max number of update steps")
    parser.add_argument("--batch_size", type=int, default=4, help="documents per mini-batch")
    parser.add_argument("--min_len", type=int, default=50, help="drop documents shorter than this")
    parser.add_argument("--max_len", type=int, default=2000, help="reserved; tokenizer truncation bounds length")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()
    args.forget_corpora = args.forget_corpora.split(",")
    args.retain_corpora = args.retain_corpora.split(",")
    args.steering_coeffs = [float(c) for c in str(args.steering_coeffs).split(",")]
    args.alphas = [float(a) for a in str(args.alphas).split(",")]
    args.layer_ids = [int(x) for x in str(args.layer_ids).split(",")]
    args.param_ids = [int(x) for x in str(args.param_ids).split(",")]
    return args


if __name__ == "__main__":
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    frozen_model, tokenizer = load_model(args.model_name_or_path, args.tokenizer_name_or_path)
    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad_(False)

    updated_model, _ = load_model(args.model_name_or_path, args.tokenizer_name_or_path)

    forget_data_list, retain_data_list = get_data(
        args.forget_corpora, args.retain_corpora,
        args.min_len, args.max_len, args.batch_size, args.data_dir,
    )

    run_rmu(updated_model, frozen_model, tokenizer,
            forget_data_list, retain_data_list, args)
