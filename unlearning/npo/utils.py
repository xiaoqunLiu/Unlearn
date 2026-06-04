"""Utilities for NPO (Negative Preference Optimization) unlearning.

Model loading and corpus reading are shared with the RMU package so the two
methods operate on identical data; the NPO-specific helpers below compute the
per-sequence likelihoods that the NPO loss requires.
"""

import os
import sys

import torch
import torch.nn as nn

# Make the repo root importable regardless of how this file is launched.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_REPO_ROOT)
from unlearning.rmu.utils import load_model, get_data  # noqa: E402  (reuse identical loaders)


def first_device(model):
    """Device of the model's first parameter (works with device_map='auto')."""
    return next(model.parameters()).device


def tokenize_batch(tokenizer, texts, device, max_length):
    """Tokenize a list of documents, masking pad positions in the labels."""
    enc = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    )
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels
    return {k: v.to(device) for k, v in enc.items()}


def get_batch_loss(logits, labels):
    """Per-sequence summed token NLL, with no reduction across the batch.

    Returns a tensor of shape [batch]; entry i is -log p(doc_i) under the model.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    per_token = loss_fct(shift_logits.transpose(-1, -2), shift_labels)  # [B, T-1]
    return per_token.sum(dim=-1)  # [B]
