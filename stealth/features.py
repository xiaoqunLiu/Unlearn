"""Differentiable pre-logit activation features for the stealth GAN.

The unlearning-trace detector classifies a model from the `model.norm` (pre-logit)
activations recorded over a *generated* response. Generation is non-differentiable,
so we cannot push the discriminator's gradient into the generator through it.

Workaround: generate the response ONCE (no-grad, detached), then **teacher-force**
the model on the fixed `[prompt; response]` sequence and read the `model.norm`
activations from that forward pass. That forward IS differentiable w.r.t. the
model's weights, so the discriminator gradient reaches the generator (the LoRA
adapter), while the feature stays identical to what the real detector consumes.
"""

import contextlib

import torch


def get_norm_module(model):
    """Final RMSNorm (pre-logit) module, unwrapping a PEFT/LoRA wrapper if present."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.norm


def apply_chat(tokenizer, prompts, use_chat):
    """Optionally wrap each prompt in the model's chat template — MUST match the
    real detector (repro/extract_activations.py), which applies the template
    whenever the tokenizer has one. Training on raw prompts while the evaluator
    uses chat-formatted prompts is a silent train/eval mismatch."""
    if not use_chat:
        return list(prompts)
    return [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True)
            for p in prompts]


def build_prompt_batch(tokenizer, prompts, device, max_length=512):
    """Left-padded tokenization of a list of prompt strings. `max_length=512`
    matches repro/extract_activations.py so prompts are truncated identically."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_length)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def generate_full(model, input_ids, attn_mask, cont_len, pad_token_id):
    """Greedy-decode `cont_len` tokens; return the full `[prompt; response]` ids
    and a matching attention mask. `eos_token_id=None` forces exactly cont_len steps
    so every row has the response in its last `cont_len` positions.
    """
    out = model.generate(input_ids=input_ids, attention_mask=attn_mask,
                         do_sample=False, max_new_tokens=cont_len,
                         eos_token_id=None, pad_token_id=pad_token_id)
    full_mask = torch.cat(
        [attn_mask, torch.ones(attn_mask.shape[0], cont_len, dtype=attn_mask.dtype, device=attn_mask.device)],
        dim=1)
    return out, full_mask


def activations_and_logits(model, full_ids, full_mask, cont_len, norm_module, grad):
    """Teacher-force `model` on `full_ids` and return:
      feat:   [B, cont_len*hidden]  — the activations that *generate* each response
              token (positions [-(cont_len+1):-1]), matching the real detector.
      logits: [B, T, vocab]
    Differentiable w.r.t. `model` params when `grad=True`.
    """
    cache = {}

    def hook(_m, _i, out):
        cache["h"] = out[0] if isinstance(out, tuple) else out

    handle = norm_module.register_forward_hook(hook)
    cm = contextlib.nullcontext() if grad else torch.no_grad()
    try:
        with cm:
            out = model(input_ids=full_ids, attention_mask=full_mask)
    finally:
        handle.remove()
    acts = cache["h"]                                  # [B, T, hidden]
    feat = acts[:, -(cont_len + 1):-1, :].reshape(acts.shape[0], -1)
    return feat, out.logits


@torch.no_grad()
def logits_only(model, full_ids, full_mask):
    return model(input_ids=full_ids, attention_mask=full_mask).logits


def masked_kl(logits_teacher, logits_student, cont_len):
    """KL( teacher || student ) averaged over the response (last cont_len) positions."""
    lt = logits_teacher[:, -cont_len:, :].float()
    ls = logits_student[:, -cont_len:, :].float()
    pt = torch.softmax(lt, dim=-1)
    log_ps = torch.log_softmax(ls, dim=-1)
    log_pt = torch.log_softmax(lt, dim=-1)
    return (pt * (log_pt - log_ps)).sum(-1).mean()
