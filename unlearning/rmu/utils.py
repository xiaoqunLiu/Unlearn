"""Utilities for RMU (Representation Misdirection for Unlearning) training.

These helpers load the base model, expose the subset of parameters that RMU
updates, cache intermediate activations during a forward pass, and read the
WMDP forget / retain corpora into text mini-batches.
"""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name_or_path, tokenizer_name_or_path=None):
    """Load a causal LM and its tokenizer in bf16, sharded across visible GPUs."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer


def get_params(model, layer_ids, param_ids):
    """Return the decoder-layer parameters RMU updates.

    `layer_ids` selects which transformer blocks to touch, and `param_ids`
    selects which parameters within each block (by their order in
    `layer.parameters()`; index 6 is the MLP `down_proj` weight for Llama /
    Mistral / Qwen2 style decoders).
    """
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params


def forward_with_cache(model, inputs, module, no_grad=True):
    """Run a forward pass and return the output activations of `module`."""
    cache = []

    def hook(module, inp, out):
        cache.append(out[0] if isinstance(out, tuple) else out)
        return None

    handle = module.register_forward_hook(hook)
    try:
        if no_grad:
            with torch.no_grad():
                model(**inputs)
        else:
            model(**inputs)
    finally:
        handle.remove()
    return cache[0]


def _read_corpus_file(path, min_len):
    """Read a `.jsonl` corpus, tolerating both `{"text": ...}` and raw-line formats."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "text" in obj:
                    text = obj["text"]
                elif isinstance(obj, str):
                    text = obj
                else:
                    text = line
            except json.JSONDecodeError:
                text = line
            if len(text) > min_len:
                data.append(str(text))
    return data


def get_data(forget_corpora, retain_corpora, min_len=50, max_len=2000,
             batch_size=4, data_dir="data"):
    """Load forget / retain corpora into lists of text mini-batches.

    Each corpus name maps to `{data_dir}/{name}.jsonl`, except the special name
    "wikitext", which is streamed from HuggingFace (wikitext-2-raw-v1 test
    split) and used as a generic retain set.
    """
    def get_dataset(name):
        if name == "wikitext":
            from datasets import load_dataset
            raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            data = [str(x) for x in raw["text"] if len(x) > min_len]
        else:
            data = _read_corpus_file(f"{data_dir}/{name}.jsonl", min_len)
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    return (
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora],
    )
