"""Merge a stealth LoRA adapter into its base unlearned model, producing a
standalone checkpoint that the REAL generation-based detector can evaluate
(repro/extract_activations.py + repro/run_probe.py).

  python -m stealth.merge_adapter \
      --base repro/models/zephyr_rmu \
      --adapter repro/models/zephyr_rmu_stealth \
      --output repro/models/zephyr_rmu_stealth_merged
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base unlearned checkpoint the adapter was trained on")
    ap.add_argument("--adapter", required=True, help="stealth LoRA adapter dir")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.output)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output)
    print(f"Merged stealth model saved to {args.output}")


if __name__ == "__main__":
    main()
