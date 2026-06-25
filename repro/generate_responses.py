"""Generate text responses from a (possibly unlearned) model for the text-based
detector. Output JSON is a list of [user, assistant] message pairs, matching the
format detection/classify_responses.py consumes (reads data[i][-1]["content"]).

Prompt formats follow the paper (Appendix A): MMLU shows question + choices and
asks for analysis then final answer; WMDP shows only the question; UltraChat uses
the raw user prompt.
"""

import argparse
import json
import os
import random

from datasets import load_dataset
from vllm import LLM, SamplingParams


def build_prompts(dataset, num_samples, data_dir, seed):
    rng = random.Random(seed)
    if dataset == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        idx = rng.sample(range(len(ds)), min(num_samples, len(ds)))
        return [
            f"{ds[i]['question'].strip()}\n{ds[i]['choices']}\n\n"
            "Please provide your analysis, then give the final answer.\n\nAnalysis:"
            for i in idx
        ]
    elif dataset == "wmdp":
        rows = []
        for sub in ("bio", "cyber"):
            p = os.path.join(data_dir, "wmdp-mcqs", f"{sub}_questions.json")
            if os.path.exists(p):
                rows.extend(json.load(open(p)))
        idx = rng.sample(range(len(rows)), min(num_samples, len(rows)))
        return [rows[i]["question"].strip() for i in idx]
    elif dataset == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        idx = rng.sample(range(len(ds)), min(num_samples, len(ds)))
        return [ds[i]["prompt"] for i in idx]
    else:
        raise ValueError(dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset", required=True, choices=["mmlu", "wmdp", "ultrachat"])
    ap.add_argument("--num_samples", type=int, default=3000)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prompts = build_prompts(args.dataset, args.num_samples, args.data_dir, args.seed)
    print(f"{len(prompts)} prompts for {args.dataset}", flush=True)

    llm = LLM(args.model_path, tensor_parallel_size=args.num_gpus, trust_remote_code=True,
              dtype="bfloat16", gpu_memory_utilization=0.9)
    tok = llm.get_tokenizer()

    formatted = []
    for p in prompts:
        if tok.chat_template is not None:
            formatted.append(tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True))
        else:
            formatted.append(p)

    outputs = llm.generate(
        formatted, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

    data = []
    for p, o in zip(prompts, outputs):
        data.append([{"role": "user", "content": p},
                     {"role": "assistant", "content": o.outputs[0].text}])

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} responses -> {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
