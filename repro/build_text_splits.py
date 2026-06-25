"""Build train/eval response files for the text-based (LLM2Vec) detector.

From per-(dataset,label) response files repro/responses/{ds}_{label}.json, write:
  splits/{label}_sfig_train.json   = MMLU[:n_tr] + WMDP[:n_tr]   (mixed S_fig train pool)
  splits/{ds}_{label}_eval.json    = held-out 355 per benchmark (disjoint from train)
for label in {zephyr, zephyr-rmu, zephyr-npo}, ds in {mmlu, wmdp, ultrachat}.
"""

import json
import os

RESP = "repro/responses"
OUT = "repro/responses/splits"
LABELS = ["zephyr", "zephyr-rmu", "zephyr-npo"]
N_TR = 1145      # train per benchmark per model
N_EVAL = 355     # held-out test per benchmark


def load(ds, label):
    p = f"{RESP}/{ds}_{label}.json"
    return json.load(open(p)) if os.path.exists(p) else []


def main():
    os.makedirs(OUT, exist_ok=True)
    for label in LABELS:
        mmlu, wmdp = load("mmlu", label), load("wmdp", label)
        ultra = load("ultrachat", label)
        # S_fig train pool: mixed MMLU + WMDP, first N_TR of each
        train = mmlu[:N_TR] + wmdp[:N_TR]
        json.dump(train, open(f"{OUT}/{label}_sfig_train.json", "w"), ensure_ascii=False)
        # held-out per-benchmark eval (disjoint from train)
        json.dump(mmlu[N_TR:N_TR + N_EVAL], open(f"{OUT}/mmlu_{label}_eval.json", "w"), ensure_ascii=False)
        json.dump(wmdp[N_TR:N_TR + N_EVAL], open(f"{OUT}/wmdp_{label}_eval.json", "w"), ensure_ascii=False)
        if ultra:
            json.dump(ultra[:N_EVAL], open(f"{OUT}/ultrachat_{label}_eval.json", "w"), ensure_ascii=False)
        print(f"{label}: train={len(train)} "
              f"eval(mmlu/wmdp/ultra)={len(mmlu[N_TR:N_TR+N_EVAL])}/{len(wmdp[N_TR:N_TR+N_EVAL])}/{len(ultra[:N_EVAL])}")


if __name__ == "__main__":
    main()
