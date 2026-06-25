#!/bin/bash
# Single-GPU eval of a ZEPHYR-RMU stealth model (detection evasion / forgetting / utility).
# Zephyr HAS a chat template, so extraction auto-applies it (do NOT pass --no_chat_template) —
# matching how the clean `zephyr` baseline activations were extracted.
# Usage: stealth_eval_zephyr.sh <adapter_dir> <label> <gpu_id> [merged_dir]
set -e
cd /egr/research-seit/xl/Unlearn
export HF_HOME=/egr/research-seit/xl/cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ADAPTER=$1
LABEL=$2
GPU=${3:-0}
MERGED=${4:-${ADAPTER}_merged}
N=1500
ACTDIR=repro/activations_2900
export CUDA_VISIBLE_DEVICES=$GPU
CR="conda run --no-capture-output -n generation python"

echo "### 1. merge ${ADAPTER} -> ${MERGED} (base zephyr_rmu, gpu ${GPU}) ###"
$CR -m stealth.merge_adapter --base repro/models/zephyr_rmu --adapter "$ADAPTER" --output "$MERGED"

echo "### 2. extract stealth activations (mmlu then wmdp, chat template ON, gpu ${GPU}) ###"
for ds in mmlu wmdp; do
  $CR repro/extract_activations.py --model_path "$MERGED" \
    --label "$LABEL" --dataset $ds --num_samples $N --batch_size 24 --output_dir $ACTDIR
done
echo "stealth activations done"

echo "### 3a. BASELINE detection probe: clean(zephyr) vs zephyr-rmu (per-benchmark) ###"
$CR repro/run_probe.py --activation_dir $ACTDIR \
  --orig_label zephyr --unlearn_label zephyr-rmu --samples_per_dataset $N --n_test_per_dataset 355 \
  --epochs 30 --lr 1e-3 --batch_size 64 --dropout 0.3 2>&1 | grep -E "MMLU|WMDP|MEAN" | sed "s/^/[BASE clean-vs-zephyr-rmu] /"

echo "### 3b. STEALTH detection probe: clean(zephyr) vs ${LABEL} (per-benchmark) ###"
$CR repro/run_probe.py --activation_dir $ACTDIR \
  --orig_label zephyr --unlearn_label "$LABEL" --samples_per_dataset $N --n_test_per_dataset 355 \
  --epochs 30 --lr 1e-3 --batch_size 64 --dropout 0.3 2>&1 | grep -E "MMLU|WMDP|MEAN" | sed "s/^/[clean-vs-${LABEL}] /"

echo "### 4. MCQ accuracy (forgetting=WMDP, utility=MMLU) ###"
$CR repro/eval_mcq.py --model_path "$MERGED" --dataset wmdp --num_samples 1000 2>&1 | grep "acc="
$CR repro/eval_mcq.py --model_path "$MERGED" --dataset mmlu --num_samples 1000 2>&1 | grep "acc="
echo "### done ${LABEL} ###"
