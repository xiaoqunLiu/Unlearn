#!/bin/bash
# Three-way evaluation of a Llama-RMU stealth model:
#   (1) detection evasion  (2) forgetting retention  (3) utility retention
# Usage: stealth_eval.sh <adapter_dir> <label> [merged_dir]
set -e
cd /egr/research-seit/xl/Unlearn
export HF_HOME=/egr/research-seit/xl/cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ADAPTER=${1:-repro/models/llama_rmu_stealth}
LABEL=${2:-llama-rmu-stealth}
MERGED=${3:-${ADAPTER}_merged}
N=1500
CR="conda run --no-capture-output -n generation python"

echo "### 1. merge ${ADAPTER} -> ${MERGED} ###"
$CR -m stealth.merge_adapter --base repro/models/llama_rmu --adapter "$ADAPTER" --output "$MERGED"

echo "### 2. extract stealth activations (mmlu GPU0, wmdp GPU1) ###"
CUDA_VISIBLE_DEVICES=0 $CR repro/extract_activations.py --model_path "$MERGED" \
  --label "$LABEL" --dataset mmlu --num_samples $N --batch_size 24 \
  --output_dir repro/activations_2900 > repro/logs/ext_${LABEL}_mmlu.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $CR repro/extract_activations.py --model_path "$MERGED" \
  --label "$LABEL" --dataset wmdp --num_samples $N --batch_size 24 \
  --output_dir repro/activations_2900 > repro/logs/ext_${LABEL}_wmdp.log 2>&1 &
wait
echo "stealth activations done"

echo "### 3. detection probe: clean vs ${LABEL} ###"
CUDA_VISIBLE_DEVICES=0 $CR repro/run_probe.py --activation_dir repro/activations_2900 \
  --orig_label llama --unlearn_label "$LABEL" --samples_per_dataset $N --n_test_per_dataset 355 \
  --epochs 30 --lr 1e-3 --batch_size 64 --dropout 0.3 2>&1 | grep -E "MMLU|WMDP|MEAN" | sed "s/^/[clean-vs-${LABEL}] /"

echo "### 4. MCQ accuracy (forgetting=WMDP, utility=MMLU) ###"
CUDA_VISIBLE_DEVICES=0 $CR repro/eval_mcq.py --model_path "$MERGED" --dataset wmdp --num_samples 1000 2>&1 | grep "acc="
CUDA_VISIBLE_DEVICES=0 $CR repro/eval_mcq.py --model_path "$MERGED" --dataset mmlu --num_samples 1000 2>&1 | grep "acc="
echo "### done ###"
