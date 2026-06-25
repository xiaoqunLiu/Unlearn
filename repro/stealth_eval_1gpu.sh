#!/bin/bash
# Single-GPU variant of stealth_eval.sh (so it can run on a free GPU while another
# trains). Three-way eval of a Llama-RMU stealth model: detection evasion, forgetting,
# utility. Usage: stealth_eval_1gpu.sh <adapter_dir> <label> <gpu_id> [merged_dir]
set -e
cd /egr/research-seit/xl/Unlearn
export HF_HOME=/egr/research-seit/xl/cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ADAPTER=$1
LABEL=$2
GPU=${3:-0}
MERGED=${4:-${ADAPTER}_merged}
N=1500
export CUDA_VISIBLE_DEVICES=$GPU
CR="conda run --no-capture-output -n generation python"

echo "### 1. merge ${ADAPTER} -> ${MERGED} (gpu ${GPU}) ###"
$CR -m stealth.merge_adapter --base repro/models/llama_rmu --adapter "$ADAPTER" --output "$MERGED"

echo "### 2. extract stealth activations (mmlu then wmdp, gpu ${GPU}) ###"
for ds in mmlu wmdp; do
  $CR repro/extract_activations.py --model_path "$MERGED" \
    --label "$LABEL" --dataset $ds --num_samples $N --batch_size 24 \
    --output_dir repro/activations_2900
done
echo "stealth activations done"

echo "### 3. detection probe: clean(llama) vs ${LABEL} ###"
$CR repro/run_probe.py --activation_dir repro/activations_2900 \
  --orig_label llama --unlearn_label "$LABEL" --samples_per_dataset $N --n_test_per_dataset 355 \
  --epochs 30 --lr 1e-3 --batch_size 64 --dropout 0.3 2>&1 | grep -E "MMLU|WMDP|MEAN" | sed "s/^/[clean-vs-${LABEL}] /"

echo "### 4. MCQ accuracy (forgetting=WMDP, utility=MMLU) ###"
$CR repro/eval_mcq.py --model_path "$MERGED" --dataset wmdp --num_samples 1000 2>&1 | grep "acc="
$CR repro/eval_mcq.py --model_path "$MERGED" --dataset mmlu --num_samples 1000 2>&1 | grep "acc="
echo "### done ${LABEL} ###"
