#!/bin/bash
# Auto-eval driver for the Zephyr-RMU covariance-match stealth runs.
# Waits for each adapter to finish training (which also frees its GPU), then runs the
# decisive eval against a RETRAINED probe + MCQ forgetting/utility. Mirrors the Llama
# stealth_eval_1gpu.sh flow but for Zephyr (base=zephyr_rmu, clean baseline=zephyr,
# clean+rmu activations already in repro/activations at N=1500).
#
# Usage: zcov_eval.sh [gpu] [name1 name2 ...]   (default gpu=0, names="zcov0 zcov5")
set -u
cd /egr/research-seit/xl/Unlearn
export HF_HOME=/egr/research-seit/xl/cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CR="conda run --no-capture-output -n generation python"
ADIR=repro/activations
N=1500
GPU=${1:-0}; shift || true
NAMES=${*:-"zcov0 zcov5"}
export CUDA_VISIBLE_DEVICES=$GPU
# beefy probe (same as stealth_eval_1gpu) — give the detector its best shot at catching the attack
PROBE="--samples_per_dataset $N --n_test_per_dataset 355 --epochs 30 --lr 1e-3 --batch_size 64 --dropout 0.3"

logf() { case "$1" in zcov0) echo repro/logs/zcov0.log;; zcov5) echo repro/logs/zcov5.log;;
                      zcovD5) echo repro/logs/zcovD5.log;; zcovD20) echo repro/logs/zcovD20.log;;
                      *) echo repro/logs/$1.log;; esac; }

wait_for() {  # $1 = adapter dir, $2 = training log
  echo ">>> waiting for $1 to finish training ..."
  while true; do
    [ -f "$1/adapter_model.safetensors" ] && grep -q "Saved covariance-match" "$2" 2>/dev/null && break
    sleep 30
  done
  echo ">>> $1 ready"
}

probe() {  # $1 = unlearn_label   -> detection (clean zephyr vs label)
  $CR repro/run_probe.py --activation_dir $ADIR --orig_label zephyr --unlearn_label "$1" $PROBE \
    2>&1 | grep -E "MMLU|WMDP|MEAN" | sed "s/^/[detect clean-vs-$1] /"
}
mcq() {    # $1 = model_path     -> forgetting (wmdp) + utility (mmlu)
  echo "[mcq $1] WMDP(forget,lower=better) / MMLU(utility,higher=better):"
  $CR repro/eval_mcq.py --model_path "$1" --dataset wmdp --num_samples 1000 2>&1 | grep "acc=" | sed "s/^/  wmdp /"
  $CR repro/eval_mcq.py --model_path "$1" --dataset mmlu --num_samples 1000 2>&1 | grep "acc=" | sed "s/^/  mmlu /"
}

echo "######## ZCOV EVAL on GPU $GPU — names: $NAMES ########"

# Wait for the first run so GPU $GPU (its trainer) frees before we use it.
FIRST=$(echo $NAMES | awk '{print $1}')
wait_for "repro/models/zephyr_rmu_${FIRST}" "$(logf $FIRST)"

echo "######## REFERENCE: plain RMU (B0) ########"
probe zephyr-rmu
mcq repro/models/zephyr_rmu

for name in $NAMES; do
  ADAPTER=repro/models/zephyr_rmu_${name}
  MERGED=${ADAPTER}_merged
  LABEL=zephyr-rmu-${name}
  wait_for "$ADAPTER" "$(logf $name)"
  echo "######## EVAL ${name} ########"
  echo "### 1. merge ###"
  $CR -m stealth.merge_adapter --base repro/models/zephyr_rmu --adapter "$ADAPTER" --output "$MERGED"
  echo "### 2. extract (mmlu, wmdp) ###"
  for ds in mmlu wmdp; do
    $CR repro/extract_activations.py --model_path "$MERGED" --label "$LABEL" \
      --dataset $ds --num_samples $N --batch_size 24 --output_dir $ADIR
  done
  echo "### 3. detection (retrained probe) ###"
  probe "$LABEL"
  echo "### 4. MCQ forgetting/utility ###"
  mcq "$MERGED"
  echo "######## done ${name} ########"
done
echo "######## ALL ZCOV EVAL DONE ########"
