#!/bin/bash
# Sweep the JOINT (covariance) stealth attack over lambda_cov and evaluate each against the
# REAL retrained detector. This is the v4 experiment: does matching the joint/correlation
# structure (not just the per-dim marginals, which v3/feature_match already failed at) move
# the detector?
#
#   lambda_cov=0  -> CONTROL: reduces exactly to v3 marginal matching; should reproduce ~100%.
#   lambda_cov=1  -> add the joint-covariance term.
#   lambda_cov=5  -> push the joint term harder.
#
# Each (train + full eval) pipeline runs on its own GPU, all in parallel. Eval reuses
# repro/stealth_eval_1gpu.sh (merge -> extract -> retrained-probe detection -> WMDP/MMLU MCQ).
#
# Usage: repro/cov_sweep.sh             # GPUs 2 3 4 for lambda 0 1 5
#        repro/cov_sweep.sh 5 6 7       # custom GPUs (one per lambda, order 0 1 5)
set -u
cd /egr/research-seit/xl/Unlearn
export HF_HOME=/egr/research-seit/xl/cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CR="conda run --no-capture-output -n generation python"
mkdir -p repro/logs repro/models

LAMBDAS=(0 1 5)
GPUS=(${1:-2} ${2:-3} ${3:-4})

run_one () {
  local lam=$1 gpu=$2
  local tag="cov${lam}"
  local out="repro/models/llama_rmu_${tag}"
  local log="repro/logs/${tag}.log"
  {
    echo "######## train ${tag} | gpu ${gpu} | lambda_cov=${lam} | $(date) ########"
    CUDA_VISIBLE_DEVICES=$gpu $CR -m stealth.cov_match \
        --clean_model meta-llama/Meta-Llama-3.1-8B \
        --unlearned_model repro/models/llama_rmu \
        --output_dir "$out" --no_chat_template \
        --lambda_marg 1.0 --lambda_cov "${lam}" --proj_dim 384 --n_proj 3 \
      || { echo "TRAIN FAILED ${tag}"; exit 1; }
    echo "######## eval ${tag} | $(date) ########"
    bash repro/stealth_eval_1gpu.sh "$out" "llama-rmu-${tag}" "$gpu" \
      || { echo "EVAL FAILED ${tag}"; exit 1; }
    echo "######## DONE ${tag} | $(date) ########"
  } > "$log" 2>&1
}

echo "launching lambda_cov sweep: ${LAMBDAS[*]} on GPUs ${GPUS[*]}"
for i in 0 1 2; do
  run_one "${LAMBDAS[$i]}" "${GPUS[$i]}" &
done
wait
echo "ALL COV SWEEP RUNS COMPLETE — see repro/logs/cov{0,1,5}.log"
