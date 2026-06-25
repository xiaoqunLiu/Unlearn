# Stealth: hiding unlearning traces from the activation detector

Two objectives, built on the **activation-based** detector (not the text/response one):

1. **Detection** (defensive): train a classifier to decide *whether a model has been
   unlearned* — clean original **A** vs RMU/NPO-unlearned **B** — from its pre-logit
   (`model.norm`) activations. This is exactly `detection/classify_activations.py` /
   `repro/run_probe.py` (already reproduced: NPO 100%, Llama-RMU ~99%, Zephyr-RMU subtler).

2. **Stealth attack** (adversarial branch): fine-tune an already-unlearned model into **B'**
   that **keeps the forgetting effect, keeps capability, but fools the detector**.

This doc covers (2). Code: `stealth/`.

## The GAN

| role | who | objective |
|------|-----|-----------|
| Discriminator **D** | the activation probe | from an activation feature, predict clean(0) vs unlearned(1) |
| Generator **G** | unlearned LLM **B** = frozen unlearned checkpoint **+ LoRA adapter** | keep unlearning, keep capability, make D output "clean" on B's activations |

Alternating loop (`stealth/gan.py`), matching the intended algorithm:

1. **Freeze B, train D** for `d_steps` to separate clean **A** from the current **B**.
2. **Freeze D, train B** (only the LoRA adapter) for `g_steps`:

   ```
   L_G = lambda_adv * CE( D(feat_B), "clean" )          # adversarial: fool the detector
       + lambda_keep * KL( B0 || B ) over MMLU + WMDP    # preserve forgetting + utility
   ```

   where **B0** = B with the adapter **disabled** (the untouched unlearned model). B0 already
   forgets WMDP and answers MMLU, so the KL term preserves *both* properties; the adapter's
   only remaining freedom is to **move internal activations without moving outputs**. That gap
   is precisely what the stealth attack exploits.

### Why a LoRA adapter
The adapter is the "stealth perturbation": small, and it gives a free behaviorally-identical
reference (`B0` = adapter off) for the KL term. `merge_and_unload` later bakes it into a
standalone checkpoint.

### The differentiability trick (the crux)
The detector reads activations over a **generated** response, but `model.generate` (argmax) is
non-differentiable — so D's gradient cannot reach G through it. Solution (`stealth/features.py`):
generate the response **once, no-grad, detached**, then **teacher-force** the model on the fixed
`[prompt; response]` sequence and read the `model.norm` activations from *that* forward pass.
That forward is differentiable w.r.t. the LoRA weights, and the feature is identical to what the
real detector consumes (activations at positions `[-(cont_len+1):-1]`, i.e. the states that
generate each response token). B's responses are **regenerated fresh every step** (per sampled
batch), so the optimized feature tracks what the deployed model actually produces — the evaluator
reads freshly-generated responses, so optimizing stale cached responses left a train/eval gap.

To match the real detector's inputs exactly, prompts go through the same chat template
(`apply_chat_template` when the tokenizer has one) and the same 512-token truncation as
`repro/extract_activations.py`. Pass `--no_chat_template` for base models without a chat template
(e.g. `meta-llama/Meta-Llama-3.1-8B`), matching the extractor's flag of the same name.

Clean-**A** features are precomputed once and cached, so training holds only **B** in GPU memory.

## Run

Train the stealth adapter (≈ tens of minutes on one 49 GB GPU for the defaults):

```bash
CUDA_VISIBLE_DEVICES=0 python -m stealth.gan \
    --clean_model HuggingFaceH4/zephyr-7b-beta \
    --unlearned_model repro/models/zephyr_rmu \
    --output_dir repro/models/zephyr_rmu_stealth \
    --num_prompts 256 --cont_len 64 --batch_size 8 \
    --iterations 200 --d_steps 5 --g_steps 5 \
    --lambda_adv 1.0 --lambda_keep 1.0
```

(For a base model without a chat template, add `--no_chat_template`.)

Key knobs:
- `--lambda_adv` / `--lambda_keep`: the **stealth ↔ fidelity trade-off**. Higher `lambda_keep`
  preserves forgetting/utility but evades less; higher `lambda_adv` evades more but risks drifting
  B's behavior (watch `L_keep(KL)` in the log — it should stay small).
- `--cont_len`: response length used for the feature. Match the evaluation detector's length
  (the real pipeline uses 100) for the best transfer; smaller is cheaper to train.
- `--lora_r`, `--lora_alpha`: adapter capacity.

The log prints, every 10 iterations, `D_acc(holdout)` (a freshly-adapted D's accuracy at telling
A from B — lower = stealthier under that round's D), `L_adv`, and `L_keep(KL)`.

## Evaluate against the REAL detector

The in-loop D is a moving surrogate; the real test is whether **B'** fools the
generation-based detector you already built. Merge, then run the real pipeline:

```bash
# 1. bake the adapter into a standalone checkpoint
python -m stealth.merge_adapter \
    --base repro/models/zephyr_rmu --adapter repro/models/zephyr_rmu_stealth \
    --output repro/models/zephyr_rmu_stealth_merged

# 2. extract activations from the stealth model (same as any model)
for ds in mmlu wmdp; do
  CUDA_VISIBLE_DEVICES=0 python repro/extract_activations.py \
    --model_path repro/models/zephyr_rmu_stealth_merged --label zephyr-rmu-stealth \
    --dataset $ds --num_samples 1500 --output_dir repro/activations
done

# 3. detector accuracy: clean vs stealth-unlearned  (success = this DROPS vs ~95% for plain RMU)
python repro/run_probe.py --activation_dir repro/activations \
    --orig_label zephyr --unlearn_label zephyr-rmu-stealth \
    --samples_per_dataset 1500 --n_test_per_dataset 355
```

**Success criteria** (report all three — stealth is only meaningful if the first two hold):
- **Forgetting kept**: WMDP forget-set accuracy/UE stays as low as plain RMU/NPO
  (generate WMDP MCQ answers and score, or compare forget-set loss to `B0`).
- **Capability kept**: MMLU accuracy / general fluency stays close to `B0`
  (the `L_keep(KL)` term targets this directly).
- **Detection evaded**: the clean-vs-`B'` probe accuracy drops materially below the
  clean-vs-`B` accuracy.

## Files
- `stealth/features.py` — differentiable activation feature (generate-then-teacher-force), KL helper.
- `stealth/gan.py` — discriminator + alternating GAN training loop; saves the LoRA adapter.
- `stealth/merge_adapter.py` — merge adapter → standalone checkpoint for the real-detector eval.
