## RMU & NPO Unlearning

This step produces the **unlearned models** (`*-rmu` / `*-npo`) that are later
probed for detectable traces. Both trainers read the WMDP corpora from `./data`
— make sure they are in place first (see [Data.md](./Data.md): the forget /
retain `.jsonl` files should live under `./data`).

Both methods share the same data loaders (`unlearning/rmu/utils.py`), so RMU and
NPO see identical forget / retain corpora; only the unlearning objective differs.

---

### RMU (Representation Misdirection for Unlearning)

Reference: Li et al., *The WMDP Benchmark* (https://arxiv.org/pdf/2403.03218).

RMU steers the hidden activations of a chosen layer toward a **random control
direction** on the forget corpora (forget loss), while a retain loss keeps
activations on retain data close to a frozen copy of the original model. Only a
small subset of decoder-layer MLP parameters is updated. Code:
`unlearning/rmu/unlearn.py`, `unlearning/rmu/utils.py`.

Run from the repository root:

```bash
python -m unlearning.rmu.unlearn \
    --model_name_or_path HuggingFaceH4/zephyr-7b-beta \
    --output_dir zephyr_rmu \
    --forget_corpora bio-forget-corpus,cyber-forget-corpus \
    --retain_corpora wikitext,wikitext \
    --steering_coeffs 6.5,6.5 \
    --alphas 1200,1200 \
    --layer_id 7 --layer_ids 5,6,7 --param_ids 6 \
    --lr 5e-5 --max_num_batches 150 --batch_size 4
```

**Key parameters**:
- `--output_dir`: Where the unlearned checkpoint is saved. Use the path the
  generation scripts expect (`zephyr_rmu/`, `yi_rmu/`, `llama-rmu/`,
  `Qwen2.5-14B-coeff460-alpha350/`).
- `--forget_corpora` / `--retain_corpora`: Comma-separated corpus names. Each
  maps to `./data/{name}.jsonl`; the special name `wikitext` streams the generic
  retain set from HuggingFace.
- `--steering_coeffs`: Steering coefficient `c` per forget corpus (the `coeff…`).
- `--alphas`: Retain-loss weight per forget corpus (the `alpha…`).
- `--layer_id`: Layer whose activations are steered/cached.
- `--layer_ids`: Layers whose parameters are updated.
- `--param_ids`: Parameter indices within each layer (index `6` is the MLP
  `down_proj` weight for Llama / Mistral / Qwen2 style decoders).

The defaults above (`steering_coeffs 6.5`, `alphas 1200`, `lr 5e-5`,
`max_num_batches 150`, layer 7) are the official Zephyr-7b RMU settings from the
WMDP repository.

---

### NPO (Negative Preference Optimization)

Reference: Zhang et al., *Negative Preference Optimization*
(https://arxiv.org/pdf/2404.05868).

NPO replaces gradient ascent on the forget set with a preference-style loss that
down-weights samples already pushed below the reference model, avoiding
catastrophic collapse:

```
L_NPO(beta) = (2/beta) * E_forget[ log(1 + (pi_theta/pi_ref)^beta) ]
```

We use the **gradient-difference (`grad_diff`)** variant, which adds a standard
language-modeling loss on the retain set:

```
L = L_NPO + retain_coeff * L_retain
```

`pi_ref` is a frozen copy of the original model. Code:
`unlearning/npo/unlearn.py`, `unlearning/npo/utils.py` (data loading is shared
with `unlearning/rmu/utils.py`).

Run from the repository root:

```bash
python -m unlearning.npo.unlearn \
    --model_name_or_path HuggingFaceH4/zephyr-7b-beta \
    --output_dir zephyr_npo \
    --forget_corpora bio-forget-corpus,cyber-forget-corpus \
    --retain_corpora wikitext,wikitext \
    --beta 0.1 --retain_coeff 1.0 \
    --lr 7e-6 --max_steps 140 --batch_size 4 --grad_accum 4 --warmup_steps 1
```

**Key parameters**:
- `--beta`: NPO temperature (smaller ⇒ gentler unlearning). Paper/recipe default `0.1`.
- `--retain_coeff`: Weight of the gradient-difference retain loss (`gdcoeff`).
- `--lr`, `--max_steps`, `--batch_size`, `--grad_accum`, `--warmup_steps`:
  Optimization schedule (effective batch = `batch_size × grad_accum`).
- `--gradient_checkpointing` / `--no_gradient_checkpointing`: NPO fine-tunes all
  parameters, so checkpointing is on by default to save memory; large models
  (e.g., Yi-34B, Qwen-14B) typically need multiple GPUs.

These defaults match the `npo_grad_diff` recipe the original checkpoints were
trained with (`beta0.1`, `gdcoeff1.0`, `batch4`, `accum4`, `warmup1`,
`mstep140`); the per-model learning rate is the main knob.

---

### Per-model hyperparameters

The paper unlearns four instruction-tuned models — **Zephyr-7B, Llama-3.1-8B,
Qwen2.5-14B, Yi-34B-Chat** — on the WMDP benchmark. The exact settings below are
from Appendix A (Tables A1 and A2) of
[arXiv:2506.14003](https://arxiv.org/abs/2506.14003).

**RMU** (`c` = `--steering_coeffs`, `γ` = `--alphas`; same value is used for both
the bio and cyber forget corpora). `--param_ids 6` (MLP `down_proj`) for all
models; common optimization is `--lr 5e-5 --max_num_batches 150 --batch_size 4`:

| Model         | `--steering_coeffs` (c) | `--alphas` (γ) | `--layer_id` | `--layer_ids` |
|---------------|:-----------------------:|:--------------:|:------------:|:-------------:|
| Zephyr-7B     | 6.5                     | 1200           | 7            | 5,6,7         |
| Llama-3.1-8B  | 45                      | 1300           | 7            | 5,6,7         |
| Qwen2.5-14B   | 460                     | 350            | 10           | 8,9,10        |
| Yi-34B-Chat   | 300                     | 350            | 15           | 13,14,15      |

(The `Qwen2.5-14B-coeff460-alpha350` checkpoint name encodes its `c`=460 / `γ`=350.)

**NPO** (gradient-difference; `γ` = `--retain_coeff`). All models use
`--beta 0.1 --max_steps 140 --batch_size 4` (with `--grad_accum 4`); only the
learning rate and `γ` differ:

| Model         | `--lr` | `--retain_coeff` (γ) |
|---------------|:------:|:--------------------:|
| Zephyr-7B     | 7e-6   | 1.0                  |
| Llama-3.1-8B  | 2e-5   | 2.0                  |
| Qwen2.5-14B   | 7e-5   | 1.0                  |
| Yi-34B-Chat   | 6e-5   | 1.0                  |

The script defaults reproduce the **Zephyr-7B** row for both methods; pass the
flags above to reproduce the other models. The RMU `--layer_id` / `--layer_ids`
match the activation hooks in `generation/generate_activations.py`.

---

### Next steps (reproducing the paper)

1. Train both unlearned variants for each base model with the commands above,
   saving to the `*_rmu` / `*_npo` paths the generation scripts reference.
2. Generate model outputs:
   - Text responses — [Response.md](./Response.md) (`generation/generate_response.py`).
   - Activations — [Response.md](./Response.md) (`generation/generate_activations.py`;
     the `*-rmu` / `*-npo` entries in `model_args_to_name` already point at the
     output dirs above).
3. Train the trace detectors — [Classification.md](./Classification.md)
   (`detection/classify_responses.py` for text outputs,
   `detection/classify_activations.py` for activations).
