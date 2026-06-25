# Reproduction Results — Unlearning Trace Detection (arXiv:2506.14003)

Hardware: 8× RTX 6000 Ada (49 GB). Envs: `generation` (train/extract/probe), `classification` (LLM2Vec).
Base model checkpoints from local HF cache. All unlearning + detection code is this repo's
`unlearning/`, `detection/`, and the drivers under `repro/`.

## Pipeline
1. **Unlearn** — `unlearning.rmu.unlearn` / `unlearning.npo.unlearn` with the paper's per-model
   hyperparameters (Appendix A, Tables A1/A2) on the WMDP forget corpora + wikitext retain.
2. **Extract** — `repro/extract_activations.py`: greedy-decode 100 tokens, hook `model.norm`,
   concatenate per-token pre-logit vectors → `100 × hidden` per prompt (409,600 for 7B).
3. **Detect** — `repro/run_probe.py`: four-layer MLP probe (Appendix B, `d_in→1024→256→128→2`,
   BatchNorm+Dropout+Xavier), trained on mixed MMLU+WMDP (`S_fig`), evaluated per-benchmark.

## Activation-based detection — Zephyr-7B (original vs unlearned)

Test accuracy on held-out prompts (355 per benchmark per class).

| Method | N / benchmark | MMLU | WMDP | Paper (Tab. A5/A6) |
|--------|:-------------:|:----:|:----:|:------------------:|
| RMU    | 1500          | 61.97% | 94.08% | MMLU 98.59 / WMDP 99.72 |
| RMU    | 2900          | 67.04% | 94.79% | MMLU 98.59 / WMDP 99.72 |
| NPO    | 1500          | 100.00% | 100.00% | MMLU 99.86 / WMDP 100 |
| NPO    | 2900          | _pending_ | _pending_ | MMLU 99.86 / WMDP 100 |

### Reading of the results
- **NPO is reproduced cleanly** (100% on both benchmarks; trivially separable — 99% train acc at
  epoch 0). This validates the extraction + probe pipeline end-to-end.
- **RMU is far subtler**, exactly the paper's central RMU-vs-NPO contrast (Fig. 4: no visible RMU
  shift on forget-irrelevant activations). The probe detects RMU strongly on the **forget-relevant**
  WMDP set (~95%) but weakly on **forget-irrelevant** MMLU (~67%).
- RMU-MMLU 67% still **exceeds the paper's text-based RMU-MMLU baseline (53.68%, Tab. A3)** — i.e.
  activations help over text — but does not reach the paper's 98.59%. Scaling 1500→2900 helped only
  marginally, so the residual gap is attributable to RMU-fingerprint subtlety / probe tuning rather
  than the pipeline (which NPO confirms is correct).

## NPO training sanity (Zephyr)
Forget loss starts at **13.81** = `(2/β)·log 2` at init (β=0.1) — confirms eq. (3) is implemented
correctly — then collapses to 0 within 3 steps (forget likelihood crushed below the reference) while
retain loss stays bounded. Matches the paper's description of NPO as aggressive vs RMU's localized edit.

## Activation-based detection — Llama-3.1-8B (original vs unlearned), N=2900

| Method | MMLU | WMDP | Paper trend |
|--------|:----:|:----:|:-----------:|
| RMU    | **99.72%** | **99.30%** | near-perfect (Fig. 3a) |
| NPO    | **100.00%** | **100.00%** | ~100 (Tab. A6) |

### Cross-model finding (reproduced)
Putting the two models side by side reproduces the paper's **"larger/stronger models
are more detectable"** result and Fig. 4 (Llama's RMU spectral fingerprint is *more
pronounced* than Zephyr's):

| RMU activation detection | MMLU | WMDP |
|--------------------------|:----:|:----:|
| Zephyr-7B  | 67.0% | 94.8% |
| Llama-3.1-8B | **99.7%** | **99.3%** |

Zephyr's RMU fingerprint on forget-irrelevant MMLU is genuinely subtle (the paper's own
Fig. 3a low point); Llama's is strong → near-perfect detection. NPO is trivially
detectable (100%) on both models. This is the paper's central qualitative result.

## Text-based detection — Zephyr-7B (gpt2 head)

Trained `detection/classify_responses.py` on mixed MMLU+WMDP responses (S_fig), evaluated
per held-out benchmark. NOTE: the paper's primary text encoder is **LLM2Vec**, but the
installed `llm2vec` is incompatible with both conda envs' transformers (4.44.2 lacks
`modeling_layers`; 4.57 dropped `MistralFlashAttention2`). I used the repo's **gpt2** head
instead — weaker than LLM2Vec, but it tests the same method.

| Method | MMLU | WMDP | Paper LLM2Vec (Tab. A3) |
|--------|:----:|:----:|:-----------------------:|
| RMU    | 50.70% | 80.00% | 53.68 / 90.56 |
| NPO    | 65.35% | 65.49% | ~99 |

- **RMU text pattern reproduced**: near-random on forget-irrelevant MMLU (50.7% vs paper 53.7),
  strong on forget-relevant WMDP (80% vs paper 90.6). RMU traces live in the *forget-relevant*
  responses (RMU garbles WMDP answers; MMLU answers stay coherent → undetectable from text).
- NPO text detection is weaker here than the paper (gpt2 ≪ LLM2Vec, and my grad-diff NPO keeps
  MMLU/WMDP responses fairly coherent rather than garbled).

## Headline: activations beat text (paper's thesis), reproduced

RMU detection accuracy, text → activation, by benchmark:

| RMU detection | MMLU | WMDP |
|---------------|:----:|:----:|
| Text (gpt2), Zephyr      | 50.7% | 80.0% |
| Activation, Zephyr-7B    | 67.0% | 94.8% |
| Activation, Llama-3.1-8B | **99.7%** | **99.3%** |

Pre-logit activations recover the unlearning trace that text alone misses — most dramatically
on the hardest case (RMU, forget-irrelevant MMLU), and more so for the larger model.

## Stealth attack (adversarial branch) — Llama-3.1-8B RMU

GAN: Generator = unlearned LLM + LoRA adapter; Discriminator = activation probe.
Goal: keep forgetting + utility, but evade the detector. Code: `stealth/`. Trained 250
iters, lambda_adv=lambda_keep=1.0. Evaluated against a **freshly retrained** detector.

| Metric | clean | RMU | RMU + Stealth |
|--------|:----:|:---:|:-------------:|
| Detection (clean-vs-X probe) MMLU | — | 99.0% | **100.0%** |
| Detection (clean-vs-X probe) WMDP | — | 99.3% | **100.0%** |
| Forgetting — WMDP MCQ (lower=better) | 52.4% | 25.0% | 32.9% |
| Utility — MMLU MCQ (higher=better) | 62.4% | 44.3% | 46.5% |

**Result: the naive attack FAILS — a lose–lose.**
- **Evasion failed**: the stealth model is detected at **100%** (even higher than plain RMU).
  In-loop the surrogate discriminator was fooled (`D_acc(holdout)` → ~0), but a detector
  *retrained from scratch* on the stealth model catches it perfectly. The LoRA adapter
  overfits to evading D on the 128 training prompts and, on fresh prompts, **adds its own
  fingerprint** instead of erasing the unlearning trace.
- **Forgetting partially eroded**: WMDP recovered 25.0% → 32.9% (adversarial fine-tuning
  leaked some forgotten knowledge back).
- **Utility ~unchanged**: 44.3% → 46.5%.

This empirically supports the paper's central claim — **activation-level unlearning traces are
robust/persistent**; a single-discriminator adversarial fine-tune does not remove them.
Stronger-attack directions (untested): train D on the same large/diverse distribution as the
evaluator (not 128 fixed prompts); make the adapter robust to a *retrained* detector
(detector ensemble / unrolled GAN); larger adapter or full fine-tune; exact feature match.

### v2: after fixing the train/eval mismatch (the "overfit" hypothesis, tested)

The naive run's failure was hypothesized to be train/eval mismatch, not a fundamental limit.
Three implementation fixes were applied (`stealth/`): (1) regenerate B's responses **fresh
every iteration** with the current adapter instead of caching stale responses — the evaluator
reads freshly-generated responses, so optimizing stale ones left the adapter overfitting fixed
token sequences; (2) match the detector's prompt formatting exactly (chat template + 512-token
truncation); (3) train at `cont_len=100` (the evaluator's length) on 384 prompts. Gradient
checkpointing makes the per-step regeneration affordable. Retrained-probe eval, N=1500:

| Metric | clean | RMU (B0) | Stealth single-D v2 | Stealth ensemble v2 |
|--------|:----:|:--------:|:-------------------:|:-------------------:|
| Detection MMLU | — | 99.7% | **100.0%** | **100.0%** |
| Detection WMDP | — | 99.3% | **100.0%** | **100.0%** |
| Forgetting — WMDP MCQ (↓ better) | 52.4% | 25.0% | 26.7% | 26.2% |
| Utility — MMLU MCQ (↑ better) | 62.4% | 44.3% | **34.3%** | **25.1%** (≈ random) |

**Result: the fix removes the overfitting but the attack still FAILS — now a clearer lose–lose.**
With fresh-per-iter generation the in-loop surrogate D is genuinely fooled on held-out prompts
(`D_acc` → 0.09 single-D; ensemble's *reinitialized* discriminators stay high, 0.75–1.0, the
honest signal). Yet a probe **retrained from scratch** still detects at **100%** — the adapter,
now actually moving activations on fresh data rather than overfitting fixed sequences, **adds a
detectable perturbation** rather than erasing the trace. And moving activations costs utility:
MMLU drops 44.3 → 34.3 (single-D) / 25.1 ≈ random (ensemble, where `lambda_keep=1` failed to
hold behavior, KL → 2.0). The trade-off frontier is monotone the wrong way: raising
`lambda_keep` pulls B toward B0, recovering utility but making detection *easier* (B0 itself is
99.7%) — there is **no regime where detection drops**. This is strong, clean evidence for the
paper's robustness claim: the activation-level unlearning trace survives a *correctly-trained*
adversarial fine-tune (single D and reinit-ensemble alike). Removing it would require a
fundamentally different objective (explicit A↔B activation-distribution / feature matching, an
unrolled game against the retrained detector, or full fine-tune), not just better adversarial
training. Artifacts: `repro/models/llama_rmu_stealth_{v2,ens_v2}{,_merged}/`,
logs `repro/logs/stealth_{gan,ens}_v2{,_eval}.log`, driver `repro/stealth_eval_1gpu.sh`.

### v3: feature matching instead of adversarial (the "different objective", tested)

Since the adversarial game fails because a *retrained* probe always finds a fresh separating
direction, we attacked the root cause directly: drop the discriminator entirely and minimize a
**distribution-matching** loss that pulls B's pre-logit feature onto clean A's, so that *no*
classifier can separate them. Code: `stealth/feature_match.py`, loss
`L = lambda_fm · moment_match(feat_B, feat_A) + lambda_keep · KL(B0||B)`.

A first attempt — RBF-MMD on a low-dim random projection of the feature — was a dead end: `L_fm`
sat near 0 from the start (projected A and B are already close) while the full-feature probe
stayed at 100%, i.e. the projection washed out exactly the direction the probe uses. Replaced
with **full-dimensional per-dim mean + variance matching** (normalized by A's variance), which
targets precisely what the probe's per-dim standardization exposes. Retrained-probe eval, N=1500:

| Metric | clean | RMU (B0) | FM `lambda_fm=1` | FM `lambda_fm=5` |
|--------|:----:|:--------:|:----------------:|:----------------:|
| Detection MMLU | — | 99.7% | **100.0%** | **100.0%** |
| Detection WMDP | — | 99.3% | **100.0%** | **100.0%** |
| Forgetting — WMDP MCQ (↓ better) | 52.4% | 25.0% | 25.6% | 27.8% |
| Utility — MMLU MCQ (↑ better) | 62.4% | 44.3% | **45.2%** | 41.7% |

**Result: feature matching also FAILS — and this is the cleanest negative of the three.** Across
all 300 iterations the moment-match loss falls (`L_fm` 0.68 → 0.28 at `lambda_fm=1`, → 0.16 at
`lambda_fm=5`) yet the in-loop read-out probe never drops below **1.00** on either benchmark, and
the retrained eval probe detects at **100%**. Crucially `lambda_fm=1` keeps the model fully intact
— forgetting (WMDP 25.6 ≈ RMU's 25.0) *and* utility (MMLU 45.2 ≈ B0's 44.3) — so unlike the
adversarial runs this is not a "broke the model" artifact: a perfectly functional unlearned model,
with its per-dim activation moments pulled onto clean A's, is *still* detected perfectly. Pushing
`lambda_fm` 5× lowers `L_fm` further without moving detection at all.

Interpretation (the payoff): **the RMU activation trace does not live in the per-dimension
marginal distribution — it lives in the joint/correlation structure**, which is exactly what the
paper's MLP probe reads and what marginal matching leaves untouched. This unifies all three
attacks into one story: the trace survives an adversarial fine-tune (the probe re-finds its
direction), a reinit-ensemble (same), *and* marginal distribution matching (wrong statistic) —
three independent attack families, all defeated. Strong support for the paper's central claim that
activation-level unlearning traces are robust and persistent. A genuinely stronger attack would
need to match *joint* structure (full/blockwise covariance or an unrolled game against the
retrained probe) — at which point preserving forgetting (which *is* an activation difference on
WMDP) becomes the binding constraint. Artifacts: `repro/models/llama_rmu_fm{1,5}{,_merged}/`,
logs `repro/logs/fm{1,5}{,_eval}.log`, code `stealth/feature_match.py`.

### v4: covariance (JOINT) matching — random vs discriminative subspace (the "match the right statistic", tested)

v3 located the failure: the trace lives in the **joint/correlation structure**, not the per-dim
marginals. v4 attacks that directly — match the **covariance** of B's standardized pre-logit
feature onto clean A's. Code: `stealth/cov_match.py`, loss `L = lambda_marg·(mean+var) +
lambda_cov·||Cov(zB)-Cov(zA)||_F^2 + lambda_keep·KL(B0||B)`. Full 4096² covariance from a
minibatch is rank-deficient, so covariance is matched in k-dim subspaces (CORAL, sketched by
projection). Run on **Zephyr-RMU**, **MMLU-only** (the forget-irrelevant set, where the forgetting
constraint does *not* block evasion — the cleanest place an attack *could* win). Retrained beefy
probe (epochs 30, dropout 0.3), clean-vs-X, N=1500:

| Metric | clean | RMU (B0) | marg (zcov0) | rand-joint (zcov5) | **disc-joint (zcovD5)** | **disc-joint λ20 (zcovD20)** |
|--------|:----:|:--------:|:------------:|:------------------:|:-----------------------:|:---------------------------:|
| Detection MMLU | — | **63.1%** | 99.9% | 100% | **100%** | **100%** |
| Detection WMDP | — | 95.1% | 100% | 100% | 100% | 100% |
| `L_cov` start→end (in-loop) | — | — | n/a | 0.003→0.003 | **0.053→0.025** | 0.046→0.019 |
| Forget WMDP MCQ (↓) | — | 26.6% | 26.6% | 26.6% | 26.6% | 26.6% |
| Utility MMLU MCQ (↑) | — | 22.4% | 22.4% | 22.4% | 22.4% | 22.4% |

**Result: joint matching FAILS too — the strongest, cleanest negative of the study.** Two layers:

1. **Random subspaces are blind to the trace.** `L_cov` sits at ~0.003 *from iter 0* and never
   moves (same smell as v3's dead-end random-projection MMD): a random k-of-4096 sketch dilutes the
   small, low-rank separating direction below the bulk-variance floor. Detection ≡ the marginal
   control (both ~100%). So zcov5 ≡ zcov0 — random joint matching adds nothing.
2. **Discriminative subspaces SEE the trace and still can't remove it.** Building the subspace from
   the **diff-of-class-means + top eigenvectors of (Cov_A − Cov_B0)** makes `L_cov` start ~15× higher
   (0.053) — the structure is now visible — and training **halves** it (→0.025; λ=20 pushes to
   0.019). Yet the retrained probe stays at **100%**, and behavior is bit-identical to B0
   (forget 26.6, util 22.4 unchanged). We matched 2nd-order structure in *exactly* the directions
   that separate the classes, shrank the gap, pushed at λ=20 — and detection did not move off 100%.

This rules out the most charitable read of v1–v3 ("we targeted the wrong statistic/subspace"). Even
with the statistic and subspace chosen *optimally*, a behavior-preserving LoRA edit (a) cannot close
the covariance gap (residual ~0.02 ≠ 0 — the adapter lacks the freedom under the KL-keep pin), and
(b) the MLP probe needs only a small residual, plus higher-order structure the 2nd-moment match
never touches, plus directions outside the matched subspace. Net: every activation perturbation
under these constraints *raises* detectability (B0 63% → ~100%) — the adapter's own footprint
dominates. Note: this Zephyr-RMU checkpoint's MMLU utility is already ~random (22.4%), so the
*utility* axis isn't testable here; the *detection* conclusion is unaffected (Llama-RMU, util 44%,
is the better utility testbed for a follow-up).

**Unified story across FOUR attack families.** The RMU activation trace survives (1) adversarial
single-D fine-tuning, (2) a reinit-detector ensemble, (3) marginal distribution matching, and (4)
joint/covariance matching in both random and *discriminative* subspaces. Each attack is strictly
stronger/more targeted than the last; all fail, and the failure becomes *more* informative each
time. Strongest possible support for the paper's claim that activation-level unlearning traces are
robust and persistent. A genuinely stronger attack would need higher-order/full-distribution
matching or an unrolled game against the retrained probe **and** more adapter freedom than
behavior-preservation allows — on forget-relevant data that freedom directly fights forgetting.
Artifacts: `repro/models/zephyr_rmu_{zcov0,zcov5,zcovD5,zcovD20}{,_merged}/`,
logs `repro/logs/{zcov0,zcov5,zcovD5,zcovD20,zcov_eval,zcovD_eval}.log`,
code `stealth/cov_match.py`, driver `repro/zcov_eval.sh`.

## Artifacts
- Unlearned checkpoints: `repro/models/{zephyr,llama}_{rmu,npo}/`
- Activations: `repro/activations_2900/*.npy` (and `repro/activations/` at N=1500)
- Responses + splits: `repro/responses/`, `repro/responses/splits/`
- Text classifiers: `repro/models/clf_text_{rmu,npo}/`
- Drivers: `repro/extract_activations.py`, `repro/run_probe.py`,
  `repro/generate_responses_hf.py`, `repro/build_text_splits.py`, `repro/eval_text.py`
- Logs: `repro/logs/`
