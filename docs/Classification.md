## Classifier Training

To train classifiers for distinguishing between original and unlearned model responses, you can use the provided classification script. Below is an example command for binary classification between original and unlearned models:

```bash
python detection/classify_responses.py \
    --response_paths ./path/to/model1-train.json \
                     ./path/to/model1-unlearn-train.json \
    --classifier llm2vec \
    --num_train_samples 5800 \
    --num_test_samples 355 \
    --output_dir ./classification_models/
```

For N-way classification with multiple models, you can include N response paths:

```bash
python detection/classify_responses.py \
    --response_paths ./path/to/model1-train.json \
                     ./path/to/model1-unlearn-train.json \
                     ./path/to/model2-train.json \
                     ./path/to/model2-unlearn-train.json \
                     ./path/to/model3-train.json \
                     ./path/to/model3-unlearn-train.json \
                     ./path/to/model4-train.json \
                     ./path/to/model4-unlearn-train.json \
    --classifier llm2vec \
    --num_train_samples 5800 \
    --num_test_samples 355 \
    --output_dir ./multi_classification_models/
```

**Parameters**:
- `--response_paths`: Paths to response files from different models (space-separated)
- `--classifier`: Type of classifier to use (`llm2vec`, `gpt2`, `t5`, or `bert`)
- `--num_train_samples`: Number of training samples per class
- `--num_test_samples`: Number of test samples per class
- `--output_dir`: Directory to save trained classifier models

## Classifier Evaluation

To evaluate trained classifiers on new data, you can use the classification script with the `--eval_only` flag. Below is an example command for binary classification evaluation:

```bash
python detection/classify_responses.py \
    --response_paths ./path/to/model1-eval.json \
                     ./path/to/model1-unlearn-eval.json \
    --classifier llm2vec \
    --eval_only \
    --num_train_samples 5 \
    --num_test_samples 355 \
    --resume_from_checkpoint ./classification_models \
    --output_dir ./classification_models_eval/
```

For multi-class classification evaluation with multiple models:

```bash
python detection/classify_responses.py \
    --response_paths ./path/to/model1-eval.json \
                     ./path/to/model1-unlearn-eval.json \
                     ./path/to/model2-eval.json \
                     ./path/to/model2-unlearn-eval.json \
                     ./path/to/model3-eval.json \
                     ./path/to/model3-unlearn-eval.json \
                     ./path/to/model4-eval.json \
                     ./path/to/model4-unlearn-eval.json \
    --classifier llm2vec \
    --eval_only \
    --num_train_samples 5 \
    --num_test_samples 355 \
    --resume_from_checkpoint ./classification_models \
    --output_dir ./multi_classification_models_eval/
```

**Note**: When using `--eval_only`, the script loads pre-trained classifier models from the checkpoint directory and evaluates them on the specified response data without further training.

**Additional Parameters**:
- `--eval_only`: Flag to enable evaluation mode (no training)
- `--resume_from_checkpoint`: Path to directory containing pre-trained classifier models

## Activation-Based Classifier

Besides classifying the **text responses** above, the paper also detects unlearning traces directly from a model's **pre-logit activations**. After extracting activation vectors with `generation/generate_activations.py` (see [Response.md](./Response.md)), train an MLP probe to distinguish original from unlearned (RMU) activations using `detection/classify_activations.py`:

```bash
python detection/classify_activations.py \
    --activation_dir ./classification-activation \
    --base_model zephyr \
    --samples_per_dataset 2900 \
    --num_train_per_class 2545 \
    --num_test_per_class 355 \
    --output_dir ./models_activation/
```

The script looks inside `--activation_dir` for the four `.npy` files that make up an original-vs-RMU pair across the MMLU and WMDP datasets:

```
{dataset}_{base_model}_model.norm_eval_activations.npy        # original model
{dataset}_{base_model}-rmu_model.norm_eval_activations.npy    # RMU model
```

for `dataset ∈ {mmlu, wmdp}`. It balances MMLU/WMDP samples (the mixed training set `S_fig`), builds a stratified train/test split (label `0 = Original`, `1 = RMU`), trains the MLP probe, and saves the model plus a `model_info.txt` summary under `--output_dir`.

**Probe architecture and training (paper Appendix B)**: the probe is a four-layer MLP `d_in → 1024 → 256 → 128 → 2`, with BatchNorm + Dropout after each hidden layer and Xavier initialization. `d_in` is the flattened activation dimension (100 generated tokens × hidden size, e.g. `409,600` for Zephyr-7B). Training uses AdamW (`weight_decay 1e-3`), a cosine LR schedule with initial `lr 8e-5` and `warmup_ratio 0.1`, `3` epochs, batch size `8`, gradient clipping `0.3`, and BF16 — all matching the defaults below.

**Parameters**:
- `--activation_dir`: Directory containing the `.npy` activation files produced by `generation/generate_activations.py`
- `--base_model`: Base model name used to locate the file pair (e.g., `zephyr`, `llama`, `qwen`)
- `--samples_per_dataset`: Samples drawn from each of MMLU/WMDP for balancing (paper: `2900` each → `5800`-sample mixed set per class)
- `--num_train_per_class` / `--num_test_per_class`: Per-class train/test split sizes (paper holds out `355` per benchmark)
- `--hidden_dims`: Comma-separated hidden layer sizes (default `1024,256,128`)
- `--dropout`: Dropout rate
- `--batch_size` / `--epochs` / `--learning_rate` / `--weight_decay`: Training hyperparameters
- `--output_dir`: Directory to save the trained probe and `model_info.txt`

**Data regimes (paper Appendix B)**: the main results train on the mixed set `S_fig` (50% WMDP + 50% MMLU). The ablations `S_f` (WMDP only) and `S_g` (MMLU only) can be reproduced by extracting activations for a single dataset and pointing `--activation_dir` at it. The same text-based classifier (`detection/classify_responses.py`) is trained on `S_fig` with `--num_train_samples 5800 --num_test_samples 355`.
