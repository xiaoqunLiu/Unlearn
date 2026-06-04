## Response Generation

Customize the `model_name` in both `generation/generate_response.py` and `generation/generate_wmdp_response.py` to point to your unlearned model's location. Ensure you have downloaded the WMDP multiple-choice question JSON files as described in [Data.md](./Data.md).

### Forget response generation

Generate responses for **forget-relevant** WMDP questions:

```
python generation/generate_wmdp_response.py \
    --model Yi-34B-Chat --temperature 0 \
    --dataset_path ./data/wmdp-mcqs/cyber_questions.json \
    --output_path ./responses/wmdp-cyber/Yi-34B-Chat.json \
    --num_gpus 4\
```
- `--model`: Name of the model to evaluate  
- `--temperature`: Sampling temperature (`0` for deterministic outputs)  
- `--dataset_path`: Path to WMDP JSON file (e.g., `cyber_questions.json` or `bio_questions.json`)  
- `--output_path`: File path to save generated responses  
- `--num_gpus`: Number of GPUs to use  

You can swap `--dataset_path` between `cyber_questions.json` and `bio_questions.json`, or modify `--model` as needed.  

### Combining WMDP Datasets

After generating responses for both bio and cyber datasets, you can combine them into a single WMDP dataset using the provided combination script:

```
python data_process/wmdp_combine.py
```

**Note**: Before running the script, customize the folder paths in `wmdp_combine.py` according to your directory structure:
- `bio_dir`: Path to your bio response files
- `cyber_dir`: Path to your cyber response files  
- `output_dir`: Path where you want the combined files saved

### Forget-irrelevant response generation

Generate responses for **forget-irrelevant** benchmarks (e.g., MMLU or UltraChat):

```
python generation/generate_response.py \
    --model Yi-34B-Chat --temperature 0 \
    --dataset MMLU --num_samples 11_000 \
    --output_path ./responses/MMLU/Yi-34B-Chat.json \
    --num_gpus 4\
```
- `--dataset`: Dataset name (`MMLU` or `UltraChat`)  
- `--num_samples`: Number of samples to generate  

Feel free to adjust `--model`, `--dataset`, `--temperature`, and other flags to match your experimental setup.  

### Data Split

After generating responses, you can split your datasets into training and evaluation sets using the provided splitting script:

```bash
python data_process/split.py
```

**Note**: Before running the script, customize the configuration variables in `split.py` according to your dataset:
- `source_dir`: Path to your response files (UltraChat, WMDP, MMLU, or other datasets)
- `train_dir`: Output directory for training split
- `eval_dir`: Output directory for evaluation split  
- `TOTAL_TRAIN`: Number of training samples
- `TOTAL_EVAL`: Number of evaluation samples

### Activation Extraction (for activation-based detection)

In addition to detecting unlearning traces from text outputs, the paper also studies traces in the model's **pre-logit activations**. Following the paper (Sec. 4 / Appendix B), `generation/generate_activations.py` greedy-decodes a **100-token** response and records the hidden state of the final `model.norm` layer (the pre-logit activation, after the last RMSNorm) for **each newly generated token**, hooking the layer once per decode step. Concatenating the per-token vectors across the 100-token response yields the activation representation for that prompt — dimension `100 × hidden_size` (e.g. `409,600` for Zephyr-7B). The paper extracts these for `3,000` sampled prompts per model/dataset:

```
python generation/generate_activations.py \
    --dataset_name mmlu_new \
    --model zephyr \
    --train 1
```

- `--dataset_name`: Name of the question set to read (e.g., `mmlu_new`, `wmdp`); the script loads the corresponding `Zephyr-7b.json` question file.
- `--model`: Model key to extract activations from. Supported keys include the original models (`zephyr`, `yi`, `llama`, `qwen`) and their unlearned counterparts (`*-rmu`, `*-npo`). Customize the model paths in the `model_args_to_name` dictionary inside the script to point to your local checkpoints.
- `--train`: `1` to read from the train question files and write `*_greedy_train_activations.npy`, `0` to read from the eval question files and write `*_eval_activations.npy`.

The output files are named `{dataset_name}_{model}_model.norm_{train|eval}_activations.npy` and are consumed directly by the activation-based classifier (see [Classification.md](./Classification.md)). Extract activations for both the original model and its unlearned counterpart (e.g., `zephyr` and `zephyr-rmu`) on each dataset (e.g., `mmlu` and `wmdp`) to form a complete original-vs-unlearned pair.

### Mixed Data Generation

If you want to combine forget-irrelevant and forget-relevant datasets to create mixed training and evaluation datasets, you can use the provided mixing scripts:

```
python data_process/mixed_train.py
python data_process/mixed_eval.py
```

**Note**: Before running the scripts, customize the configuration variables according to your datasets:
- `src1`: Path to your first dataset directory (e.g., MMLU-train/MMLU-eval)
- `src2`: Path to your second dataset directory (e.g., wmdp-train/wmdp-eval)  
- `out_dir`: Output directory for mixed datasets
- `n_per`: Number of samples to take from each dataset (2900 for train, 180 for eval by default)