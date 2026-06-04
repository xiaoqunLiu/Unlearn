import argparse
import json
import numpy as np
import random
import torch
import os
# from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer

def load_model_tok(args):

    # The `*-rmu` / `*-npo` paths below are the unlearned checkpoints produced by
    # `python -m unlearning.rmu.unlearn ... --output_dir <path>` and
    # `python -m unlearning.npo.unlearn ... --output_dir <path>` (see docs/Unlearn.md).
    model_args_to_name = {
        "zephyr-rmu" : "zephyr_rmu/",
        "zephyr": "HuggingFaceH4/zephyr-7b-beta",
        "zephyr-npo" : "zephyr_npo/",
        "yi":  "01-ai/Yi-34B-Chat",
        "yi-rmu": "yi_rmu/",
        "yi-npo": "yi_npo/",
        "llama": "meta-llama/Meta-Llama-3.1-8B",
        "llama-rmu": "llama-rmu/",
        "llama-npo": "llama-npo/",
        "qwen": "Qwen/Qwen2.5-14B",
        "qwen-rmu": "Qwen2.5-14B-coeff460-alpha350/",
        "qwen-npo" : "qwen-npo/",
    }

    model_name = model_args_to_name[args.model]
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    ## We use attn_implementation="eager" when we want to return attention weights and not return None
    if args.model == "llama":
        tokenizer = AutoTokenizer.from_pretrained("llama-rmu/")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.padding_side = "left"
    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side="right"

    hooked_layers = {
        "zephyr-rmu" : [[5,6,7], 31],
        "zephyr": [[5,6,7], 31],
        "yi":  [[13,14,15], 59],
        "yi-rmu": [[13,14,15], 59],
        "yi-npo": [[13,14,15], 59],
        "llama-rmu": [[5,6,7], 31],
        "llama": [[5,6,7], 31],
        "llama-npo": [[5,6,7], 31],
        "qwen": [[8,9,10], 31],
        "qwen-rmu": [[8,9,10], 31],
        "qwen-npo": [[8,9,10], 31],
        "zephyr-npo" : [[5,6,7], 31],
    }
    # import pdb;pdb.set_trace()

    return model, tokenizer, hooked_layers[args.model]


def generate_responses_base_llm(args):
    model, tokenizer, hooked_layers = load_model_tok(args)

    if args.train == 1:
        with open(f"../../shared/yiwei/train_file/{args.dataset_name}/Zephyr-7b.json") as f:
            questions = json.load(f)
    else:
        # with open(f"../../shared/yiwei/eval_file/mmlu-wmdp-eval/MMLU-wmdp-eval/Zephyr-7b.json") as f:
        with open(f"../../shared/yiwei/eval_file/{args.dataset_name}-eval/Zephyr-7b.json") as f:
            questions = json.load(f)
    # import pdb;pdb.set_trace()

    # if args.model == "qwen" or args.model == 'qwen14-rmu':
    #     layers_to_hook = [f'model.layers.{layerno}.mlp.gate_proj' for layerno in hooked_layers[0]]
    # else:
    #     layers_to_hook = [f'model.layers.{layerno}.mlp.down_proj' for layerno in hooked_layers[0]]
    layers_to_hook = []
    layers_to_hook.append(f'model.norm')
    activations_all = []

    for idx, question in enumerate(questions):
        print(idx)

        prompt_message =questions[idx][0]
        if args.model == 'qwen' or args.model == 'qwen-rmu' or args.model == 'qwen-npo' or args.model == 'yi-npo':
            prompt_message = [prompt_message]

        # import pdb;pdb.set_trace()

        activations = {name: [] for name in layers_to_hook}
        def activation_hook(name):
            def hook_fn(module, input, output):
                # import pdb;pdb.set_trace()
                output_layer = output[0,-1,:].detach().tolist()
                activations[name].extend(output_layer)
            return hook_fn

        hook_handles = []
        for name in layers_to_hook:
            layer = dict(model.named_modules())[name]
            hook_handles.append(layer.register_forward_hook(activation_hook(name)))

        if args.model == "llama-rmu" or args.model == "llama" or args.model == "llama-npo":
            prompt_message = prompt_message['content']
            gen_inputs = tokenizer(prompt_message, return_tensors="pt", padding=True).to(f'cuda:0')
        else:
            # import pdb;pdb.set_trace()
            gen_inputs_notok = tokenizer.apply_chat_template(prompt_message,  add_generation_prompt=True,tokenize=False)
            gen_inputs = tokenizer(gen_inputs_notok, return_tensors="pt", padding=True).to(f'cuda:0')

        input_length = gen_inputs["input_ids"].shape[1]
        output = model.generate(gen_inputs['input_ids'],attention_mask=gen_inputs['attention_mask'], do_sample=False, max_new_tokens=100, output_hidden_states=True, return_dict_in_generate=True, eos_token_id=None) #, temperature=0.0) ## top_p=0.95

        # import pdb;pdb.set_trace()
        for hook in hook_handles:
            hook.remove()

        # import pdb;pdb.set_trace()

        activations_all.append(activations)

    # import pdb;pdb.set_trace()
    for layer in layers_to_hook:
        vectors = [data[layer] for data in activations_all]
        data_array = np.array(vectors,  dtype=np.float32)
        if args.train == 1:
            fname = f"{args.dataset_name}_{args.model}_{layer}_greedy_train_activations.npy"
        else:
            # fname = f"{args.model}_{layer}_mmlu-wmdp-eval_activations.npy"
            fname = f"{args.dataset_name}_{args.model}_{layer}_eval_activations.npy"
        np.save(fname, data_array)


    # with open(f"{args.dataset_name}_{args.model}_new.json", "w") as f:
    #     json.dump(activations_all, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='mmlu_new')
    parser.add_argument("--model", type=str, default='zephyr')
    parser.add_argument("--train", type=int, default=1)
    args = parser.parse_args()
    generate_responses_base_llm(args)
