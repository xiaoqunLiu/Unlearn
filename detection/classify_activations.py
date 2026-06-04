#!/usr/bin/env python3

import argparse
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments

class ActivationTensorDataset(TorchDataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__()
        # convert once to torch
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        return {"vectors": self.X[idx], "labels": self.y[idx]}

class MLPHFClassifier(nn.Module):
    """Four-layer MLP probe on pre-logit activations (paper Appendix B).

    Architecture: d_in -> 1024 -> 256 -> 128 -> 2. Each hidden layer is followed
    by BatchNorm and Dropout, and every linear layer uses Xavier initialization.
    """
    def __init__(self, dim_in, num_labels,
                 hidden_dims=(1024, 256, 128),
                 dropout=0.1):
        super().__init__()
        layers = []
        prev = dim_in
        for h in hidden_dims:
            linear = nn.Linear(prev, h)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        head = nn.Linear(prev, num_labels)
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, vectors, labels=None):
        logits = self.net(vectors)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}

def find_model_pairs(base_dir, base_model="zephyr"):
    """Find activation files for original model and RMU model pair"""

    # Look for MMLU and WMDP files for both original and RMU models
    datasets = ["mmlu", "wmdp"]
    model_files = {}

    for dataset in datasets:
        original_file = os.path.join(base_dir, f"{dataset}_{base_model}_model.norm_eval_activations.npy")
        rmu_file = os.path.join(base_dir, f"{dataset}_{base_model}-rmu_model.norm_eval_activations.npy")

        if os.path.exists(original_file) and os.path.exists(rmu_file):
            model_files[dataset] = {
                "original": original_file,
                "rmu": rmu_file
            }
            print(f"Found {dataset} pair:")
            print(f"  Original: {os.path.basename(original_file)}")
            print(f"  RMU: {os.path.basename(rmu_file)}")
        else:
            print(f"Warning: Missing {dataset} files for {base_model}")
            if not os.path.exists(original_file):
                print(f"  Missing: {os.path.basename(original_file)}")
            if not os.path.exists(rmu_file):
                print(f"  Missing: {os.path.basename(rmu_file)}")

    return model_files

def balance_datasets(data_dict, samples_per_dataset=500):
    """Balance MMLU and WMDP data for each model type"""
    balanced_data = {"original": [], "rmu": []}

    for dataset in data_dict:
        for model_type in ["original", "rmu"]:
            data = data_dict[dataset][model_type]

            # Sample with replacement if we don't have enough data
            if len(data) >= samples_per_dataset:
                indices = np.random.choice(len(data), samples_per_dataset, replace=False)
            else:
                indices = np.random.choice(len(data), samples_per_dataset, replace=True)
                print(f"Warning: {dataset} {model_type} has only {len(data)} samples, using replacement")

            balanced_data[model_type].append(data[indices])

    # Concatenate balanced data
    for model_type in balanced_data:
        balanced_data[model_type] = np.vstack(balanced_data[model_type])

    return balanced_data

def load_and_split(model_files, args):
    """Load activation files for original vs RMU classification"""
    print("Loading activation files for RMU vs Original classification:")

    if not model_files:
        raise ValueError("No model file pairs found")

    # Load all data first
    data_dict = {}
    for dataset, files in model_files.items():
        data_dict[dataset] = {}
        for model_type, file_path in files.items():
            arr = np.load(file_path)
            print(f"  {dataset} {model_type}: {arr.shape}")
            data_dict[dataset][model_type] = arr

    # Balance datasets (equal samples from MMLU and WMDP)
    print(f"\nBalancing datasets ({args.samples_per_dataset} samples per dataset)...")
    balanced_data = balance_datasets(data_dict, args.samples_per_dataset)

    # Create labels: 0 = Original, 1 = RMU
    X_original = balanced_data["original"]
    X_rmu = balanced_data["rmu"]

    y_original = np.zeros(len(X_original))
    y_rmu = np.ones(len(X_rmu))

    # Combine data
    X = np.vstack([X_original, X_rmu])
    y = np.concatenate([y_original, y_rmu])

    print(f"\nFinal dataset:")
    print(f"  Combined shape: {X.shape}")
    print(f"  Original samples: {len(X_original)}")
    print(f"  RMU samples: {len(X_rmu)}")
    print(f"  Label distribution: {np.bincount(y.astype(int))}")

    # Stratified split
    train_size = args.num_train_per_class * 2  # 2 classes
    test_size = args.num_test_per_class * 2

    print(f"\nDataset split:")
    print(f"  Train size per class: {args.num_train_per_class}")
    print(f"  Test size per class: {args.num_test_per_class}")
    print(f"  Total train size: {train_size}")
    print(f"  Total test size: {test_size}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_size,
        test_size=test_size,
        stratify=y,
        random_state=42
    )

    class_names = ["Original", "RMU"]

    return (
        ActivationTensorDataset(X_train, y_train),
        ActivationTensorDataset(X_test, y_test),
        class_names
    )

def main():
    parser = argparse.ArgumentParser(description="Train binary classifier: Original vs RMU models")

    # Data arguments
    parser.add_argument("--activation_dir", type=str,
                        default="/egr/research-optml/shared/soumyadeep_reveng/classification-activation",
                        help="Directory containing activation files")
    parser.add_argument("--base_model", type=str, default="zephyr",
                        help="Base model name (e.g., zephyr, llama, qwen)")
    parser.add_argument("--samples_per_dataset", type=int, default=2900,
                        help="Samples per dataset (MMLU/WMDP) for balancing (paper: 2900 each)")
    parser.add_argument("--num_train_per_class", type=int, default=2545,
                        help="Number of training samples per class")
    parser.add_argument("--num_test_per_class", type=int, default=355,
                        help="Held-out test samples per class (paper: 355 per benchmark)")

    # Model architecture (paper Appendix B: d_in -> 1024 -> 256 -> 128 -> 2)
    parser.add_argument("--hidden_dims", type=str, default="1024,256,128",
                        help="Comma-separated hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Training arguments (paper Appendix B)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=8e-5,
                        help="Initial learning rate (cosine schedule)")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="AdamW weight decay")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./models_activation",
                        help="Output directory for trained model")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find model pairs
    print(f"Searching for {args.base_model} model pairs in: {args.activation_dir}")
    model_files = find_model_pairs(args.activation_dir, args.base_model)

    if not model_files:
        raise ValueError(f"No {args.base_model} model pairs found in {args.activation_dir}")

    # Load data and create datasets
    ds_train, ds_test, class_names = load_and_split(model_files, args)

    # Model parameters
    D = ds_train.X.shape[1]  # Feature dimension
    C = 2  # Binary classification: Original vs RMU

    print(f"\nModel configuration:")
    print(f"  Input dimension: {D}")
    print(f"  Number of classes: {C}")
    print(f"  Class names: {class_names}")

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    hidden_dims = tuple(int(x) for x in str(args.hidden_dims).split(","))
    model = MLPHFClassifier(
        dim_in=D,
        num_labels=C,
        hidden_dims=hidden_dims,
        dropout=args.dropout
    ).to(device)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training configuration
    model_name = f"{args.base_model}_original_vs_rmu_classifier"
    output_path = os.path.join(args.output_dir, model_name)

    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        bf16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(output_path, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        compute_metrics=compute_metrics,
    )

    print(f"\nStarting training...")
    print(f"Output directory: {output_path}")

    # Train the model
    trainer.train()

    # Final evaluation
    print(f"\nFinal evaluation:")
    eval_results = trainer.evaluate()

    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    # Save class names and model info
    info_path = os.path.join(output_path, "model_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Model: {args.base_model} Original vs RMU Classifier\n")
        f.write(f"Classes:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")
        f.write(f"\nDatasets used: {list(model_files.keys())}\n")
        f.write(f"Samples per dataset: {args.samples_per_dataset}\n")
        f.write(f"Train samples per class: {args.num_train_per_class}\n")
        f.write(f"Test samples per class: {args.num_test_per_class}\n")

    print(f"\nModel info saved to: {info_path}")
    print(f"Model saved to: {output_path}")

if __name__ == "__main__":
    main()
