import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

sys.path.append(os.path.join(os.getcwd(), "code", "classification"))
from gru_model import GRUDepressionClassifier

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def parse_path(path):
    base = os.path.basename(path)
    parts = re.findall(r'\d+', base)
    if len(parts) >= 3:
        speaker_id = parts[0]
        utt_idx = int(parts[1])
        seg_idx = int(parts[2])
        return speaker_id, (utt_idx, seg_idx)
    elif len(parts) == 2:
        return parts[0], (int(parts[1]), 0)
    return None, None

def get_metadata_path(feature_dir):
    if "edaic" in feature_dir:
        return "utterance_table_edaic_segmented_split.csv"
    elif "modma" in feature_dir:
        return "utterance_table_modma_segmented_split.csv"
    elif "mix" in feature_dir:
        return "utterance_table_mix_segmented_split.csv"
    raise ValueError(f"Could not determine metadata path from directory name: {feature_dir}")

def build_sequences(feature_dir, split, pooling="mean", max_len=150):
    metadata_csv = get_metadata_path(feature_dir)
    df = pd.read_csv(metadata_csv)
    df_split = df[df["split"] == split].reset_index(drop=True)
    
    if len(df_split) == 0:
        return None, None, None
        
    X_path = os.path.join(feature_dir, f"X_{split}_{pooling}.npy")
    y_path = os.path.join(feature_dir, f"y_{split}.npy")
    
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Feature file not found: {X_path}")
        
    X = np.load(X_path)
    
    # Group segments by speaker
    speaker_data = {}
    for i, row in df_split.iterrows():
        path = row["file_path"]
        spk_id, sort_key = parse_path(path)
        if spk_id is None:
            continue
        label = row["label"]
        feat = X[i]
        
        if spk_id not in speaker_data:
            speaker_data[spk_id] = {
                "feats": [],
                "sort_keys": [],
                "label": label
            }
        speaker_data[spk_id]["feats"].append(feat)
        speaker_data[spk_id]["sort_keys"].append(sort_key)
        
    # Build sorted sequences
    sequences = []
    labels = []
    speaker_ids = []
    for spk_id, data in speaker_data.items():
        # Sort indices based on chronological sort_keys
        sorted_indices = sorted(range(len(data["sort_keys"])), key=lambda k: data["sort_keys"][k])
        sorted_feats = [data["feats"][idx] for idx in sorted_indices]
        
        sequences.append(np.array(sorted_feats))
        labels.append(data["label"])
        speaker_ids.append(spk_id)
        
    # Pad sequences
    num_sequences = len(sequences)
    feature_dim = X.shape[1]
    
    padded_sequences = np.zeros((num_sequences, max_len, feature_dim), dtype=np.float32)
    masks = np.zeros((num_sequences, max_len), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_len)
        padded_sequences[i, :seq_len] = seq[:seq_len]
        masks[i, :seq_len] = 1.0
        
    return padded_sequences, np.array(labels), masks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Directory containing train/val features.")
    parser.add_argument("--test_data", type=str, required=True, help="Directory containing test features.")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "concat"])
    parser.add_argument("--exp_name", type=str, default="gru_experiment")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_len", type=int, default=150)
    args = parser.parse_args()

    # Load Train Data (Concatenate Train + Val sequences)
    print(f"Building training sequences (pooling: {args.pooling})...")
    X_train, y_train, mask_train = build_sequences(args.train_data, "train", args.pooling, args.max_len)
    X_val, y_val, mask_val = build_sequences(args.train_data, "val", args.pooling, args.max_len)
    
    if X_val is not None:
        X_train_all = np.concatenate([X_train, X_val], axis=0)
        y_train_all = np.concatenate([y_train, y_val], axis=0)
        mask_train_all = np.concatenate([mask_train, mask_val], axis=0)
    else:
        X_train_all, y_train_all, mask_train_all = X_train, y_train, mask_train
        
    print(f"Training sequences shape: {X_train_all.shape}, labels: {y_train_all.shape}")
    
    # Load Test Data
    print(f"Building testing sequences (pooling: {args.pooling})...")
    X_test, y_test, mask_test = build_sequences(args.test_data, "test", args.pooling, args.max_len)
    if X_test is None:
        # Fallback to validation if test split doesn't exist
        X_test, y_test, mask_test = build_sequences(args.test_data, "val", args.pooling, args.max_len)
    print(f"Testing sequences shape: {X_test.shape}, labels: {y_test.shape}")

    train_ds = TensorDataset(torch.tensor(X_train_all), torch.tensor(y_train_all).long(), torch.tensor(mask_train_all))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).long(), torch.tensor(mask_test))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cpu") # Force CPU for task safety and stable macOS processing
    print(f"Using device: {device}")

    # Compute class weights for weighted CE
    class_counts = torch.bincount(torch.tensor(y_train_all).long())
    class_weights = len(y_train_all) / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)

    model = GRUDepressionClassifier(
        input_dim=X_train_all.shape[2], 
        hidden_dim=128, 
        num_layers=2, 
        dropout=0.3, 
        num_classes=2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Train loop
    print("\n--- Training GRU Model ---")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for seqs, labels, masks in train_loader:
            seqs, labels, masks = seqs.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(seqs, masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f}")

    # Eval loop
    print("\n--- Evaluation ---")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for seqs, labels, masks in test_loader:
            seqs, labels, masks = seqs.to(device), labels.to(device), masks.to(device)
            logits = model(seqs, masks)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
        
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)
    
    print(f"Speaker-Level Accuracy: {acc:.4f}")
    print(f"Speaker-Level F1 Score: {f1:.4f}")
    print(f"Speaker-Level ROC AUC:  {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    # Save results
    os.makedirs("output", exist_ok=True)
    txt_path = os.path.join("output", f"{args.exp_name}_results.txt")
    with open(txt_path, "w") as f:
        f.write(f"GRU Sequence Model: {args.exp_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC:  {auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    report_dict = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
    csv_data = {
        "Metric": ["Accuracy", "F1 Score", "ROC AUC", "Precision", "Recall", "Support"],
        "Class_0_Healthy": ["", report_dict['0']['f1-score'], "", report_dict['0']['precision'], report_dict['0']['recall'], report_dict['0']['support']],
        "Class_1_Depressed": ["", report_dict['1']['f1-score'], "", report_dict['1']['precision'], report_dict['1']['recall'], report_dict['1']['support']],
        "Overall": [acc, f1, auc, "", "", report_dict['macro avg']['support']]
    }
    df_res = pd.DataFrame(csv_data)
    csv_path = os.path.join("output", f"{args.exp_name}_results.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Results successfully saved to {txt_path} and {csv_path}")

if __name__ == "__main__":
    main()
