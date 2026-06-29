import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import sys
sys.path.append(os.path.join(os.getcwd(), "code", "classification"))
from models import ContrastiveAlignmentNet, SupConLoss

# Seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load MODMA test labels for evaluation
modma_metadata = pd.read_csv("data/utterance_table_modma_segmented_split.csv")
modma_test_df = modma_metadata[modma_metadata["split"] == "test"].copy()
modma_test_df["speaker_id"] = modma_test_df["file_path"].apply(lambda x: re.search(r'\d+', os.path.basename(x)).group())

def evaluate_speaker_level(preds_segment, probs_segment, df_test):
    df = df_test.copy()
    df["pred"] = preds_segment
    df["prob"] = probs_segment
    
    speaker_results = []
    for spk_id, group in df.groupby("speaker_id"):
        true_label = group["label"].iloc[0]
        maj_vote = 1 if group["pred"].mean() >= 0.5 else 0
        avg_prob = group["prob"].mean()
        prob_vote = 1 if avg_prob >= 0.5 else 0
        
        speaker_results.append({
            "speaker_id": spk_id,
            "true_label": true_label,
            "maj_vote": maj_vote,
            "avg_prob": avg_prob,
            "prob_vote": prob_vote
        })
        
    df_spk = pd.DataFrame(speaker_results)
    acc_maj = accuracy_score(df_spk["true_label"], df_spk["maj_vote"])
    f1_maj = f1_score(df_spk["true_label"], df_spk["maj_vote"], zero_division=0)
    
    num_correct_mdd = int(df_spk[(df_spk["true_label"] == 1) & (df_spk["maj_vote"] == 1)].shape[0])
    num_correct_hc = int(df_spk[(df_spk["true_label"] == 0) & (df_spk["maj_vote"] == 0)].shape[0])
    total_mdd = int(df_spk[df_spk["true_label"] == 1].shape[0])
    total_hc = int(df_spk[df_spk["true_label"] == 0].shape[0])
    
    return f"{num_correct_mdd}/{total_mdd} MDD, {num_correct_hc}/{total_hc} HC", f1_maj, acc_maj

def run_clead_sweep(train_dir, test_dir, tau, lam, epochs=10, batch_size=128):
    # Load features
    X_train = np.concatenate([
        np.load(os.path.join(train_dir, "X_train_mean.npy")),
        np.load(os.path.join(train_dir, "X_val_mean.npy"))
    ], axis=0)
    y_train = np.concatenate([
        np.load(os.path.join(train_dir, "y_train.npy")),
        np.load(os.path.join(train_dir, "y_val.npy"))
    ], axis=0)
    
    X_test = np.load(os.path.join(test_dir, "X_test_mean.npy"))
    y_test = np.load(os.path.join(test_dir, "y_test.npy"))
    
    full_train = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True, drop_last=True)
    
    device = torch.device("cpu")
    class_counts = torch.bincount(torch.tensor(y_train).long())
    class_weights = len(y_train) / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)
    
    # Force seed inside for consistency across runs
    np.random.seed(42)
    torch.manual_seed(42)
    
    model = ContrastiveAlignmentNet(input_dim=X_train.shape[1], proj_dim=256, num_classes=2).to(device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_supcon = SupConLoss(temperature=tau)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    model.train()
    for epoch in range(epochs):
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            proj, logits = model(features)
            
            proj_unsqueezed = proj.unsqueeze(1)
            supcon_loss = criterion_supcon(proj_unsqueezed, labels=labels)
            ce_loss = criterion_ce(logits, labels)
            
            # Weighted loss
            loss = lam * supcon_loss + (1 - lam) * ce_loss
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_preds = []
    all_probs = []
    
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)
            _, logits = model(features)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    preds = np.array(all_preds)
    probs = np.array(all_probs)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except:
        auc = 0.5
        
    spk_str, spk_f1, spk_acc = evaluate_speaker_level(preds, probs, modma_test_df)
    return acc, f1, auc, spk_str, spk_acc

def main():
    temperatures = [0.05, 0.1, 0.2]
    lambdas = [0.3, 0.5, 0.7]
    
    sweep_results = []
    
    # 1. WavLM Base-Plus Layer 7 (MIX -> ZH Transfer)
    print("Running hyperparameter sweep on WavLM Base-Plus (Layer 7)...")
    train_dir_base = "features/wavlm_base_plus/features_mix_layer7"
    test_dir_base = "features/wavlm_base_plus/features_modma_layer7"
    
    for tau in temperatures:
        for lam in lambdas:
            print(f"  Evaluating base-plus with tau={tau}, lambda={lam}...")
            acc, f1, auc, spk_vote, spk_acc = run_clead_sweep(train_dir_base, test_dir_base, tau, lam)
            sweep_results.append({
                "Model": "WavLM Base-Plus",
                "Tau": tau,
                "Lambda": lam,
                "Acc": acc,
                "F1": f1,
                "AUC": auc,
                "Spk_Vote": spk_vote,
                "Spk_Acc": spk_acc
            })

    # 2. WavLM Large Layer 14 (MIX -> ZH Transfer)
    print("\nRunning hyperparameter sweep on WavLM Large (Layer 14)...")
    train_dir_large = "features/wavlm_large/features_mix_layer14"
    test_dir_large = "features/wavlm_large/features_modma_layer14"
    
    for tau in temperatures:
        for lam in lambdas:
            print(f"  Evaluating large with tau={tau}, lambda={lam}...")
            acc, f1, auc, spk_vote, spk_acc = run_clead_sweep(train_dir_large, test_dir_large, tau, lam)
            sweep_results.append({
                "Model": "WavLM Large",
                "Tau": tau,
                "Lambda": lam,
                "Acc": acc,
                "F1": f1,
                "AUC": auc,
                "Spk_Vote": spk_vote,
                "Spk_Acc": spk_acc
            })
            
    # Save formatted report
    os.makedirs("output", exist_ok=True)
    report_path = "output/hyperparameter_sweep_results.md"
    
    with open(report_path, "w") as f:
        f.write("# Hyperparameter Sweep Report for CLeaD Contrastive Alignment\n\n")
        f.write("This report summarizes the performance of CLeaD under different values of supervised contrastive loss temperature ($\\tau$) and loss weighting weight ($\\lambda$) on the **MIX -> ZH** (cross-lingual transfer) task.\n\n")
        
        f.write("## 1. WavLM Base-Plus (Layer 7)\n\n")
        f.write("| Temperature ($\\tau$) | Loss Weight ($\\lambda$) | Segment Acc | Segment F1 | Segment AUC | Speaker Vote (MDD/HC) | Speaker Acc |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :--- | :---: |\n")
        for row in sweep_results:
            if row["Model"] == "WavLM Base-Plus":
                f.write(f"| {row['Tau']} | {row['Lambda']} | {row['Acc']*100:.2f}% | {row['F1']:.4f} | {row['AUC']:.4f} | {row['Spk_Vote']} | {row['Spk_Acc']*100:.2f}% |\n")
                
        f.write("\n## 2. WavLM Large (Layer 14)\n\n")
        f.write("| Temperature ($\\tau$) | Loss Weight ($\\lambda$) | Segment Acc | Segment F1 | Segment AUC | Speaker Vote (MDD/HC) | Speaker Acc |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :--- | :---: |\n")
        for row in sweep_results:
            if row["Model"] == "WavLM Large":
                f.write(f"| {row['Tau']} | {row['Lambda']} | {row['Acc']*100:.2f}% | {row['F1']:.4f} | {row['AUC']:.4f} | {row['Spk_Vote']} | {row['Spk_Acc']*100:.2f}% |\n")
                
    print(f"\nHyperparameter sweep results report written successfully to {report_path}")

if __name__ == "__main__":
    main()
