import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import sys
sys.path.append(os.path.join(os.getcwd(), "code", "classification"))
from models import ContrastiveAlignmentNet, SupConLoss

# Ensure random seeds are set for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def extract_speaker_id(path):
    return re.search(r'\d+', os.path.basename(path)).group()

def run_loso_for_layer(model_dir, layer, dims):
    print(f"\n==========================================")
    print(f"RUNNING MODMA LOSO EVALUATION (Layer {layer})")
    print(f"==========================================")
    
    # 1. Load splits and reconstruct overall dataset
    df = pd.read_csv("data/utterance_table_modma_segmented_split.csv")
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    df_all = pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)
    df_all["speaker_id"] = df_all["file_path"].apply(extract_speaker_id)
    
    feature_path = f"features/{model_dir}/features_modma_layer{layer}"
    X_train = np.load(os.path.join(feature_path, "X_train_mean.npy"))
    y_train = np.load(os.path.join(feature_path, "y_train.npy"))
    X_val = np.load(os.path.join(feature_path, "X_val_mean.npy"))
    y_val = np.load(os.path.join(feature_path, "y_val.npy"))
    X_test = np.load(os.path.join(feature_path, "X_test_mean.npy"))
    y_test = np.load(os.path.join(feature_path, "y_test.npy"))
    
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    
    unique_speakers = df_all["speaker_id"].unique()
    num_speakers = len(unique_speakers)
    print(f"Loaded {len(df_all)} segments across {num_speakers} unique speakers.")
    
    # Classifiers to evaluate
    classifiers = ["LR", "SVM-Linear", "SVM-RBF", "CLeaD", "CLeaD w/o SupCon"]
    
    # Stores segment-level predictions and probabilities for all speakers
    loso_preds = {clf: np.zeros(len(df_all)) for clf in classifiers}
    loso_probs = {clf: np.zeros(len(df_all)) for clf in classifiers}
    
    # 2. LOSO CV Loop
    for idx, test_spk in enumerate(unique_speakers):
        print(f"Fold {idx+1}/{num_speakers}: Testing Speaker {test_spk} ...")
        
        test_mask = (df_all["speaker_id"] == test_spk).values
        train_mask = ~test_mask
        
        X_train_fold, y_train_fold = X_all[train_mask], y_all[train_mask]
        X_test_fold, y_test_fold = X_all[test_mask], y_all[test_mask]
        
        # Scale features
        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_test_fold_scaled = scaler.transform(X_test_fold)
        
        # --- A. Logistic Regression ---
        clf_lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        clf_lr.fit(X_train_fold_scaled, y_train_fold)
        loso_preds["LR"][test_mask] = clf_lr.predict(X_test_fold_scaled)
        loso_probs["LR"][test_mask] = clf_lr.predict_proba(X_test_fold_scaled)[:, 1]
        
        # --- B. SVM Linear ---
        clf_svl = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=5000, random_state=42, C=0.1)
        clf_svl.fit(X_train_fold_scaled, y_train_fold)
        loso_preds["SVM-Linear"][test_mask] = clf_svl.predict(X_test_fold_scaled)
        loso_probs["SVM-Linear"][test_mask] = clf_svl.decision_function(X_test_fold_scaled)
        
        # --- C. SVM RBF ---
        clf_svr = SVC(kernel="rbf", class_weight="balanced", probability=False, tol=1e-2, max_iter=5000, random_state=42, C=1.0)
        clf_svr.fit(X_train_fold_scaled, y_train_fold)
        loso_preds["SVM-RBF"][test_mask] = clf_svr.predict(X_test_fold_scaled)
        loso_probs["SVM-RBF"][test_mask] = clf_svr.decision_function(X_test_fold_scaled)
        
        # --- D/E. CLeaD and CLeaD w/o SupCon ---
        for use_sc, clf_name in [(True, "CLeaD"), (False, "CLeaD w/o SupCon")]:
            device = torch.device("cpu")
            torch.manual_seed(42)
            
            # Sub-train-val split inside fold for DL training
            # 80/20 train/val split of the 51 training speakers
            train_spks = unique_speakers[unique_speakers != test_spk]
            np.random.seed(42)
            np.random.shuffle(train_spks)
            dl_train_spks = train_spks[:int(0.8 * len(train_spks))]
            
            dl_train_mask = df_all["speaker_id"].isin(dl_train_spks).values & train_mask
            X_tr, y_tr = X_all[dl_train_mask], y_all[dl_train_mask]
            
            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_tr).float(), torch.tensor(y_tr).long()),
                batch_size=128, shuffle=True, drop_last=True
            )
            
            class_counts = torch.bincount(torch.tensor(y_tr).long())
            class_weights = len(y_tr) / (len(class_counts) * class_counts.float())
            class_weights = class_weights.to(device)
            
            model = ContrastiveAlignmentNet(input_dim=X_all.shape[1], proj_dim=128, num_classes=2).to(device)
            criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
            criterion_supcon = SupConLoss(temperature=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            
            model.train()
            for epoch in range(5): # Fast train epochs for fold loop efficiency
                for features, labels in train_loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    proj, logits = model(features)
                    
                    ce_loss = criterion_ce(logits, labels)
                    if use_sc:
                        proj_unsqueezed = proj.unsqueeze(1)
                        supcon_loss = criterion_supcon(proj_unsqueezed, labels=labels)
                        loss = 0.5 * supcon_loss + 0.5 * ce_loss
                    else:
                        loss = ce_loss
                        
                    loss.backward()
                    optimizer.step()
                    
            model.eval()
            with torch.no_grad():
                feat_tensor = torch.tensor(X_test_fold).float().to(device)
                _, logits = model(feat_tensor)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                
                loso_preds[clf_name][test_mask] = preds.cpu().numpy()
                loso_probs[clf_name][test_mask] = probs.cpu().numpy()
                
    # 3. Compute Segment and Speaker Level Metrics
    results = []
    print("\n--- Final Consolidated LOSO Results ---")
    for clf in classifiers:
        preds = loso_preds[clf]
        probs = loso_probs[clf]
        
        acc = accuracy_score(y_all, preds)
        f1 = f1_score(y_all, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_all, probs)
        except:
            auc = 0.5
            
        # Speaker Level voting
        df_all["pred"] = preds
        df_all["prob"] = probs
        
        spk_preds = []
        spk_true = []
        for spk_id, group in df_all.groupby("speaker_id"):
            spk_preds.append(1 if group["pred"].mean() >= 0.5 else 0)
            spk_true.append(group["label"].iloc[0])
            
        spk_acc = accuracy_score(spk_true, spk_preds)
        spk_f1 = f1_score(spk_true, spk_preds, zero_division=0)
        
        print(f"[{clf}] Segment Acc: {acc:.4f} | Segment F1: {f1:.4f} | Speaker Acc: {spk_acc:.4f} | Speaker F1: {spk_f1:.4f}")
        results.append({
            "Layer": layer,
            "Model": clf,
            "Seg_Acc": acc,
            "Seg_F1": f1,
            "Seg_AUC": auc,
            "Spk_Acc": spk_acc,
            "Spk_F1": spk_f1
        })
        
    return results

def main():
    # Run for WavLM Base-Plus (layer 7) and WavLM Large (layer 14)
    res_base = run_loso_for_layer("wavlm_base_plus", 7, 768)
    res_large = run_loso_for_layer("wavlm_large", 14, 1024)
    
    # Save formatted report
    os.makedirs("output", exist_ok=True)
    report_path = "output/modma_loso_results.md"
    
    with open(report_path, "w") as f:
        f.write("# MODMA Leave-One-Speaker-Out (LOSO) Evaluation Report\n\n")
        f.write("This report summarizes Leave-One-Speaker-Out cross-validation performance on the **MODMA** dataset (52 unique speakers) across different classification backbones.\n\n")
        
        f.write("## 1. WavLM Base-Plus (Layer 7)\n\n")
        f.write("| Model | Segment Accuracy | Segment F1 | Segment AUC | Speaker Accuracy | Speaker F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for row in res_base:
            f.write(f"| {row['Model']} | {row['Seg_Acc']*100:.2f}% | {row['Seg_F1']:.4f} | {row['Seg_AUC']:.4f} | {row['Spk_Acc']*100:.2f}% | {row['Spk_F1']:.4f} |\n")
            
        f.write("\n## 2. WavLM Large (Layer 14)\n\n")
        f.write("| Model | Segment Accuracy | Segment F1 | Segment AUC | Speaker Accuracy | Speaker F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for row in res_large:
            f.write(f"| {row['Model']} | {row['Seg_Acc']*100:.2f}% | {row['Seg_F1']:.4f} | {row['Seg_AUC']:.4f} | {row['Spk_Acc']*100:.2f}% | {row['Spk_F1']:.4f} |\n")
            
    print(f"\nMODMA LOSO results report written successfully to {report_path}")

if __name__ == "__main__":
    main()
