import os
import re
import sys
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.append(os.path.join(os.getcwd(), "code", "classification"))
from gru_model import GRUDepressionClassifier
from models import ContrastiveAlignmentNet, SupConLoss

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Path to MODMA metadata to map segments back to speaker/participant IDs
modma_metadata = pd.read_csv("utterance_table_modma_segmented_split.csv")
modma_test_df = modma_metadata[modma_metadata["split"] == "test"].copy()
modma_test_df["speaker_id"] = modma_test_df["file_path"].apply(lambda x: re.search(r'\d+', os.path.basename(x)).group())

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

def evaluate_speaker_level(preds_segment, probs_segment, df_test):
    df = df_test.copy()
    df["pred"] = preds_segment
    df["prob"] = probs_segment
    
    speaker_results = []
    for spk_id, group in df.groupby("speaker_id"):
        true_label = group["label"].iloc[0]
        # Majority voting
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
    
    return f"{num_correct_mdd}/5 MDD, {num_correct_hc}/5 HC", f1_maj, acc_maj

def run_segment_classifier(train_dir, test_dir, clf_type):
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
    
    if clf_type != "lr":
        # Scale features for numerical stability and faster SVM convergence
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    if clf_type == "lr":
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
    elif clf_type == "svm_linear":
        step = 5 if len(X_train) > 5000 else 1
        X_train_sub = X_train[::step]
        y_train_sub = y_train[::step]
        
        n_splits = min(3, int((y_train_sub == 1).sum()), int((y_train_sub == 0).sum()))
        n_splits = max(n_splits, 2)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        base_clf = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=5000, random_state=42)
        param_grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
        grid = GridSearchCV(base_clf, param_grid, cv=cv, scoring="f1", n_jobs=1, refit=False)
        grid.fit(X_train_sub, y_train_sub)
        
        best_params = grid.best_params_
        clf = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=5000, random_state=42, **best_params)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.decision_function(X_test)
    elif clf_type == "svm_rbf":
        step = 5 if len(X_train) > 5000 else 1
        X_train_sub = X_train[::step]
        y_train_sub = y_train[::step]
        
        n_splits = min(3, int((y_train_sub == 1).sum()), int((y_train_sub == 0).sum()))
        n_splits = max(n_splits, 2)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        base_clf = SVC(kernel="rbf", class_weight="balanced", probability=False, cache_size=2000, tol=1e-2, max_iter=5000, random_state=42)
        param_grid = {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 0.001, 0.01]
        }
        grid = GridSearchCV(base_clf, param_grid, cv=cv, scoring="f1", n_jobs=1, refit=False)
        grid.fit(X_train_sub, y_train_sub)
        
        best_params = grid.best_params_
        clf = SVC(kernel="rbf", class_weight="balanced", probability=False, cache_size=2000, tol=1e-2, max_iter=5000, random_state=42, **best_params)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.decision_function(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except:
        auc = 0.5
        
    spk_str, spk_f1, spk_acc = "-", 0.0, 0.0
    if "modma" in test_dir:
        spk_str, spk_f1, spk_acc = evaluate_speaker_level(preds, probs, modma_test_df)
        
    return acc, f1, auc, spk_str, spk_f1, spk_acc

def build_sequences(feature_dir, split, max_len=150):
    metadata_csv = get_metadata_path(feature_dir)
    df = pd.read_csv(metadata_csv)
    df_split = df[df["split"] == split].reset_index(drop=True)
    
    if len(df_split) == 0:
        return None, None, None
        
    X = np.load(os.path.join(feature_dir, f"X_{split}_mean.npy"))
    
    # Group segments
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
        
    # Build sequences
    sequences = []
    labels = []
    for spk_id, data in speaker_data.items():
        # Sort indices based on chronological sort_keys
        sorted_indices = sorted(range(len(data["sort_keys"])), key=lambda k: data["sort_keys"][k])
        sorted_feats = [data["feats"][idx] for idx in sorted_indices]
        sequences.append(np.array(sorted_feats))
        labels.append(data["label"])
        
    num_sequences = len(sequences)
    feature_dim = X.shape[1]
    
    padded_sequences = np.zeros((num_sequences, max_len, feature_dim), dtype=np.float32)
    masks = np.zeros((num_sequences, max_len), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_len)
        padded_sequences[i, :seq_len] = seq[:seq_len]
        masks[i, :seq_len] = 1.0
        
    return padded_sequences, np.array(labels), masks

def run_gru_classifier(train_dir, test_dir, epochs=100, batch_size=16, lr=1e-3, max_len=150):
    # Reset random seeds inside the function to ensure reproducible initializations across all calls
    np.random.seed(42)
    torch.manual_seed(42)
    
    X_train, y_train, mask_train = build_sequences(train_dir, "train", max_len)
    X_val, y_val, mask_val = build_sequences(train_dir, "val", max_len)
    
    if X_val is not None:
        X_train_all = np.concatenate([X_train, X_val], axis=0)
        y_train_all = np.concatenate([y_train, y_val], axis=0)
        mask_train_all = np.concatenate([mask_train, mask_val], axis=0)
    else:
        X_train_all, y_train_all, mask_train_all = X_train, y_train, mask_train
        
    X_test, y_test, mask_test = build_sequences(test_dir, "test", max_len)
    if X_test is None:
        X_test, y_test, mask_test = build_sequences(test_dir, "val", max_len)
        
    train_ds = TensorDataset(torch.tensor(X_train_all), torch.tensor(y_train_all).long(), torch.tensor(mask_train_all))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).long(), torch.tensor(mask_test))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cpu")
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
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    model.train()
    for epoch in range(epochs):
        for seqs, labels, masks in train_loader:
            seqs, labels, masks = seqs.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(seqs, masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
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
    except:
        auc = 0.5
        
    spk_str = "-"
    if "modma" in test_dir:
        # For GRU, speaker results are directly the output predictions
        num_correct_mdd = int(all_labels[(all_labels == 1) & (all_preds == 1)].shape[0])
        num_correct_hc = int(all_labels[(all_labels == 0) & (all_preds == 0)].shape[0])
        spk_str = f"{num_correct_mdd}/5 MDD, {num_correct_hc}/5 HC"
        
    return acc, f1, auc, spk_str, f1, acc

def run_clead_classifier(train_dir, test_dir, epochs=100, batch_size=32):
    # Reset random seeds inside the function to ensure reproducible initializations
    np.random.seed(42)
    torch.manual_seed(42)
    
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
    
    model = ContrastiveAlignmentNet(input_dim=768, proj_dim=256, num_classes=2).to(device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_supcon = SupConLoss(temperature=0.1)
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
            loss = 0.5 * supcon_loss + 0.5 * ce_loss
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
        
    spk_str, spk_f1, spk_acc = "-", 0.0, 0.0
    if "modma" in test_dir:
        spk_str, spk_f1, spk_acc = evaluate_speaker_level(preds, probs, modma_test_df)
        
    return acc, f1, auc, spk_str, spk_f1, spk_acc

def main():
    configs = [
        {"name": "EN -> EN", "train": "edaic", "test": "edaic"},
        {"name": "EN -> ZH", "train": "edaic", "test": "modma"},
        {"name": "ZH -> EN", "train": "modma", "test": "edaic"},
        {"name": "ZH -> ZH", "train": "modma", "test": "modma"},
        {"name": "MIX -> EN", "train": "mix", "test": "edaic"},
        {"name": "MIX -> ZH", "train": "mix", "test": "modma"}
    ]
    
    results = []
    
    for layer in [6, 7, 8, 9]:
        print(f"\n==========================================")
        print(f"RUNNING ABLATION FOR LAYER {layer}")
        print(f"==========================================")
        
        for cfg in configs:
            cfg_name = cfg["name"]
            train_ds = cfg["train"]
            test_ds = cfg["test"]
            
            train_dir = f"features/features_{train_ds}_layer{layer}"
            test_dir = f"features/features_{test_ds}_layer{layer}"
            
            if not os.path.exists(os.path.join(train_dir, "X_train_mean.npy")):
                print(f"Skipping {cfg_name} (no features)...")
                continue
                
            # 1. Logistic Regression (Baseline)
            print(f"  Running LR for {cfg_name}...")
            acc_lr, f1_lr, auc_lr, spk_lr, spk_f1_lr, spk_acc_lr = run_segment_classifier(train_dir, test_dir, "lr")
            results.append({
                "Layer": layer, "Config": cfg_name, "Model": "LR",
                "Acc": acc_lr, "F1": f1_lr, "AUC": auc_lr,
                "Speaker_Vote": spk_lr, "Speaker_F1": spk_f1_lr, "Speaker_Acc": spk_acc_lr
            })
            
            # 2. SVM-Linear
            print(f"  Running SVM-Linear for {cfg_name}...")
            acc_svl, f1_svl, auc_svl, spk_svl, spk_f1_svl, spk_acc_svl = run_segment_classifier(train_dir, test_dir, "svm_linear")
            results.append({
                "Layer": layer, "Config": cfg_name, "Model": "SVM-Linear",
                "Acc": acc_svl, "F1": f1_svl, "AUC": auc_svl,
                "Speaker_Vote": spk_svl, "Speaker_F1": spk_f1_svl, "Speaker_Acc": spk_acc_svl
            })
            
            # 3. SVM-RBF
            print(f"  Running SVM-RBF for {cfg_name}...")
            acc_svr, f1_svr, auc_svr, spk_svr, spk_f1_svr, spk_acc_svr = run_segment_classifier(train_dir, test_dir, "svm_rbf")
            results.append({
                "Layer": layer, "Config": cfg_name, "Model": "SVM-RBF",
                "Acc": acc_svr, "F1": f1_svr, "AUC": auc_svr,
                "Speaker_Vote": spk_svr, "Speaker_F1": spk_f1_svr, "Speaker_Acc": spk_acc_svr
            })
            
            # 4. GRU Sequence Model
            print(f"  Running GRU Sequence Classifier for {cfg_name}...")
            try:
                acc_gru, f1_gru, auc_gru, spk_gru, spk_f1_gru, spk_acc_gru = run_gru_classifier(train_dir, test_dir)
                results.append({
                    "Layer": layer, "Config": cfg_name, "Model": "GRU",
                    "Acc": acc_gru, "F1": f1_gru, "AUC": auc_gru,
                    "Speaker_Vote": spk_gru, "Speaker_F1": spk_f1_gru, "Speaker_Acc": spk_acc_gru
                })
            except Exception as e:
                print(f"    GRU failed: {e}")
                
            # 5. CLeaD Representation Model
            print(f"  Running CLeaD for {cfg_name}...")
            try:
                acc_cld, f1_cld, auc_cld, spk_cld, spk_f1_cld, spk_acc_cld = run_clead_classifier(train_dir, test_dir)
                results.append({
                    "Layer": layer, "Config": cfg_name, "Model": "CLeaD",
                    "Acc": acc_cld, "F1": f1_cld, "AUC": auc_cld,
                    "Speaker_Vote": spk_cld, "Speaker_F1": spk_f1_cld, "Speaker_Acc": spk_acc_cld
                })
            except Exception as e:
                print(f"    CLeaD failed: {e}")
                
    # Compile results table
    df_res = pd.DataFrame(results)
    df_res.to_csv("output/comprehensive_ablation_results.csv", index=False)
    
    # Text table summary
    table = []
    table.append("===============================================================================================================")
    table.append("                             COMPREHENSIVE MULTI-LAYER & MULTI-MODEL ABLATION                                 ")
    table.append("===============================================================================================================")
    table.append(f"{'Layer':<6} | {'Config':<10} | {'Model':<12} | {'Seg Acc':<8} | {'Seg F1':<8} | {'Seg AUC':<8} | {'Speaker Vote (MDD/HC) / GRU Speaker Acc'}")
    table.append("---------------------------------------------------------------------------------------------------------------")
    
    prev_layer = None
    prev_config = None
    for _, row in df_res.iterrows():
        layer_str = f"{int(row['Layer']):<6}" if row['Layer'] != prev_layer else f"{'':<6}"
        config_str = f"{row['Config']:<10}" if (row['Layer'] != prev_layer or row['Config'] != prev_config) else f"{'':<10}"
        
        table.append(f"{layer_str} | {config_str} | {row['Model']:<12} | {row['Acc']:.4f} | {row['F1']:.4f} | {row['AUC']:.4f} | {row['Speaker_Vote']}")
        if row['Model'] == "CLeaD":
            table.append("---------------------------------------------------------------------------------------------------------------")
            
        prev_layer = row['Layer']
        prev_config = row['Config']
        
    table_str = "\n".join(table)
    print("\n" + table_str)
    
    with open("output/comprehensive_ablation_summary.txt", "w") as f:
        f.write(table_str)
    print("\nSaved comprehensive results to output/comprehensive_ablation_summary.txt")

    # Automatically update README.md with the newly obtained results
    try:
        readme = []
        readme.append("# Cross-Lingual Depression Detection (WavLM + CLeaD + Sequence Models)")
        readme.append("")
        readme.append("This repository contains the codebase for cross-lingual zero-shot depression detection from speech, specifically targeting the transfer gap between Germanic (English) and Tonal (Mandarin) languages.")
        readme.append("")
        readme.append("## Pipeline Structure")
        readme.append("```")
        readme.append("                                AUDIO INPUT")
        readme.append("                                     |")
        readme.append("                                     v")
        readme.append("                        10s sliding segment window")
        readme.append("                                     |")
        readme.append("                                     v")
        readme.append("                  WavLM-Large Encoder (Frozen middle layers)")
        readme.append("                                     |")
        readme.append("                                     v")
        readme.append("                         Segment Embeddings (768-d)")
        readme.append("                                     |")
        readme.append("         +---------------------------+---------------------------+")
        readme.append("         v (Static Segment Pooling)                              v (Temporal Sequence Modeling)")
        readme.append("  +-----------------------------+                         +-----------------------------+")
        readme.append("  | Mean / Max Segment Pooling  |                         |  Group Segments by Speaker  |")
        readme.append("  +-----------------------------+                         +-----------------------------+")
        readme.append("         |                                                               |")
        readme.append("         +--------------------------+                                    v")
        readme.append("         |                          |                             +-----------------------------+")
        readme.append("         v                          v                             | Chronological sort by time  |")
        readme.append("  +------------------+       +--------------+                     +-----------------------------+")
        readme.append("  | CLeaD Alignment  |       | SVM-RBF      |                                    |")
        readme.append("  | Head             |       | Classifier   |                                    v")
        readme.append("  +------------------+       +--------------+                     +-----------------------------+")
        readme.append("         |                          |                             |  Bidirectional GRU          |")
        readme.append("         v (SupCon Loss)            |                             +-----------------------------+")
        readme.append("  +------------------+              |                                    |")
        readme.append("  | Projection (128) |              |                                    v")
        readme.append("  +------------------+              |                             +-----------------------------+")
        readme.append("         |                          |                             |  Self-Attention Pooling     |")
        readme.append("         v                          v                             +-----------------------------+")
        readme.append("  +------------------+       +--------------+                                    |")
        readme.append("  | Linear Class.    |       | Support      |                                    v")
        readme.append("  | Head             |       | Vectors      |                     +-----------------------------+")
        readme.append("  +------------------+       +--------------+                     |  Linear Classifier Head     |")
        readme.append("         |                          |                             +-----------------------------+")
        readme.append("         +--------------------------+                                            |")
        readme.append("                                    |                                            v")
        readme.append("                             [Segment Preds]                       [Speaker-level Sequence Pred]")
        readme.append("                                    |")
        readme.append("                                    v (Speaker Majority Vote)")
        readme.append("                             [Speaker Preds]")
        readme.append("```")
        readme.append("")
        readme.append("## Component Overview")
        readme.append("1. **Feature Extractor:** We use `microsoft/wavlm-base-plus` (Layer 6) to extract robust, noise-augmented speech representations.")
        readme.append("2. **CLeaD (Contrastive Alignment):** A dual-head architecture using Supervised Contrastive Loss (SupCon) to pull same-class representations together across English and Mandarin domains, mapping them to a shared clinical manifold.")
        readme.append("3. **Non-Linear Classifier (SVM-RBF):** Radial Basis Function kernel SVM is applied to standardized segment embeddings to capture non-linear decision boundaries.")
        readme.append("4. **Sequence Modeling (Bi-GRU):** Chronological sequence modeling groups segment embeddings per speaker and feeds them to a bidirectional GRU with self-attention pooling to capture temporal trajectories.")
        readme.append("")
        readme.append("## Datasets")
        readme.append("- **E-DAIC:** English corpus used for baseline training and evaluation.")
        readme.append("- **MODMA:** Mandarin corpus used to validate zero-shot cross-lingual alignment.")
        readme.append("")
        readme.append("## How to Run the Pipeline")
        readme.append("")
        readme.append("### 1. Preprocessing")
        readme.append("To segment the audio datasets into 10-second sliding windows:")
        readme.append("```bash")
        readme.append("python3 code/preprocessing/segment_edaic_sliding.py")
        readme.append("python3 code/preprocessing/split_metadata.py --input_csv utterance_table_edaic_segmented.csv")
        readme.append("```")
        readme.append("")
        readme.append("### 2. Feature Extraction")
        readme.append("To extract the mean, max, and concatenated pooling features across multiple layers:")
        readme.append("```bash")
        readme.append("python3 extract_ablation_features.py")
        readme.append("```")
        readme.append("")
        readme.append("### 3. Run Comprehensive Multi-Model Ablation Study")
        readme.append("To train and evaluate LR, SVM-Linear, SVM-RBF, Bi-GRU, and CLeaD across all configurations and layers:")
        readme.append("```bash")
        readme.append("python3 run_comprehensive_ablation.py")
        readme.append("```")
        readme.append("")
        readme.append("## Results")
        readme.append("")
        readme.append("Below are the segment-level and speaker-level evaluation scores obtained from the comprehensive ablation run:")
        readme.append("")
        readme.append("### 1. Segment-Level Metrics (WavLM Layer 6)")
        readme.append("| Configuration | Model | Accuracy | F1 Score | ROC AUC |")
        readme.append("| :--- | :--- | :---: | :---: | :---: |")
        
        df_l6 = df_res[df_res["Layer"] == 6]
        for cfg_name, group in df_l6.groupby("Config", sort=False):
            first = True
            for _, row in group.iterrows():
                cfg_str = f"**{cfg_name}**" if first else ""
                readme.append(f"| {cfg_str} | {row['Model']} | {row['Acc']*100:.2f}% | {row['F1']:.4f} | {row['AUC']:.4f} |")
                first = False
            readme.append("| | | | | |")
            
        readme.append("")
        readme.append("### 2. Speaker-Level Majority Vote Metrics (MODMA Test Set)")
        readme.append("| Configuration | Model | MDD Correct | HC Correct | Speaker Acc |")
        readme.append("| :--- | :--- | :---: | :---: | :---: |")
        
        # Extract speaker level metrics for ZH->ZH and MIX->ZH on Layer 6
        for cfg_name in ["ZH -> ZH", "MIX -> ZH"]:
            df_cfg = df_l6[(df_l6["Config"] == cfg_name) & (df_l6["Model"].isin(["LR", "GRU", "CLeaD"]))]
            first = True
            for _, row in df_cfg.iterrows():
                cfg_str = f"**{cfg_name}**" if first else ""
                spk_vote = row['Speaker_Vote']
                mdd_corr = "N/A"
                hc_corr = "N/A"
                spk_acc_str = f"{row['Speaker_Acc']*100:.2f}%" if not pd.isna(row['Speaker_Acc']) else "N/A"
                if isinstance(spk_vote, str) and "MDD" in spk_vote:
                    parts = spk_vote.split(",")
                    mdd_corr = parts[0].split("/")[0].strip() + "/5"
                    hc_corr = parts[1].split("/")[0].strip() + "/5"
                readme.append(f"| {cfg_str} | {row['Model']} | {mdd_corr} | {hc_corr} | {spk_acc_str} |")
                first = False
            readme.append("| | | | | |")
            
        readme.append("")
        readme.append("### 3. WavLM Layer Ablation Study (MIX -> ZH Transfer)")
        readme.append("| WavLM Layer | Model | Segment Accuracy | Segment F1 | Segment AUC | Speaker Vote (MDD/HC) |")
        readme.append("| :---: | :--- | :---: | :---: | :---: | :--- |")
        
        df_mix_zh = df_res[df_res["Config"] == "MIX -> ZH"]
        for layer_val in [6, 7, 8, 9]:
            df_layer = df_mix_zh[(df_mix_zh["Layer"] == layer_val) & (df_mix_zh["Model"].isin(["LR", "GRU", "CLeaD"]))]
            first = True
            for _, row in df_layer.iterrows():
                l_str = f"**Layer {layer_val}**" if first else ""
                readme.append(f"| {l_str} | {row['Model']} | {row['Acc']*100:.2f}% | {row['F1']:.4f} | {row['AUC']:.4f} | {row['Speaker_Vote']} |")
                first = False
            readme.append("| | | | | | |")
            
        with open("README.md", "w") as f_rm:
            f_rm.write("\n".join(readme))
        print("Successfully updated README.md with the pipeline structure and fresh metrics!")
    except Exception as e_rm:
        print(f"Failed to auto-generate README.md: {e_rm}")

if __name__ == "__main__":
    main()
