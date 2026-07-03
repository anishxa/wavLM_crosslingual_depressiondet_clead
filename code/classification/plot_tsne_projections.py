import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.join(os.getcwd(), "code", "classification"))
from models import ContrastiveAlignmentNet, SupConLoss

def main():
    # Paths and configurations
    layer = 7
    model_dir = "wavlm_base_plus"
    train_dir = f"features/{model_dir}/features_mix_layer{layer}"
    test_edaic_dir = f"features/{model_dir}/features_edaic_layer{layer}"
    test_modma_dir = f"features/{model_dir}/features_modma_layer{layer}"
    
    # Ensure features are present
    if not os.path.exists(os.path.join(train_dir, "X_train_mean.npy")):
        print(f"Features not found at {train_dir}. Skipping t-SNE plot.")
        return

    # Load combined training features
    X_train = np.concatenate([
        np.load(os.path.join(train_dir, "X_train_mean.npy")),
        np.load(os.path.join(train_dir, "X_val_mean.npy"))
    ], axis=0)
    y_train = np.concatenate([
        np.load(os.path.join(train_dir, "y_train.npy")),
        np.load(os.path.join(train_dir, "y_val.npy"))
    ], axis=0)

    # Load English and Mandarin test sets for visualization
    X_edaic_test = np.load(os.path.join(test_edaic_dir, "X_test_mean.npy"))
    y_edaic_test = np.load(os.path.join(test_edaic_dir, "y_test.npy"))
    lang_edaic_test = np.zeros_like(y_edaic_test) # 0 for English

    X_modma_test = np.load(os.path.join(test_modma_dir, "X_test_mean.npy"))
    y_modma_test = np.load(os.path.join(test_modma_dir, "y_test.npy"))
    lang_modma_test = np.ones_like(y_modma_test) # 1 for Mandarin

    # Combine test sets
    X_test = np.concatenate([X_edaic_test, X_modma_test], axis=0)
    y_test = np.concatenate([y_edaic_test, y_modma_test], axis=0)
    lang_test = np.concatenate([lang_edaic_test, lang_modma_test], axis=0)

    # Initialize alignment model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing ContrastiveAlignmentNet on device: {device}...")
    torch.manual_seed(42)
    np.random.seed(42)
    model = ContrastiveAlignmentNet(input_dim=X_train.shape[1], proj_dim=128, num_classes=2).to(device)

    # Get projections before alignment
    model.eval()
    with torch.no_grad():
        projs_before, _ = model(torch.tensor(X_test).float().to(device))
        projs_before = projs_before.cpu().numpy()

    # Train model (CLeaD)
    print("Training CLeaD model...")
    class_counts = torch.bincount(torch.tensor(y_train).long())
    class_weights = len(y_train) / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)
    
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_supcon = SupConLoss(temperature=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()),
        batch_size=32, shuffle=True, drop_last=True
    )

    model.train()
    for epoch in range(30):
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

    # Get projections after alignment
    model.eval()
    with torch.no_grad():
        projs_after, _ = model(torch.tensor(X_test).float().to(device))
        projs_after = projs_after.cpu().numpy()

    # Run t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_before = tsne.fit_transform(projs_before)
    tsne_after = tsne.fit_transform(projs_after)

    # Plotting
    print("Plotting projections...")
    
    # Map combinations of language & label to names/colors
    labels_combo = []
    for lang, label in zip(lang_test, y_test):
        lang_str = "English (E-DAIC)" if lang == 0 else "Mandarin (MODMA)"
        label_str = "HC" if label == 0 else "MDD"
        labels_combo.append(f"{lang_str} - {label_str}")
        
    labels_combo = np.array(labels_combo)

    # Set up matplotlib figure with a top/down (2x1) layout
    fig, axes = plt.subplots(2, 1, figsize=(10, 16), dpi=300)
    
    # Plotting styles for grayscale safety and caption matching
    plot_styles = {
        "English (E-DAIC) - HC": {
            "marker": "o",
            "facecolor": "none",
            "edgecolor": "#1f77b4",
            "linewidth": 1.5,
            "s": 35
        },
        "English (E-DAIC) - MDD": {
            "marker": "o",
            "facecolor": "#d62728",
            "edgecolor": "#d62728",
            "linewidth": 1.0,
            "s": 35
        },
        "Mandarin (MODMA) - HC": {
            "marker": "^",
            "facecolor": "none",
            "edgecolor": "#1f77b4",  # Using same blue for HC to be grayscale-safe
            "linewidth": 1.5,
            "s": 40
        },
        "Mandarin (MODMA) - MDD": {
            "marker": "^",
            "facecolor": "#d62728",  # Using same red for MDD to be grayscale-safe
            "edgecolor": "#d62728",
            "linewidth": 1.0,
            "s": 40
        }
    }

    # Before CLeaD
    for category, style in plot_styles.items():
        idx = (labels_combo == category)
        axes[0].scatter(
            tsne_before[idx, 0], tsne_before[idx, 1],
            marker=style["marker"],
            facecolors=style["facecolor"],
            edgecolors=style["edgecolor"],
            linewidths=style["linewidth"],
            s=style["s"],
            label=category,
            alpha=0.75
        )
    axes[0].set_title("128-d Projection Space BEFORE CLeaD Training", fontsize=14, fontweight='bold', pad=15)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_xlabel("t-SNE Dimension 1", fontsize=12)
    axes[0].set_ylabel("t-SNE Dimension 2", fontsize=12)

    # After CLeaD
    for category, style in plot_styles.items():
        idx = (labels_combo == category)
        axes[1].scatter(
            tsne_after[idx, 0], tsne_after[idx, 1],
            marker=style["marker"],
            facecolors=style["facecolor"],
            edgecolors=style["edgecolor"],
            linewidths=style["linewidth"],
            s=style["s"],
            label=category,
            alpha=0.75
        )
    axes[1].set_title("128-d Projection Space AFTER CLeaD Training", fontsize=14, fontweight='bold', pad=15)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=12)
    axes[1].set_ylabel("t-SNE Dimension 2", fontsize=12)

    # Global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=11, frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    # Save Plot
    os.makedirs("output", exist_ok=True)
    out_path = "output/clead_tsne_projection.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"t-SNE visualization saved to {out_path}!")

if __name__ == "__main__":
    main()
