import os
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from confidence_intervals import evaluate_with_conf_int

def metric_f1(labels, preds):
    return f1_score(labels, preds, zero_division=0)
    
def metric_auc(labels, probs):
    try:
        return roc_auc_score(labels, probs)
    except:
        return 0.5

def main():
    print("# Bootstrapped Confidence Intervals (95%, N=2000)")
    print("Evaluating F1 and AUC metrics with luferrer/ConfidenceIntervals.\n")
    
    configs = [
        "EN_to_EN",
        "EN_to_ZH",
        "ZH_to_EN",
        "ZH_to_ZH",
        "MIX_to_EN",
        "MIX_to_ZH"
    ]
    
    classifiers = ["LR", "SVM-Linear", "SVM-RBF", "GRU", "CLeaD", "CLeaD w/o SupCon"]
    layers_base = [6, 7, 8, 9]
    layers_large = [12, 14, 16, 18]
    
    for i in range(4):
        base_layer = layers_base[i]
        large_layer = layers_large[i]
        
        print(f"\n### Layer Pair: Base-Plus L{base_layer} vs Large L{large_layer}")
        print("| Config | Classifier | Base F1 (95% CI) | Large F1 (95% CI) | Base AUC (95% CI) | Large AUC (95% CI) |")
        print("| :--- | :--- | :---: | :---: | :---: | :---: |")
        
        for cfg in configs:
            for clf in classifiers:
                base_dir = f"predictions/base-plus/layer{base_layer}/{cfg}/{clf}"
                large_dir = f"predictions/large/layer{large_layer}/{cfg}/{clf}"
                
                def get_ci(dir_path):
                    if not os.path.exists(os.path.join(dir_path, "y_true.npy")):
                        return "-", "-"
                        
                    y_true = np.load(os.path.join(dir_path, "y_true.npy"))
                    preds = np.load(os.path.join(dir_path, "preds.npy"))
                    probs = np.load(os.path.join(dir_path, "probs.npy"))
                    
                    f1_val, (f1_low, f1_high) = evaluate_with_conf_int(
                        samples=preds, metric=metric_f1, labels=y_true, num_bootstraps=2000, alpha=5
                    )
                    
                    auc_val, (auc_low, auc_high) = evaluate_with_conf_int(
                        samples=probs, metric=metric_auc, labels=y_true, num_bootstraps=2000, alpha=5
                    )
                    
                    return f"{f1_val:.4f} [{f1_low:.4f}, {f1_high:.4f}]", f"{auc_val:.4f} [{auc_low:.4f}, {auc_high:.4f}]"
                
                b_f1, b_auc = get_ci(base_dir)
                l_f1, l_auc = get_ci(large_dir)
                
                if b_f1 == "-" and l_f1 == "-":
                    continue
                    
                cfg_display = cfg.replace("_to_", " -> ")
                print(f"| {cfg_display} | {clf} | {b_f1} | {l_f1} | {b_auc} | {l_auc} |")

if __name__ == "__main__":
    main()
