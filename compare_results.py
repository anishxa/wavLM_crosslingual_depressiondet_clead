import pandas as pd
import numpy as np
import os

def load_and_prepare(csv_path, label):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df["Model_Variant"] = label
    return df

def generate_comparison():
    df_base = load_and_prepare("output/comprehensive_ablation_results_base-plus.csv", "Base-Plus")
    df_large = load_and_prepare("output/comprehensive_ablation_results_large.csv", "Large")
    
    if df_base is None or df_large is None:
        print("Both base-plus and large results CSV files are required to generate comparison.")
        return
        
    # Mapping of layer indices for fair comparison
    # Base: 6 -> Large: 12
    # Base: 7 -> Large: 14
    # Base: 8 -> Large: 16
    # Base: 9 -> Large: 18
    layer_map_base_to_large = {6: 12, 7: 14, 8: 16, 9: 18}
    
    # We will construct a nice side-by-side comparison table
    # Columns: Config, Model, Layer Pair, Base-Plus Acc/F1, Large Acc/F1, Speaker Vote (Base vs Large)
    records = []
    
    for base_layer, large_layer in layer_map_base_to_large.items():
        df_b_lay = df_base[df_base["Layer"] == base_layer]
        df_l_lay = df_large[df_large["Layer"] == large_layer]
        
        # Merge on Config and Model
        merged = pd.merge(
            df_b_lay, df_l_lay, 
            on=["Config", "Model"], 
            suffixes=("_base", "_large")
        )
        
        for _, row in merged.iterrows():
            records.append({
                "Config": row["Config"],
                "Model": row["Model"],
                "Layers (Base / Large)": f"L{base_layer} / L{large_layer}",
                "Base_Acc": row["Acc_base"],
                "Base_F1": row["F1_base"],
                "Base_Spk_Acc": row["Speaker_Acc_base"],
                "Base_Spk_Vote": row["Speaker_Vote_base"],
                "Large_Acc": row["Acc_large"],
                "Large_F1": row["F1_large"],
                "Large_Spk_Acc": row["Speaker_Acc_large"],
                "Large_Spk_Vote": row["Speaker_Vote_large"]
            })
            
    df_comp = pd.DataFrame(records)
    
    # Save the full comparison to CSV
    df_comp.to_csv("output/model_comparison_detailed.csv", index=False)
    
    # Now build a markdown report
    markdown = []
    markdown.append("# Comparison Report: WavLM Base-Plus vs. WavLM Large")
    markdown.append("")
    markdown.append("This report compares the performance of `microsoft/wavlm-base-plus` and `microsoft/wavlm-large` models in the cross-lingual depression detection pipeline across different configurations, architectures, and layers.")
    markdown.append("")
    
    # 1. Zero-Shot Cross-Lingual Transfer Comparison (EN -> ZH & ZH -> EN)
    markdown.append("## 1. Zero-Shot Cross-Lingual Transfer")
    markdown.append("Zero-shot cross-lingual transfer tests the model's ability to generalize to a completely unseen language (e.g. English trained model evaluated on Mandarin segments and vice versa).")
    markdown.append("")
    
    for config in ["EN -> ZH", "ZH -> EN"]:
        markdown.append(f"### Configuration: {config}")
        markdown.append("| Layers | Model | Base-Plus Acc | Large Acc | Base-Plus F1 | Large F1 | Base Speaker Acc | Large Speaker Acc |")
        markdown.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        
        df_cfg = df_comp[df_comp["Config"] == config]
        for _, row in df_cfg.iterrows():
            base_acc_str = f"{row['Base_Acc']*100:.2f}%"
            large_acc_str = f"{row['Large_Acc']*100:.2f}%"
            # Format improvements/diffs
            if row['Large_Acc'] > row['Base_Acc']:
                large_acc_str = f"**{large_acc_str}** (+{(row['Large_Acc'] - row['Base_Acc'])*100:.1f}%)"
            elif row['Large_Acc'] < row['Base_Acc']:
                large_acc_str = f"{large_acc_str} (-{(row['Base_Acc'] - row['Large_Acc'])*100:.1f}%)"
                
            base_f1_str = f"{row['Base_F1']:.4f}"
            large_f1_str = f"{row['Large_F1']:.4f}"
            if row['Large_F1'] > row['Base_F1']:
                large_f1_str = f"**{large_f1_str}** (+{row['Large_F1'] - row['Base_F1']:.3f})"
            elif row['Large_F1'] < row['Base_F1']:
                large_f1_str = f"{large_f1_str} (-{row['Base_F1'] - row['Large_F1']:.3f})"
                
            base_spk_acc_str = f"{row['Base_Spk_Acc']*100:.1f}%" if not pd.isna(row['Base_Spk_Acc']) and row['Base_Spk_Acc'] > 0 else "N/A"
            large_spk_acc_str = f"{row['Large_Spk_Acc']*100:.1f}%" if not pd.isna(row['Large_Spk_Acc']) and row['Large_Spk_Acc'] > 0 else "N/A"
            if large_spk_acc_str != "N/A" and base_spk_acc_str != "N/A":
                if row['Large_Spk_Acc'] > row['Base_Spk_Acc']:
                    large_spk_acc_str = f"**{large_spk_acc_str}** (+{(row['Large_Spk_Acc'] - row['Base_Spk_Acc'])*100:.1f}%)"
                elif row['Large_Spk_Acc'] < row['Base_Spk_Acc']:
                    large_spk_acc_str = f"{large_spk_acc_str} (-{(row['Base_Spk_Acc'] - row['Large_Spk_Acc'])*100:.1f}%)"
                    
            markdown.append(f"| {row['Layers (Base / Large)']} | {row['Model']} | {base_acc_str} | {large_acc_str} | {base_f1_str} | {large_f1_str} | {base_spk_acc_str} | {large_spk_acc_str} |")
        markdown.append("")
        
    # 2. Monolingual Baselines Comparison (EN -> EN & ZH -> ZH)
    markdown.append("## 2. Monolingual Baselines")
    markdown.append("Monolingual configurations train and test on the same language/domain to establish performance upper bounds.")
    markdown.append("")
    
    for config in ["EN -> EN", "ZH -> ZH"]:
        markdown.append(f"### Configuration: {config}")
        markdown.append("| Layers | Model | Base-Plus Acc | Large Acc | Base-Plus F1 | Large F1 | Base Speaker Acc | Large Speaker Acc |")
        markdown.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        
        df_cfg = df_comp[df_comp["Config"] == config]
        for _, row in df_cfg.iterrows():
            base_acc_str = f"{row['Base_Acc']*100:.2f}%"
            large_acc_str = f"{row['Large_Acc']*100:.2f}%"
            if row['Large_Acc'] > row['Base_Acc']:
                large_acc_str = f"**{large_acc_str}** (+{(row['Large_Acc'] - row['Base_Acc'])*100:.1f}%)"
            elif row['Large_Acc'] < row['Base_Acc']:
                large_acc_str = f"{large_acc_str} (-{(row['Base_Acc'] - row['Large_Acc'])*100:.1f}%)"
                
            base_f1_str = f"{row['Base_F1']:.4f}"
            large_f1_str = f"{row['Large_F1']:.4f}"
            if row['Large_F1'] > row['Base_F1']:
                large_f1_str = f"**{large_f1_str}** (+{row['Large_F1'] - row['Base_F1']:.3f})"
            elif row['Large_F1'] < row['Base_F1']:
                large_f1_str = f"{large_f1_str} (-{row['Base_F1'] - row['Large_F1']:.3f})"
                
            base_spk_acc_str = f"{row['Base_Spk_Acc']*100:.1f}%" if not pd.isna(row['Base_Spk_Acc']) and row['Base_Spk_Acc'] > 0 else "N/A"
            large_spk_acc_str = f"{row['Large_Spk_Acc']*100:.1f}%" if not pd.isna(row['Large_Spk_Acc']) and row['Large_Spk_Acc'] > 0 else "N/A"
            if large_spk_acc_str != "N/A" and base_spk_acc_str != "N/A":
                if row['Large_Spk_Acc'] > row['Base_Spk_Acc']:
                    large_spk_acc_str = f"**{large_spk_acc_str}** (+{(row['Large_Spk_Acc'] - row['Base_Spk_Acc'])*100:.1f}%)"
                elif row['Large_Spk_Acc'] < row['Base_Spk_Acc']:
                    large_spk_acc_str = f"{large_spk_acc_str} (-{(row['Base_Spk_Acc'] - row['Large_Spk_Acc'])*100:.1f}%)"
                    
            markdown.append(f"| {row['Layers (Base / Large)']} | {row['Model']} | {base_acc_str} | {large_acc_str} | {base_f1_str} | {large_f1_str} | {base_spk_acc_str} | {large_spk_acc_str} |")
        markdown.append("")
        
    # 3. Mixed Domain Generalization Comparison (MIX -> EN & MIX -> ZH)
    markdown.append("## 3. Mixed-Domain Generalization")
    markdown.append("Mixed-domain models train on a pooled combination of English and Mandarin speech, then test on monolingual domains to leverage multi-lingual representations.")
    markdown.append("")
    
    for config in ["MIX -> EN", "MIX -> ZH"]:
        markdown.append(f"### Configuration: {config}")
        markdown.append("| Layers | Model | Base-Plus Acc | Large Acc | Base-Plus F1 | Large F1 | Base Speaker Acc | Large Speaker Acc |")
        markdown.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        
        df_cfg = df_comp[df_comp["Config"] == config]
        for _, row in df_cfg.iterrows():
            base_acc_str = f"{row['Base_Acc']*100:.2f}%"
            large_acc_str = f"{row['Large_Acc']*100:.2f}%"
            if row['Large_Acc'] > row['Base_Acc']:
                large_acc_str = f"**{large_acc_str}** (+{(row['Large_Acc'] - row['Base_Acc'])*100:.1f}%)"
            elif row['Large_Acc'] < row['Base_Acc']:
                large_acc_str = f"{large_acc_str} (-{(row['Base_Acc'] - row['Large_Acc'])*100:.1f}%)"
                
            base_f1_str = f"{row['Base_F1']:.4f}"
            large_f1_str = f"{row['Large_F1']:.4f}"
            if row['Large_F1'] > row['Base_F1']:
                large_f1_str = f"**{large_f1_str}** (+{row['Large_F1'] - row['Base_F1']:.3f})"
            elif row['Large_F1'] < row['Base_F1']:
                large_f1_str = f"{large_f1_str} (-{row['Base_F1'] - row['Large_F1']:.3f})"
                
            base_spk_acc_str = f"{row['Base_Spk_Acc']*100:.1f}%" if not pd.isna(row['Base_Spk_Acc']) and row['Base_Spk_Acc'] > 0 else "N/A"
            large_spk_acc_str = f"{row['Large_Spk_Acc']*100:.1f}%" if not pd.isna(row['Large_Spk_Acc']) and row['Large_Spk_Acc'] > 0 else "N/A"
            if large_spk_acc_str != "N/A" and base_spk_acc_str != "N/A":
                if row['Large_Spk_Acc'] > row['Base_Spk_Acc']:
                    large_spk_acc_str = f"**{large_spk_acc_str}** (+{(row['Large_Spk_Acc'] - row['Base_Spk_Acc'])*100:.1f}%)"
                elif row['Large_Spk_Acc'] < row['Base_Spk_Acc']:
                    large_spk_acc_str = f"{large_spk_acc_str} (-{(row['Base_Spk_Acc'] - row['Large_Spk_Acc'])*100:.1f}%)"
                    
            markdown.append(f"| {row['Layers (Base / Large)']} | {row['Model']} | {base_acc_str} | {large_acc_str} | {base_f1_str} | {large_f1_str} | {base_spk_acc_str} | {large_spk_acc_str} |")
        markdown.append("")
        
    # Summary of Findings
    markdown.append("## 4. Key Findings & Insights")
    markdown.append("")
    # We will write a dynamic summary based on the actual numbers
    # First, let's find the best configuration/model improvements
    best_imp_acc = df_comp.loc[(df_comp["Large_Acc"] - df_comp["Base_Acc"]).idxmax()]
    best_imp_f1 = df_comp.loc[(df_comp["Large_F1"] - df_comp["Base_F1"]).idxmax()]
    
    markdown.append(f"- **Max Accuracy Improvement:** The model variant `Large` showed the greatest accuracy gain in **{best_imp_acc['Config']}** using **{best_imp_acc['Model']}** on **{best_imp_acc['Layers (Base / Large)']}**, improving by **+{(best_imp_acc['Large_Acc'] - best_imp_acc['Base_Acc'])*100:.1f}%** (from {best_imp_acc['Base_Acc']*100:.1f}% to {best_imp_acc['Large_Acc']*100:.1f}%).")
    markdown.append(f"- **Max F1 Score Improvement:** The greatest F1 score gain was in **{best_imp_f1['Config']}** using **{best_imp_f1['Model']}** on **{best_imp_f1['Layers (Base / Large)']}**, improving by **+{best_imp_f1['Large_F1'] - best_imp_f1['Base_F1']:.3f}** (from {best_imp_f1['Base_F1']:.4f} to {best_imp_f1['Large_F1']:.4f}).")
    
    # Calculate average accuracies per model type
    avg_acc_base = df_comp.groupby("Model")["Base_Acc"].mean()
    avg_acc_large = df_comp.groupby("Model")["Large_Acc"].mean()
    
    markdown.append("- **Average Segment Accuracy by Classifier:**")
    markdown.append("  | Classifier | Base-Plus Avg Acc | Large Avg Acc | Gain |")
    markdown.append("  | :--- | :---: | :---: | :---: |")
    for clf in avg_acc_base.index:
        gain = avg_acc_large[clf] - avg_acc_base[clf]
        gain_str = f"+{gain*100:.1f}%" if gain >= 0 else f"{gain*100:.1f}%"
        markdown.append(f"  | {clf} | {avg_acc_base[clf]*100:.1f}% | {avg_acc_large[clf]*100:.1f}% | {gain_str} |")
        
    out_path = "output/model_comparison.md"
    with open(out_path, "w") as f:
        f.write("\n".join(markdown))
        
    print(f"Generated comparison report at {out_path}!")

if __name__ == "__main__":
    generate_comparison()
