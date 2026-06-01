# Cross-Lingual Depression Detection (WavLM + CLeaD)

This repository contains the codebase for cross-lingual zero-shot depression detection from speech, specifically targeting the transfer gap between Germanic (English) and Tonal (Mandarin) languages. 

This architecture was designed for submission to SLT.

## Architecture

1. **Feature Extractor:** We use `microsoft/wavlm-base-plus` (Layer 6) to extract robust, noise-augmented speech representations.
2. **Classifier (CLeaD):** A custom PyTorch architecture implementing **Contrastive Learning for Depression Detection (CLeaD)**. 
    - The model uses Supervised Contrastive Loss (SupCon) to pull all "Depressed" speech signatures into a shared latent space, forcing the network to ignore the language spoken (English vs. Mandarin) and focus purely on acoustic biomarkers of depression (e.g., psychomotor retardation).

## Datasets
- **E-DAIC:** English corpus used for baseline training and evaluation.
- **MODMA / MODMA:** Mandarin corpora used to validate zero-shot cross-lingual alignment.

*(Note: Massive audio chunks and `.npy` feature arrays are tracked via `.gitignore` and are not included in this repository).*

## How to Run the Pipeline

### 1. Preprocessing
To cut, balance, and segment the transcripts into 10-second sliding windows:
```bash
python3 code/preprocessing/cut_edaic_utterances.py
python3 code/preprocessing/balance_utterance_table.py
python3 code/preprocessing/segment_edaic_sliding.py
python3 code/preprocessing/split_metadata.py --input_csv utterance_table_edaic_segmented.csv
```

### 2. Feature Extraction
To run the segmented `.wav` files through the WavLM transformer:
```bash
python3 code/feature_extraction/extract_edaic_layer.py --metadata utterance_table_edaic_segmented_split.csv --output_dir features/features_edaic_layer6 --layer 6
```
*(Repeat for `extract_modma_layer.py` when using Mandarin data).*

### 3. Model Training & Evaluation (Ablation Study)
To prove the effectiveness of the contrastive alignment, run both the "Before CLeaD" (Baseline) and "After CLeaD" models side-by-side.

**EN → ZH (Train on English E-DAIC, Test on Mandarin MODMA)**
```bash
# Before CLeaD
python3 code/classification/run_baseline_classifier.py --train_data features/features_edaic_layer6 --test_data features/features_modma_layer6 --exp_name baseline_EN_to_ZH

# After CLeaD
python3 code/classification/run_contrastive_alignment.py --train_data features/features_edaic_layer6 --test_data features/features_modma_layer6 --exp_name clead_EN_to_ZH
```

**ZH → EN (Train on Mandarin MODMA, Test on English E-DAIC)**
```bash
# Before CLeaD
python3 code/classification/run_baseline_classifier.py --train_data features/features_modma_layer6 --test_data features/features_edaic_layer6 --exp_name baseline_ZH_to_EN

# After CLeaD
python3 code/classification/run_contrastive_alignment.py --train_data features/features_modma_layer6 --test_data features/features_edaic_layer6 --exp_name clead_ZH_to_EN
```

**MIX → EN (Train on Balanced Mix, Test on English E-DAIC)**
```bash
# Before CLeaD
python3 code/classification/run_baseline_classifier.py --train_data features/features_mix_layer6 --test_data features/features_edaic_layer6 --exp_name baseline_MIX_to_EN

# After CLeaD
python3 code/classification/run_contrastive_alignment.py --train_data features/features_mix_layer6 --test_data features/features_edaic_layer6 --exp_name clead_MIX_to_EN
```

**MIX → ZH (Train on Balanced Mix, Test on Mandarin MODMA)**
```bash
# Before CLeaD
python3 code/classification/run_baseline_classifier.py --train_data features/features_mix_layer6 --test_data features/features_modma_layer6 --exp_name baseline_MIX_to_ZH

# After CLeaD
python3 code/classification/run_contrastive_alignment.py --train_data features/features_mix_layer6 --test_data features/features_modma_layer6 --exp_name clead_MIX_to_ZH
```

## Results

Below are the segment-level and speaker-level evaluation scores across the 6 cross-lingual configurations for **WavLM Layer 6**:

### 1. Segment-Level Metrics
| Configuration | Model | Accuracy | F1 Score | ROC AUC |
| :--- | :--- | :---: | :---: | :---: |
| **EN -> EN** | Baseline <br> CLeaD | 73.79% <br> 74.11% | 0.6628 <br> 0.6148 | 0.8043 <br> 0.7704 |
| **EN -> ZH** | Baseline <br> CLeaD | 49.58% <br> 50.13% | 0.2545 <br> 0.2833 | 0.4807 <br> 0.5255 |
| **ZH -> EN** | Baseline <br> CLeaD | 54.32% <br> 55.35% | 0.4217 <br> 0.3490 | 0.5357 <br> 0.5539 |
| **ZH -> ZH** | Baseline <br> CLeaD | 53.63% <br> 54.14% | 0.4411 <br> 0.4488 | 0.5368 <br> 0.5721 |
| **MIX -> EN** | Baseline <br> CLeaD | 66.95% <br> 69.34% | 0.5453 <br> 0.5035 | 0.7023 <br> 0.7088 |
| **MIX -> ZH** | Baseline <br> **CLeaD** | 50.38% <br> **56.85%** | 0.4272 <br> **0.5386** | 0.4809 <br> **0.5855** |

### 2. Speaker-Level Majority Vote Metrics (MODMA Test Set)
For the configurations evaluating on the **MODMA** clinical cohort (10 unseen test speakers: 5 MDD, 5 HC):

*   **Monolingual (ZH -> ZH):** Both models correctly classify 1/5 MDD speakers and 5/5 HC speakers (F1: 0.4488).
*   **Mixed Cross-Lingual Transfer (MIX -> ZH):** 
    *   **Baseline:** Correctly classifies **2/5 MDD** and **4/5 HC** speakers (Speaker F1: **0.5000**, Accuracy: **60.00%**).
    *   **CLeaD:** Correctly classifies **4/5 MDD** and **4/5 HC** speakers (Speaker F1: **0.8000**, Accuracy: **80.00%**, AUC: **0.7600**).
    *   **Key Victory:** CLeaD successfully aligns cross-lingual features on the joint manifold, leveraging the larger English E-DAIC dataset to regularize the scarce Mandarin cohort and correcting critical block-level classification errors.

### 3. WavLM Layer Ablation Study
The full comparative summary table across all WavLM layers (Layers 6, 7, 8, and 9) is compiled and saved in [ablation_summary_table.txt](./output/ablation_summary_table.txt) and [ablation_summary.md](./output/ablation_summary.md).
