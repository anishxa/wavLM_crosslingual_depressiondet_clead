# Multi-Layer WavLM Ablation for Cross-Lingual Depression Detection: Contrastive Representation Alignment and Chronological Sequence Classifiers

This repository contains the codebase for cross-lingual zero-shot depression detection from speech, targeting the transfer gap between Germanic (English) and Tonal (Mandarin) languages. We perform a multi-layer evaluation using both `microsoft/wavlm-base-plus` and `microsoft/wavlm-large` speech encoders.

## Master Pipeline Structure
```
                                AUDIO INPUT
                                     |
                                     v
                        10s sliding segment window
                                     |
                                     v
                 WavLM Speech Encoder (Frozen middle layers)
                  * Base-Plus (768-d)  |  * Large (1024-d)
                                     |
                                     v
                             Segment Embeddings
                                     |
         +---------------------------+---------------------------+
         v (Static Segment Pooling)                              v (Temporal Sequence Modeling)
  +-----------------------------+                         +-----------------------------+
  | Mean Segment Pooling        |                         |  Group Segments by Speaker  |
  +-----------------------------+                         +-----------------------------+
         |                                                               |
         +--------------------------+                                    v
         |                          |                             +-----------------------------+
         v                          v                             | Chronological sort by time  |
  +------------------+       +--------------+                     +-----------------------------+
  | CLeaD Alignment  |       | SVM-RBF      |                                    |
  | Head             |       | Classifier   |                                    v
  +------------------+       +--------------+                     +-----------------------------+
         |                          |                             |  Bidirectional GRU          |
         v (SupCon Loss)            |                             +-----------------------------+
  +------------------+              |                                    |
  | Projection (256) |              |                                    v
  +------------------+              |                             +-----------------------------+
         |                          |                             |  Self-Attention Pooling     |
         v                          v                             +-----------------------------+
  +------------------+       +--------------+                                    |
  | Linear Class.    |       | Support      |                                    v
  | Head             |       | Vectors      |                     +-----------------------------+
  +------------------+       +--------------+                     |  Linear Classifier Head     |
         |                          |                             +-----------------------------+
         +--------------------------+                                            |
                                    |                                            v
                             [Segment Preds]                       [Speaker-level Sequence Pred]
                                    |
                                    v (Speaker Majority Vote)
                             [Speaker Preds]
```

## Component Overview
1. **Feature Extractor:** We compare `microsoft/wavlm-base-plus` (using layers 6, 7, 8, 9) and `microsoft/wavlm-large` (using layers 12, 14, 16, 18) to extract robust speech representations.
2. **CLeaD (Contrastive Alignment):** A dual-head architecture using Supervised Contrastive Loss (SupCon) to align same-class representations across English and Mandarin domains into a shared manifold.
3. **Non-Linear Classifier (SVM-RBF):** Radial Basis Function SVM is applied to standardized segment embeddings to capture non-linear decision boundaries.
4. **Sequence Modeling (Bi-GRU):** Groups segment embeddings per speaker and feeds them to a bidirectional GRU with self-attention pooling to capture temporal trajectories.

## Datasets
* **E-DAIC:** English corpus used for baseline training and evaluation.
* **MODMA:** Mandarin corpus used to validate zero-shot cross-lingual alignment.

---

## How to Run the Pipeline

### 1. Preprocessing
To segment the audio datasets into 10-second sliding windows and build mixed domain tables:
```bash
# 1. Segment and split EDAIC
python3 code/preprocessing/segment_edaic_sliding.py
python3 code/preprocessing/split_metadata.py --input_csv utterance_table_edaic_segmented.csv

# 2. Segment and split MODMA
python3 code/preprocessing/segment_modma_sliding.py
python3 code/preprocessing/split_metadata.py --input_csv utterance_table_modma_segmented.csv

# 3. Combine them to create the MIX dataset metadata
python3 code/preprocessing/build_mixed_metadata.py
```

### 2. Feature Extraction
Extract pooling features for a specific model variant:
```bash
# Extract WavLM Base-Plus features (Layers 6, 7, 8, 9)
python3 extract_ablation_features.py --model base-plus --device auto

# Extract WavLM Large features (Layers 12, 14, 16, 18)
python3 extract_ablation_features.py --model large --device auto
```

### 3. Downstream Ablation Study
Train and evaluate LR, SVM-Linear, SVM-RBF, Bi-GRU, and CLeaD classifiers:
```bash
# Run ablation study on Base-Plus features
python3 run_comprehensive_ablation.py --model base-plus

# Run ablation study on Large features
python3 run_comprehensive_ablation.py --model large
```

### 4. Generate Performance Comparison
To compile a side-by-side comparison report of both model variants:
```bash
python3 compare_results.py
```

---

## Model Comparison Summary

Below is a summary of the performance comparison between **WavLM Base-Plus** (L6–L9) and **WavLM Large** (L12–L18):

### 1. Zero-Shot Cross-Lingual Transfer

#### English $\rightarrow$ Mandarin (`EN -> ZH`)
*Trained on English (E-DAIC), tested on Mandarin (MODMA).*

| Layers (Base / Large) | Classifier Model | Base-Plus Acc | Large Acc | Base-Plus F1 | Large F1 | Base Speaker Acc | Large Speaker Acc |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **L6 / L12** | CLeaD (Contrastive) | 49.42% | 48.62% | 0.2744 | 0.0000 | 50.0% | 50.0% |
| **L7 / L14** | GRU (Sequential) | 50.00% | 50.00% | 0.6154 | 0.0000 | 50.0% | 50.0% |
| **L8 / L16** | GRU (Sequential) | 40.00% | 50.00% | 0.5714 | 0.0000 | 40.0% | 50.0% |

#### Mandarin $\rightarrow$ English (`ZH -> EN`)
*Trained on Mandarin (MODMA), tested on English (E-DAIC).*

| Layers (Base / Large) | Classifier Model | Base-Plus Acc | Large Acc | Base-Plus F1 | Large F1 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **L6 / L12** | LR (Linear Baseline) | 54.32% | **64.29%** | 0.4217 | 0.0000 |
| **L6 / L12** | CLeaD (Contrastive) | 52.55% | 35.71% | 0.3304 | **0.5263** |
| **L9 / L18** | SVM-RBF (Non-linear) | 50.73% | **64.29%** | 0.2466 | 0.0000 |

### 2. Monolingual Baselines (Upper Bounds)

#### English $\rightarrow$ English (`EN -> EN`)

| Layers (Base / Large) | Classifier Model | Base-Plus Acc | Large Acc | Base-Plus F1 | Large F1 | Base Speaker Acc | Large Speaker Acc |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **L6 / L12** | GRU (Sequential) | 47.83% | **82.61%** | 0.3333 | **0.6667** | 47.8% | **82.6%** |
| **L7 / L14** | GRU (Sequential) | 47.83% | **86.96%** | 0.3333 | **0.7273** | 47.8% | **87.0%** |
| **L7 / L14** | CLeaD (Contrastive) | 72.57% | **74.40%** | 0.5856 | **0.5923** | N/A | N/A |

### Detailed Performance Reports
* **Model Comparison Report:** See [output/model_comparison.md](file:///Users/anishapattanayak/Documents/SLT/Dep_Det/WavLM_Depression_Detection/output/model_comparison.md) for full detailed charts.
* **WavLM Large Details:** See [README_large.md](file:///Users/anishapattanayak/Documents/SLT/Dep_Det/WavLM_Depression_Detection/README_large.md) for Layer 12–18 results.
* **WavLM Base-Plus Details:** See [README_base-plus.md](file:///Users/anishapattanayak/Documents/SLT/Dep_Det/WavLM_Depression_Detection/README_base-plus.md) for Layer 6–9 results.