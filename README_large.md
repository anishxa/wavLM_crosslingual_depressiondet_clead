# Multi-Layer WavLM Ablation for Cross-Lingual Depression Detection: Contrastive Representation Alignment and Chronological Sequence Classifiers

This repository contains the codebase for cross-lingual zero-shot depression detection from speech, specifically targeting the transfer gap between Germanic (English) and Tonal (Mandarin) languages.

## Pipeline Structure
```
                                AUDIO INPUT
                                     |
                                     v
                        10s sliding segment window
                                     |
                                     v
                  WavLM-Base-Plus Encoder (Frozen middle layers)
                                     |
                                     v
                         Segment Embeddings (768-d)
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
1. **Feature Extractor:** We use `microsoft/wavlm-large` (Layer 12) to extract robust, noise-augmented speech representations.
2. **CLeaD (Contrastive Alignment):** A dual-head architecture using Supervised Contrastive Loss (SupCon) to pull same-class representations together across English and Mandarin domains, mapping them to a shared clinical manifold.
3. **Non-Linear Classifier (SVM-RBF):** Radial Basis Function kernel SVM is applied to standardized segment embeddings to capture non-linear decision boundaries.
4. **Sequence Modeling (Bi-GRU):** Chronological sequence modeling groups segment embeddings per speaker and feeds them to a bidirectional GRU with self-attention pooling to capture temporal trajectories.

## Datasets
- **E-DAIC:** English corpus used for baseline training and evaluation.
- **MODMA:** Mandarin corpus used to validate zero-shot cross-lingual alignment.

## How to Run the Pipeline

### 1. Preprocessing
To segment the audio datasets into 10-second sliding windows and create the MIX dataset metadata:
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
To extract the mean pooling features across multiple layers:
```bash
python3 extract_ablation_features.py
```

### 3. Run Comprehensive Multi-Model Ablation Study
To train and evaluate LR, SVM-Linear, SVM-RBF, Bi-GRU, and CLeaD across all configurations and layers:
```bash
python3 run_comprehensive_ablation.py
```

## Results

Below are the segment-level and speaker-level evaluation scores obtained from the comprehensive ablation run:

### 1. Segment-Level Metrics (WavLM Layer 12)
| Configuration | Model | Accuracy | F1 Score | ROC AUC |
| :--- | :--- | :---: | :---: | :---: |
| **EN -> EN** | LR | 64.92% | 0.5006 | 0.6783 |
|  | SVM-Linear | 67.67% | 0.5584 | 0.7191 |
|  | SVM-RBF | 71.16% | 0.5142 | 0.7518 |
|  | GRU | 82.61% | 0.6667 | 0.7843 |
|  | CLeaD | 73.15% | 0.5426 | 0.7658 |
| | | | | |
| **EN -> ZH** | LR | 48.62% | 0.0000 | 0.5000 |
|  | SVM-Linear | 48.62% | 0.0000 | 0.5000 |
|  | SVM-RBF | 48.62% | 0.0000 | 0.5000 |
|  | GRU | 50.00% | 0.0000 | 0.5200 |
|  | CLeaD | 48.62% | 0.0000 | 0.5000 |
| | | | | |
| **ZH -> EN** | LR | 64.29% | 0.0000 | 0.5000 |
|  | SVM-Linear | 64.29% | 0.0000 | 0.5000 |
|  | SVM-RBF | 64.29% | 0.0000 | 0.5000 |
|  | GRU | 65.22% | 0.3333 | 0.5588 |
|  | CLeaD | 35.71% | 0.5263 | 0.5266 |
| | | | | |
| **ZH -> ZH** | LR | 48.62% | 0.0000 | 0.5000 |
|  | SVM-Linear | 48.62% | 0.0000 | 0.5000 |
|  | SVM-RBF | 48.62% | 0.0000 | 0.5000 |
|  | GRU | 50.00% | 0.6154 | 0.5200 |
|  | CLeaD | 51.38% | 0.6788 | 0.5000 |
| | | | | |
| **MIX -> EN** | LR | 61.51% | 0.4821 | 0.6324 |
|  | SVM-Linear | 62.00% | 0.4816 | 0.6317 |
|  | SVM-RBF | 69.32% | 0.5110 | 0.7176 |
|  | GRU | 52.17% | 0.4762 | 0.7059 |
|  | CLeaD | 68.25% | 0.5694 | 0.7233 |
| | | | | |
| **MIX -> ZH** | LR | 51.38% | 0.6788 | 0.5000 |
|  | SVM-Linear | 51.38% | 0.6788 | 0.5000 |
|  | SVM-RBF | 51.38% | 0.6788 | 0.5000 |
|  | GRU | 50.00% | 0.6667 | 0.5200 |
|  | CLeaD | 51.38% | 0.6788 | 0.5000 |
| | | | | |

### 2. Speaker-Level Majority Vote Metrics (MODMA Test Set)
| Configuration | Model | MDD Correct | HC Correct | Speaker Acc |
| :--- | :--- | :---: | :---: | :---: |
| **ZH -> ZH** | LR | 0/5 | 5/5 | 50.00% |
|  | GRU | 4/5 | 1/5 | 50.00% |
|  | CLeaD | 5/5 | 0/5 | 50.00% |
| | | | | |
| **MIX -> ZH** | LR | 5/5 | 0/5 | 50.00% |
|  | GRU | 5/5 | 0/5 | 50.00% |
|  | CLeaD | 5/5 | 0/5 | 50.00% |
| | | | | |

### 3. WavLM Layer Ablation Study (MIX -> ZH Transfer)
| WavLM Layer | Model | Segment Accuracy | Segment F1 | Segment AUC | Speaker Vote (MDD/HC) |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Layer 12** | LR | 51.38% | 0.6788 | 0.5000 | 5/5 MDD, 0/5 HC |
|  | GRU | 50.00% | 0.6667 | 0.5200 | 5/5 MDD, 0/5 HC |
|  | CLeaD | 51.38% | 0.6788 | 0.5000 | 5/5 MDD, 0/5 HC |
| | | | | | |
| **Layer 14** | LR | 51.38% | 0.6788 | 0.5000 | 5/5 MDD, 0/5 HC |
|  | GRU | 50.00% | 0.6667 | 0.5200 | 5/5 MDD, 0/5 HC |
|  | CLeaD | 51.38% | 0.6788 | 0.5000 | 5/5 MDD, 0/5 HC |
| | | | | | |
| **Layer 16** | LR | 51.38% | 0.6788 | 0.5000 | 5/5 MDD, 0/5 HC |
|  | GRU | 50.00% | 0.6667 | 0.5200 | 5/5 MDD, 0/5 HC |
|  | CLeaD | 51.38% | 0.6788 | 0.5000 | 5/5 MDD, 0/5 HC |
| | | | | | |
| **Layer 18** | LR | 51.38% | 0.6788 | 0.5000 | 5/5 MDD, 0/5 HC |
|  | GRU | 50.00% | 0.6667 | 0.5200 | 5/5 MDD, 0/5 HC |
|  | CLeaD | 51.38% | 0.6788 | 0.5000 | 5/5 MDD, 0/5 HC |
| | | | | | |