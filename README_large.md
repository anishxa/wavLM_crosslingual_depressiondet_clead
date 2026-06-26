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
python3 code/preprocessing/split_metadata.py --input_csv data/utterance_table_edaic_segmented.csv

# 2. Segment and split MODMA
python3 code/preprocessing/segment_modma_sliding.py
python3 code/preprocessing/split_metadata.py --input_csv data/utterance_table_modma_segmented.csv

# 3. Combine them to create the MIX dataset metadata
python3 code/preprocessing/build_mixed_metadata.py
```

### 2. Feature Extraction
To extract the mean pooling features across multiple layers:
```bash
python3 code/feature_extraction/extract_ablation_features.py
```

### 3. Run Comprehensive Multi-Model Ablation Study
To train and evaluate LR, SVM-Linear, SVM-RBF, Bi-GRU, and CLeaD across all configurations and layers:
```bash
python3 code/classification/run_comprehensive_ablation.py
```

## Results

Below are the segment-level and speaker-level evaluation scores obtained from the comprehensive ablation run:

### 1. Segment-Level Metrics (WavLM Layer 12)
| Configuration | Model | Accuracy | F1 Score | ROC AUC |
| :--- | :--- | :---: | :---: | :---: |
| **EN -> EN** | LR | 71.51% | 0.6441 | 0.7956 |
|  | SVM-Linear | 73.99% | 0.6791 | 0.8226 |
|  | SVM-RBF | 75.46% | 0.6020 | 0.8185 |
|  | GRU | 56.52% | 0.3750 | 0.6324 |
|  | CLeaD | 76.97% | 0.6215 | 0.8352 |
| | | | | |
| **EN -> ZH** | LR | 44.90% | 0.1213 | 0.2995 |
|  | SVM-Linear | 46.28% | 0.0599 | 0.2771 |
|  | SVM-RBF | 48.04% | 0.0296 | 0.3294 |
|  | GRU | 60.00% | 0.7143 | 0.6400 |
|  | CLeaD | 47.70% | 0.0280 | 0.3944 |
| | | | | |
| **ZH -> EN** | LR | 50.81% | 0.5209 | 0.5355 |
|  | SVM-Linear | 52.52% | 0.4745 | 0.5274 |
|  | SVM-RBF | 50.45% | 0.2553 | 0.4909 |
|  | GRU | 56.52% | 0.0000 | 0.5686 |
|  | CLeaD | 52.25% | 0.3879 | 0.4993 |
| | | | | |
| **ZH -> ZH** | LR | 49.42% | 0.3984 | 0.4469 |
|  | SVM-Linear | 48.08% | 0.3819 | 0.4488 |
|  | SVM-RBF | 46.83% | 0.3332 | 0.4388 |
|  | GRU | 60.00% | 0.3333 | 0.6400 |
|  | CLeaD | 50.17% | 0.4056 | 0.4828 |
| | | | | |
| **MIX -> EN** | LR | 68.74% | 0.5971 | 0.7257 |
|  | SVM-Linear | 70.65% | 0.6198 | 0.7545 |
|  | SVM-RBF | 71.13% | 0.5314 | 0.7542 |
|  | GRU | 26.09% | 0.4138 | 0.5098 |
|  | CLeaD | 69.42% | 0.4657 | 0.7132 |
| | | | | |
| **MIX -> ZH** | LR | 41.23% | 0.2990 | 0.3517 |
|  | SVM-Linear | 41.40% | 0.2853 | 0.3584 |
|  | SVM-RBF | 44.36% | 0.2483 | 0.4112 |
|  | GRU | 50.00% | 0.6667 | 0.5200 |
|  | CLeaD | 44.40% | 0.2101 | 0.4297 |
| | | | | |

### 2. Speaker-Level Majority Vote Metrics (MODMA Test Set)
| Configuration | Model | MDD Correct | HC Correct | Speaker Acc |
| :--- | :--- | :---: | :---: | :---: |
| **ZH -> ZH** | LR | 2/5 | 4/5 | 60.00% |
|  | GRU | 1/5 | 5/5 | 60.00% |
|  | CLeaD | 3/5 | 4/5 | 70.00% |
| | | | | |
| **MIX -> ZH** | LR | 1/5 | 4/5 | 50.00% |
|  | GRU | 5/5 | 0/5 | 50.00% |
|  | CLeaD | 0/5 | 4/5 | 40.00% |
| | | | | |

### 3. WavLM Layer Ablation Study (MIX -> ZH Transfer)
| WavLM Layer | Model | Segment Accuracy | Segment F1 | Segment AUC | Speaker Vote (MDD/HC) |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Layer 12** | LR | 41.23% | 0.2990 | 0.3517 | 1/5 MDD, 4/5 HC |
|  | GRU | 50.00% | 0.6667 | 0.5200 | 5/5 MDD, 0/5 HC |
|  | CLeaD | 44.40% | 0.2101 | 0.4297 | 0/5 MDD, 4/5 HC |
| | | | | | |
| **Layer 14** | LR | 44.15% | 0.3182 | 0.3956 | 1/5 MDD, 4/5 HC |
|  | GRU | 50.00% | 0.6667 | 0.5400 | 5/5 MDD, 0/5 HC |
|  | CLeaD | 44.90% | 0.2415 | 0.4481 | 0/5 MDD, 4/5 HC |
| | | | | | |
| **Layer 16** | LR | 44.74% | 0.3415 | 0.4048 | 1/5 MDD, 4/5 HC |
|  | GRU | 50.00% | 0.0000 | 0.4000 | 0/5 MDD, 5/5 HC |
|  | CLeaD | 47.79% | 0.3700 | 0.4532 | 3/5 MDD, 4/5 HC |
| | | | | | |
| **Layer 18** | LR | 45.32% | 0.3432 | 0.4144 | 1/5 MDD, 4/5 HC |
|  | GRU | 50.00% | 0.6667 | 0.4800 | 5/5 MDD, 0/5 HC |
|  | CLeaD | 48.87% | 0.3053 | 0.4840 | 0/5 MDD, 5/5 HC |
| | | | | | |