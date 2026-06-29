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
|  | GRU | 56.52% | 0.3750 | 0.5686 |
|  | CLeaD | 75.40% | 0.5693 | 0.8172 |
|  | CLeaD w/o SupCon | 67.88% | 0.6577 | 0.8453 |
| | | | | |
| **EN -> ZH** | LR | 44.90% | 0.1213 | 0.2995 |
|  | SVM-Linear | 46.28% | 0.0599 | 0.2771 |
|  | SVM-RBF | 48.04% | 0.0296 | 0.3294 |
|  | GRU | 50.00% | 0.6154 | 0.6400 |
|  | CLeaD | 47.54% | 0.0470 | 0.3704 |
|  | CLeaD w/o SupCon | 48.29% | 0.2358 | 0.3775 |
| | | | | |
| **ZH -> EN** | LR | 50.81% | 0.5209 | 0.5355 |
|  | SVM-Linear | 52.52% | 0.4745 | 0.5274 |
|  | SVM-RBF | 50.45% | 0.2553 | 0.4909 |
|  | GRU | 65.22% | 0.0000 | 0.5196 |
|  | CLeaD | 50.93% | 0.4686 | 0.5183 |
|  | CLeaD w/o SupCon | 50.46% | 0.5166 | 0.5675 |
| | | | | |
| **ZH -> ZH** | LR | 49.42% | 0.3984 | 0.4469 |
|  | SVM-Linear | 48.08% | 0.3819 | 0.4488 |
|  | SVM-RBF | 46.83% | 0.3332 | 0.4388 |
|  | GRU | 50.00% | 0.0000 | 0.3200 |
|  | CLeaD | 48.41% | 0.4237 | 0.4658 |
|  | CLeaD w/o SupCon | 46.53% | 0.4052 | 0.4589 |
| | | | | |
| **MIX -> EN** | LR | 68.74% | 0.5971 | 0.7257 |
|  | SVM-Linear | 70.65% | 0.6198 | 0.7545 |
|  | SVM-RBF | 71.13% | 0.5314 | 0.7542 |
|  | GRU | 34.78% | 0.4000 | 0.4804 |
|  | CLeaD | 68.81% | 0.5357 | 0.7375 |
|  | CLeaD w/o SupCon | 65.93% | 0.6361 | 0.8161 |
| | | | | |
| **MIX -> ZH** | LR | 41.23% | 0.2990 | 0.3517 |
|  | SVM-Linear | 41.40% | 0.2853 | 0.3584 |
|  | SVM-RBF | 44.36% | 0.2483 | 0.4112 |
|  | GRU | 50.00% | 0.6667 | 0.3600 |
|  | CLeaD | 43.98% | 0.2098 | 0.3779 |
|  | CLeaD w/o SupCon | 45.57% | 0.3942 | 0.4222 |
| | | | | |

### 2. Speaker-Level Majority Vote Metrics (MODMA Test Set)
| Configuration | Model | MDD Correct | HC Correct | Speaker Acc |
| :--- | :--- | :---: | :---: | :---: |
| **ZH -> ZH** | LR | 2/5 | 4/5 | 60.00% |
|  | GRU | 0/5 | 5/5 | 50.00% |
|  | CLeaD | 3/5 | 4/5 | 70.00% |
|  | CLeaD w/o SupCon | 3/5 | 4/5 | 70.00% |
| | | | | |
| **MIX -> ZH** | LR | 1/5 | 4/5 | 50.00% |
|  | GRU | 5/5 | 0/5 | 50.00% |
|  | CLeaD | 0/5 | 5/5 | 50.00% |
|  | CLeaD w/o SupCon | 3/5 | 4/5 | 70.00% |
| | | | | |

### 3. WavLM Layer Ablation Study (MIX -> ZH Transfer)
| WavLM Layer | Model | Segment Accuracy | Segment F1 | Segment AUC | Speaker Vote (MDD/HC) |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Layer 12** | LR | 41.23% | 0.2990 | 0.3517 | 1/5 MDD, 4/5 HC |
|  | GRU | 50.00% | 0.6667 | 0.3600 | 5/5 MDD, 0/5 HC |
|  | CLeaD | 43.98% | 0.2098 | 0.3779 | 0/5 MDD, 5/5 HC |
|  | CLeaD w/o SupCon | 45.57% | 0.3942 | 0.4222 | 3/5 MDD, 4/5 HC |
| | | | | | |
| **Layer 14** | LR | 44.15% | 0.3182 | 0.3956 | 1/5 MDD, 4/5 HC |
|  | GRU | 50.00% | 0.6667 | 0.3200 | 5/5 MDD, 0/5 HC |
|  | CLeaD | 47.41% | 0.4576 | 0.4570 | 3/5 MDD, 4/5 HC |
|  | CLeaD w/o SupCon | 45.45% | 0.3061 | 0.4461 | 0/5 MDD, 4/5 HC |
| | | | | | |
| **Layer 16** | LR | 44.74% | 0.3415 | 0.4048 | 1/5 MDD, 4/5 HC |
|  | GRU | 50.00% | 0.6667 | 0.4400 | 5/5 MDD, 0/5 HC |
|  | CLeaD | 46.12% | 0.2456 | 0.4516 | 0/5 MDD, 5/5 HC |
|  | CLeaD w/o SupCon | 48.08% | 0.4920 | 0.4521 | 3/5 MDD, 4/5 HC |
| | | | | | |
| **Layer 18** | LR | 45.32% | 0.3432 | 0.4144 | 1/5 MDD, 4/5 HC |
|  | GRU | 50.00% | 0.6667 | 0.4000 | 5/5 MDD, 0/5 HC |
|  | CLeaD | 49.50% | 0.5127 | 0.4666 | 3/5 MDD, 4/5 HC |
|  | CLeaD w/o SupCon | 47.16% | 0.2384 | 0.4666 | 0/5 MDD, 5/5 HC |
| | | | | | |