# Cross-Lingual Depression Detection (WavLM + CLeaD + Sequence Models)

This repository contains the codebase for cross-lingual zero-shot depression detection from speech, specifically targeting the transfer gap between Germanic (English) and Tonal (Mandarin) languages.

## Pipeline Structure
```
                                AUDIO INPUT
                                     |
                                     v
                        10s sliding segment window
                                     |
                                     v
                  WavLM-Large Encoder (Frozen middle layers)
                                     |
                                     v
                         Segment Embeddings (768-d)
                                     |
         +---------------------------+---------------------------+
         v (Static Segment Pooling)                              v (Temporal Sequence Modeling)
  +-----------------------------+                         +-----------------------------+
  | Mean / Max Segment Pooling  |                         |  Group Segments by Speaker  |
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
  | Projection (128) |              |                                    v
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
1. **Feature Extractor:** We use `microsoft/wavlm-base-plus` (Layer 6) to extract robust, noise-augmented speech representations.
2. **CLeaD (Contrastive Alignment):** A dual-head architecture using Supervised Contrastive Loss (SupCon) to pull same-class representations together across English and Mandarin domains, mapping them to a shared clinical manifold.
3. **Non-Linear Classifier (SVM-RBF):** Radial Basis Function kernel SVM is applied to standardized segment embeddings to capture non-linear decision boundaries.
4. **Sequence Modeling (Bi-GRU):** Chronological sequence modeling groups segment embeddings per speaker and feeds them to a bidirectional GRU with self-attention pooling to capture temporal trajectories.

## Datasets
- **E-DAIC:** English corpus used for baseline training and evaluation.
- **MODMA:** Mandarin corpus used to validate zero-shot cross-lingual alignment.

## How to Run the Pipeline

### 1. Preprocessing
To segment the audio datasets into 10-second sliding windows:
```bash
python3 code/preprocessing/segment_edaic_sliding.py
python3 code/preprocessing/split_metadata.py --input_csv utterance_table_edaic_segmented.csv
```

### 2. Feature Extraction
To extract the mean, max, and concatenated pooling features across multiple layers:
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

### 1. Segment-Level Metrics (WavLM Layer 6)
| Configuration | Model | Accuracy | F1 Score | ROC AUC |
| :--- | :--- | :---: | :---: | :---: |
| **EN -> EN** | LR | 73.79% | 0.6628 | 0.8043 |
|  | SVM-Linear | 72.85% | 0.6476 | 0.7900 |
|  | SVM-RBF | 72.47% | 0.5581 | 0.7583 |
|  | GRU | 52.17% | 0.3529 | 0.5980 |
|  | CLeaD | 72.43% | 0.6143 | 0.7712 |
| | | | | |
| **EN -> ZH** | LR | 49.58% | 0.2545 | 0.4807 |
|  | SVM-Linear | 49.50% | 0.1913 | 0.5139 |
|  | SVM-RBF | 48.91% | 0.2245 | 0.5385 |
|  | GRU | 30.00% | 0.3636 | 0.3600 |
|  | CLeaD | 51.42% | 0.3171 | 0.5625 |
| | | | | |
| **ZH -> EN** | LR | 54.32% | 0.4217 | 0.5357 |
|  | SVM-Linear | 54.30% | 0.4100 | 0.5343 |
|  | SVM-RBF | 54.21% | 0.2955 | 0.5193 |
|  | GRU | 73.91% | 0.0000 | 0.4118 |
|  | CLeaD | 54.27% | 0.3881 | 0.5325 |
| | | | | |
| **ZH -> ZH** | LR | 53.63% | 0.4411 | 0.5368 |
|  | SVM-Linear | 55.39% | 0.4660 | 0.5613 |
|  | SVM-RBF | 54.76% | 0.4494 | 0.5606 |
|  | GRU | 60.00% | 0.3333 | 0.8000 |
|  | CLeaD | 55.47% | 0.4702 | 0.5837 |
| | | | | |
| **MIX -> EN** | LR | 66.95% | 0.5453 | 0.7023 |
|  | SVM-Linear | 66.29% | 0.5375 | 0.7001 |
|  | SVM-RBF | 69.83% | 0.5139 | 0.7254 |
|  | GRU | 47.83% | 0.4000 | 0.6471 |
|  | CLeaD | 67.55% | 0.4776 | 0.6822 |
| | | | | |
| **MIX -> ZH** | LR | 50.38% | 0.4272 | 0.4809 |
|  | SVM-Linear | 51.63% | 0.4411 | 0.5009 |
|  | SVM-RBF | 54.64% | 0.4419 | 0.5668 |
|  | GRU | 70.00% | 0.5714 | 0.7200 |
|  | CLeaD | 56.31% | 0.4690 | 0.5830 |
| | | | | |

### 2. Speaker-Level Majority Vote Metrics (MODMA Test Set)
| Configuration | Model | MDD Correct | HC Correct | Speaker Acc |
| :--- | :--- | :---: | :---: | :---: |
| **ZH -> ZH** | LR | 1/5 | 5/5 | 60.00% |
|  | GRU | 1/5 | 5/5 | 60.00% |
|  | CLeaD | 2/5 | 5/5 | 70.00% |
| | | | | |
| **MIX -> ZH** | LR | 2/5 | 4/5 | 60.00% |
|  | GRU | 2/5 | 5/5 | 70.00% |
|  | CLeaD | 3/5 | 5/5 | 80.00% |
| | | | | |

### 3. WavLM Layer Ablation Study (MIX -> ZH Transfer)
| WavLM Layer | Model | Segment Accuracy | Segment F1 | Segment AUC | Speaker Vote (MDD/HC) |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Layer 6** | LR | 50.38% | 0.4272 | 0.4809 | 2/5 MDD, 4/5 HC |
|  | GRU | 70.00% | 0.5714 | 0.7200 | 2/5 MDD, 5/5 HC |
|  | CLeaD | 56.31% | 0.4690 | 0.5830 | 3/5 MDD, 5/5 HC |
| | | | | | |
| **Layer 7** | LR | 47.20% | 0.4352 | 0.4570 | 3/5 MDD, 4/5 HC |
|  | GRU | 80.00% | 0.7500 | 0.6000 | 3/5 MDD, 5/5 HC |
|  | CLeaD | 51.84% | 0.4323 | 0.5282 | 2/5 MDD, 5/5 HC |
| | | | | | |
| **Layer 8** | LR | 46.16% | 0.4339 | 0.4513 | 2/5 MDD, 3/5 HC |
|  | GRU | 70.00% | 0.6667 | 0.7200 | 3/5 MDD, 4/5 HC |
|  | CLeaD | 49.33% | 0.3883 | 0.5051 | 2/5 MDD, 5/5 HC |
| | | | | | |
| **Layer 9** | LR | 46.20% | 0.4415 | 0.4476 | 2/5 MDD, 3/5 HC |
|  | GRU | 60.00% | 0.5000 | 0.5200 | 2/5 MDD, 4/5 HC |
|  | CLeaD | 52.63% | 0.4155 | 0.5330 | 1/5 MDD, 5/5 HC |
| | | | | | |