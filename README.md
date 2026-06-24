# Multi-Layer WavLM Ablation for Cross-Lingual Depression Detection: Contrastive Representation Alignment and Chronological Sequence Classifiers

This repository contains the codebase for cross-lingual zero-shot depression detection from speech, specifically targeting the transfer gap between Germanic (English) and Tonal (Mandarin) languages. We perform a multi-layer evaluation using both `microsoft/wavlm-base-plus` and `microsoft/wavlm-large` speech encoders.

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

## Results & Performance Scores

Here are the complete results obtained from our comprehensive ablation studies across both WavLM model variants.

### 1. Model Comparison Summary (Base-Plus vs. Large)

#### Zero-Shot Cross-Lingual Transfer

##### English $\rightarrow$ Mandarin (`EN -> ZH`)
*Trained on English (E-DAIC), tested on Mandarin (MODMA).*

| Layers (Base / Large) | Classifier Model | Base-Plus Acc | Large Acc | Base-Plus F1 | Large F1 | Base Speaker Acc | Large Speaker Acc |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **L6 / L12** | CLeaD (Contrastive) | 49.42% | 48.62% | 0.2744 | 0.0000 | 50.0% | 50.0% |
| **L7 / L14** | GRU (Sequential) | 50.00% | 50.00% | 0.6154 | 0.0000 | 50.0% | 50.0% |
| **L8 / L16** | GRU (Sequential) | 40.00% | 50.00% | 0.5714 | 0.0000 | 40.0% | 50.0% |

##### Mandarin $\rightarrow$ English (`ZH -> EN`)
*Trained on Mandarin (MODMA), tested on English (E-DAIC).*

| Layers (Base / Large) | Classifier Model | Base-Plus Acc | Large Acc | Base-Plus F1 | Large F1 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **L6 / L12** | LR (Linear Baseline) | 54.32% | **64.29%** | 0.4217 | 0.0000 |
| **L6 / L12** | CLeaD (Contrastive) | 52.55% | 35.71% | 0.3304 | **0.5263** |
| **L9 / L18** | SVM-RBF (Non-linear) | 50.73% | **64.29%** | 0.2466 | 0.0000 |

#### Monolingual Baselines (Upper Bounds)

##### English $\rightarrow$ English (`EN -> EN`)

| Layers (Base / Large) | Classifier Model | Base-Plus Acc | Large Acc | Base-Plus F1 | Large F1 | Base Speaker Acc | Large Speaker Acc |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **L6 / L12** | GRU (Sequential) | 47.83% | **82.61%** | 0.3333 | **0.6667** | 47.8% | **82.6%** |
| **L7 / L14** | GRU (Sequential) | 47.83% | **86.96%** | 0.3333 | **0.7273** | 47.8% | **87.0%** |
| **L7 / L14** | CLeaD (Contrastive) | 72.57% | **74.40%** | 0.5856 | **0.5923** | N/A | N/A |

---

### 2. Segment-Level Metrics (WavLM Base-Plus - Layer 6)
| Configuration | Model | Accuracy | F1 Score | ROC AUC |
| :--- | :--- | :---: | :---: | :---: |
| **EN -> EN** | LR | 73.79% | 0.6628 | 0.8043 |
|  | SVM-Linear | 72.85% | 0.6476 | 0.7900 |
|  | SVM-RBF | 72.47% | 0.5581 | 0.7583 |
|  | GRU | 47.83% | 0.3333 | 0.5980 |
|  | CLeaD | 72.42% | 0.5929 | 0.7537 |
| | | | | |
| **EN -> ZH** | LR | 49.58% | 0.2545 | 0.4807 |
|  | SVM-Linear | 49.50% | 0.1913 | 0.5139 |
|  | SVM-RBF | 48.91% | 0.2245 | 0.5385 |
|  | GRU | 30.00% | 0.4615 | 0.4000 |
|  | CLeaD | 49.42% | 0.2744 | 0.5278 |
| | | | | |
| **ZH -> EN** | LR | 54.32% | 0.4217 | 0.5357 |
|  | SVM-Linear | 54.30% | 0.4100 | 0.5343 |
|  | SVM-RBF | 54.21% | 0.2955 | 0.5193 |
|  | GRU | 69.57% | 0.0000 | 0.4020 |
|  | CLeaD | 52.55% | 0.3304 | 0.5109 |
| | | | | |
| **ZH -> ZH** | LR | 53.63% | 0.4411 | 0.5368 |
|  | SVM-Linear | 55.39% | 0.4660 | 0.5613 |
|  | SVM-RBF | 54.76% | 0.4494 | 0.5606 |
|  | GRU | 80.00% | 0.7500 | 0.8000 |
|  | CLeaD | 55.26% | 0.4637 | 0.5816 |
| | | | | |
| **MIX -> EN** | LR | 66.95% | 0.5453 | 0.7023 |
|  | SVM-Linear | 66.29% | 0.5375 | 0.7001 |
|  | SVM-RBF | 69.83% | 0.5139 | 0.7254 |
|  | GRU | 47.83% | 0.4000 | 0.6373 |
|  | CLeaD | 65.73% | 0.4981 | 0.6732 |
| | | | | |
| **MIX -> ZH** | LR | 50.38% | 0.4272 | 0.4809 |
|  | SVM-Linear | 51.63% | 0.4411 | 0.5009 |
|  | SVM-RBF | 54.64% | 0.4419 | 0.5668 |
|  | GRU | 70.00% | 0.5714 | 0.6400 |
|  | CLeaD | 55.43% | 0.5007 | 0.5746 |

---

### 3. Segment-Level Metrics (WavLM Large - Layer 12)
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

---

### 4. Speaker-Level Majority Vote Metrics (MODMA Test Set)

#### WavLM Base-Plus (Layer 6)
| Configuration | Model | MDD Correct | HC Correct | Speaker Acc |
| :--- | :--- | :---: | :---: | :---: |
| **ZH -> ZH** | LR | 1/5 | 5/5 | 60.00% |
|  | GRU | 3/5 | 5/5 | 80.00% |
|  | CLeaD | 2/5 | 5/5 | 70.00% |
| | | | | |
| **MIX -> ZH** | LR | 2/5 | 4/5 | 60.00% |
|  | GRU | 2/5 | 5/5 | 70.00% |
|  | CLeaD | 4/5 | 5/5 | 90.00% |

#### WavLM Large (Layer 12)
| Configuration | Model | MDD Correct | HC Correct | Speaker Acc |
| :--- | :--- | :---: | :---: | :---: |
| **ZH -> ZH** | LR | 0/5 | 5/5 | 50.00% |
|  | GRU | 4/5 | 1/5 | 50.00% |
|  | CLeaD | 5/5 | 0/5 | 50.00% |
| | | | | |
| **MIX -> ZH** | LR | 5/5 | 0/5 | 50.00% |
|  | GRU | 5/5 | 0/5 | 50.00% |
|  | CLeaD | 5/5 | 0/5 | 50.00% |

---

### 5. WavLM Layer Ablation Study (MIX -> ZH Transfer)

#### WavLM Base-Plus
| WavLM Layer | Model | Segment Accuracy | Segment F1 | Segment AUC | Speaker Vote (MDD/HC) |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Layer 6** | LR | 50.38% | 0.4272 | 0.4809 | 2/5 MDD, 4/5 HC |
|  | GRU | 70.00% | 0.5714 | 0.6400 | 2/5 MDD, 5/5 HC |
|  | CLeaD | 55.43% | 0.5007 | 0.5746 | 4/5 MDD, 5/5 HC |
| | | | | | |
| **Layer 7** | LR | 47.20% | 0.4352 | 0.4570 | 3/5 MDD, 4/5 HC |
|  | GRU | 80.00% | 0.7500 | 0.6000 | 3/5 MDD, 5/5 HC |
|  | CLeaD | 52.55% | 0.4915 | 0.5271 | 3/5 MDD, 5/5 HC |
| | | | | | |
| **Layer 8** | LR | 46.16% | 0.4339 | 0.4513 | 2/5 MDD, 3/5 HC |
|  | GRU | 80.00% | 0.7500 | 0.6000 | 3/5 MDD, 5/5 HC |
|  | CLeaD | 49.96% | 0.4418 | 0.5123 | 2/5 MDD, 5/5 HC |
| | | | | | |
| **Layer 9** | LR | 46.20% | 0.4415 | 0.4476 | 2/5 MDD, 3/5 HC |
|  | GRU | 50.00% | 0.0000 | 0.6000 | 0/5 MDD, 5/5 HC |
|  | CLeaD | 51.29% | 0.4661 | 0.5123 | 3/5 MDD, 5/5 HC |

#### WavLM Large
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

---

### Detailed Sub-Reports
For more details on each run, see:
* **WavLM Large details:** [README_large.md](file:///Users/anishapattanayak/Documents/SLT/Dep_Det/WavLM_Depression_Detection/README_large.md)
* **WavLM Base-Plus details:** [README_base-plus.md](file:///Users/anishapattanayak/Documents/SLT/Dep_Det/WavLM_Depression_Detection/README_base-plus.md)
* **Side-by-side comparison report:** [output/model_comparison.md](file:///Users/anishapattanayak/Documents/SLT/Dep_Det/WavLM_Depression_Detection/output/model_comparison.md)