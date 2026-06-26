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
python3 code/preprocessing/split_metadata.py --input_csv data/utterance_table_edaic_segmented.csv

# 2. Segment and split MODMA
python3 code/preprocessing/segment_modma_sliding.py
python3 code/preprocessing/split_metadata.py --input_csv data/utterance_table_modma_segmented.csv

# 3. Combine them to create the MIX dataset metadata
python3 code/preprocessing/build_mixed_metadata.py
```

### 2. Feature Extraction
Extract pooling features for a specific model variant:
```bash
# Extract WavLM Base-Plus features (Layers 6, 7, 8, 9)
python3 code/feature_extraction/extract_ablation_features.py --model base-plus --device auto

# Extract WavLM Large features (Layers 12, 14, 16, 18)
python3 code/feature_extraction/extract_ablation_features.py --model large --device auto
```

### 3. Downstream Ablation Study
Train and evaluate LR, SVM-Linear, SVM-RBF, Bi-GRU, and CLeaD classifiers:
```bash
# Run ablation study on Base-Plus features
python3 code/classification/run_comprehensive_ablation.py --model base-plus

# Run ablation study on Large features
python3 code/classification/run_comprehensive_ablation.py --model large
```

### 4. Generate Performance Comparison
To compile a side-by-side comparison report of both model variants:
```bash
python3 code/classification/compare_results.py
```

---

## Results & Performance Scores

## 1. Zero-Shot Cross-Lingual Transfer
Zero-shot cross-lingual transfer tests the model's ability to generalize to a completely unseen language (e.g. English trained model evaluated on Mandarin segments and vice versa).

### Configuration: EN -> ZH
| Layers | Model | Base-Plus Acc | Large Acc | Base-Plus F1 | Large F1 | Base Speaker Acc | Large Speaker Acc |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| L6 / L12 | LR | 49.58% | 48.62% (-1.0%) | 0.2545 | 0.0000 (-0.254) | 50.0% | 50.0% |
| L6 / L12 | SVM-Linear | 49.50% | 48.62% (-0.9%) | 0.1913 | 0.0000 (-0.191) | 50.0% | 50.0% |
| L6 / L12 | SVM-RBF | 48.91% | 48.62% (-0.3%) | 0.2245 | 0.0000 (-0.224) | 50.0% | 50.0% |
| L6 / L12 | GRU | 30.00% | **50.00%** (+20.0%) | 0.4615 | 0.0000 (-0.462) | 30.0% | **50.0%** (+20.0%) |
| L6 / L12 | CLeaD | 49.42% | 48.62% (-0.8%) | 0.2744 | 0.0000 (-0.274) | 50.0% | 50.0% |
| L7 / L14 | LR | 46.78% | **48.62%** (+1.8%) | 0.3323 | 0.0000 (-0.332) | 50.0% | 50.0% |
| L7 / L14 | SVM-Linear | 46.70% | **48.62%** (+1.9%) | 0.2981 | 0.0000 (-0.298) | 50.0% | 50.0% |
| L7 / L14 | SVM-RBF | 45.74% | **48.62%** (+2.9%) | 0.1678 | 0.0000 (-0.168) | 50.0% | 50.0% |
| L7 / L14 | GRU | 50.00% | 50.00% | 0.6154 | 0.0000 (-0.615) | 50.0% | 50.0% |
| L7 / L14 | CLeaD | 49.33% | 48.62% (-0.7%) | 0.3585 | 0.0000 (-0.359) | 50.0% | 50.0% |
| L8 / L16 | LR | 49.00% | 48.62% (-0.4%) | 0.3994 | 0.0000 (-0.399) | 50.0% | 50.0% |
| L8 / L16 | SVM-Linear | 48.41% | **48.62%** (+0.2%) | 0.3816 | 0.0000 (-0.382) | 50.0% | 50.0% |
| L8 / L16 | SVM-RBF | 48.33% | **48.62%** (+0.3%) | 0.3200 | 0.0000 (-0.320) | 50.0% | 50.0% |
| L8 / L16 | GRU | 40.00% | **50.00%** (+10.0%) | 0.5714 | 0.0000 (-0.571) | 40.0% | **50.0%** (+10.0%) |
| L8 / L16 | CLeaD | 49.71% | 48.62% (-1.1%) | 0.4738 | 0.0000 (-0.474) | 60.0% | 50.0% (-10.0%) |
| L9 / L18 | LR | 47.24% | **48.62%** (+1.4%) | 0.2294 | 0.0000 (-0.229) | 50.0% | 50.0% |
| L9 / L18 | SVM-Linear | 47.24% | **48.62%** (+1.4%) | 0.2180 | 0.0000 (-0.218) | 50.0% | 50.0% |
| L9 / L18 | SVM-RBF | 50.50% | 48.62% (-1.9%) | 0.4165 | 0.0000 (-0.417) | 40.0% | **50.0%** (+10.0%) |
| L9 / L18 | GRU | 40.00% | **50.00%** (+10.0%) | 0.4000 | 0.0000 (-0.400) | 40.0% | **50.0%** (+10.0%) |
| L9 / L18 | CLeaD | 52.72% | 48.62% (-4.1%) | 0.4271 | 0.0000 (-0.427) | 60.0% | 50.0% (-10.0%) |

### Configuration: ZH -> EN
| Layers | Model | Base-Plus Acc | Large Acc | Base-Plus F1 | Large F1 | Base Speaker Acc | Large Speaker Acc |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| L6 / L12 | LR | 54.32% | **64.29%** (+10.0%) | 0.4217 | 0.0000 (-0.422) | N/A | N/A |
| L6 / L12 | SVM-Linear | 54.30% | **64.29%** (+10.0%) | 0.4100 | 0.0000 (-0.410) | N/A | N/A |
| L6 / L12 | SVM-RBF | 54.21% | **64.29%** (+10.1%) | 0.2955 | 0.0000 (-0.295) | N/A | N/A |
| L6 / L12 | GRU | 69.57% | 65.22% (-4.3%) | 0.0000 | **0.3333** (+0.333) | 69.6% | 65.2% (-4.3%) |
| L6 / L12 | CLeaD | 52.55% | 35.71% (-16.8%) | 0.3304 | **0.5263** (+0.196) | N/A | N/A |
| L7 / L14 | LR | 55.78% | **64.29%** (+8.5%) | 0.3687 | 0.0000 (-0.369) | N/A | N/A |
| L7 / L14 | SVM-Linear | 53.34% | **64.29%** (+10.9%) | 0.3540 | 0.0000 (-0.354) | N/A | N/A |
| L7 / L14 | SVM-RBF | 51.59% | **64.29%** (+12.7%) | 0.3206 | 0.0000 (-0.321) | N/A | N/A |
| L7 / L14 | GRU | 73.91% | 43.48% (-30.4%) | 0.0000 | **0.3158** (+0.316) | 73.9% | 43.5% (-30.4%) |
| L7 / L14 | CLeaD | 53.94% | **64.29%** (+10.3%) | 0.3155 | 0.0000 (-0.315) | N/A | N/A |
| L8 / L16 | LR | 54.67% | **64.29%** (+9.6%) | 0.3591 | 0.0000 (-0.359) | N/A | N/A |
| L8 / L16 | SVM-Linear | 53.76% | **64.29%** (+10.5%) | 0.3784 | 0.0000 (-0.378) | N/A | N/A |
| L8 / L16 | SVM-RBF | 50.81% | **64.29%** (+13.5%) | 0.3047 | 0.0000 (-0.305) | N/A | N/A |
| L8 / L16 | GRU | 43.48% | 26.09% (-17.4%) | 0.0000 | **0.4138** (+0.414) | 43.5% | 26.1% (-17.4%) |
| L8 / L16 | CLeaD | 52.00% | 35.76% (-16.2%) | 0.3212 | **0.5264** (+0.205) | N/A | N/A |
| L9 / L18 | LR | 54.42% | **64.29%** (+9.9%) | 0.2645 | 0.0000 (-0.264) | N/A | N/A |
| L9 / L18 | SVM-Linear | 54.75% | **64.29%** (+9.5%) | 0.2756 | 0.0000 (-0.276) | N/A | N/A |
| L9 / L18 | SVM-RBF | 50.73% | **64.29%** (+13.6%) | 0.2466 | 0.0000 (-0.247) | N/A | N/A |
| L9 / L18 | GRU | 43.48% | 26.09% (-17.4%) | 0.3158 | **0.4138** (+0.098) | 43.5% | 26.1% (-17.4%) |
| L9 / L18 | CLeaD | 54.52% | **64.27%** (+9.8%) | 0.2727 | 0.0000 (-0.273) | N/A | N/A |

## 2. Monolingual Baselines
Monolingual configurations train and test on the same language/domain to establish performance upper bounds.

### Configuration: EN -> EN
| Layers | Model | Base-Plus Acc | Large Acc | Base-Plus F1 | Large F1 | Base Speaker Acc | Large Speaker Acc |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| L6 / L12 | LR | 73.79% | 64.92% (-8.9%) | 0.6628 | 0.5006 (-0.162) | N/A | N/A |
| L6 / L12 | SVM-Linear | 72.85% | 67.67% (-5.2%) | 0.6476 | 0.5584 (-0.089) | N/A | N/A |
| L6 / L12 | SVM-RBF | 72.47% | 71.16% (-1.3%) | 0.5581 | 0.5142 (-0.044) | N/A | N/A |
| L6 / L12 | GRU | 47.83% | **82.61%** (+34.8%) | 0.3333 | **0.6667** (+0.333) | 47.8% | **82.6%** (+34.8%) |
| L6 / L12 | CLeaD | 72.42% | **73.15%** (+0.7%) | 0.5929 | 0.5426 (-0.050) | N/A | N/A |
| L7 / L14 | LR | 71.90% | 67.24% (-4.7%) | 0.6439 | 0.5617 (-0.082) | N/A | N/A |
| L7 / L14 | SVM-Linear | 71.66% | 68.21% (-3.4%) | 0.6408 | 0.5871 (-0.054) | N/A | N/A |
| L7 / L14 | SVM-RBF | 72.20% | **72.55%** (+0.3%) | 0.5658 | 0.5590 (-0.007) | N/A | N/A |
| L7 / L14 | GRU | 47.83% | **86.96%** (+39.1%) | 0.3333 | **0.7273** (+0.394) | 47.8% | **87.0%** (+39.1%) |
| L7 / L14 | CLeaD | 72.57% | **74.40%** (+1.8%) | 0.5856 | **0.5923** (+0.007) | N/A | N/A |
| L8 / L16 | LR | 68.23% | 65.65% (-2.6%) | 0.6083 | 0.5396 (-0.069) | N/A | N/A |
| L8 / L16 | SVM-Linear | 67.45% | 66.23% (-1.2%) | 0.5981 | 0.5563 (-0.042) | N/A | N/A |
| L8 / L16 | SVM-RBF | 67.10% | 50.91% (-16.2%) | 0.4927 | **0.5750** (+0.082) | N/A | N/A |
| L8 / L16 | GRU | 82.61% | 69.57% (-13.0%) | 0.6000 | 0.5333 (-0.067) | 82.6% | 69.6% (-13.0%) |
| L8 / L16 | CLeaD | 65.81% | **70.58%** (+4.8%) | 0.5459 | 0.5247 (-0.021) | N/A | N/A |
| L9 / L18 | LR | 68.34% | 65.25% (-3.1%) | 0.6165 | 0.5428 (-0.074) | N/A | N/A |
| L9 / L18 | SVM-Linear | 67.58% | 64.62% (-3.0%) | 0.6048 | 0.5493 (-0.056) | N/A | N/A |
| L9 / L18 | SVM-RBF | 68.15% | 42.28% (-25.9%) | 0.5332 | **0.5462** (+0.013) | N/A | N/A |
| L9 / L18 | GRU | 82.61% | 78.26% (-4.3%) | 0.6000 | 0.5455 (-0.055) | 82.6% | 78.3% (-4.3%) |
| L9 / L18 | CLeaD | 65.43% | **70.30%** (+4.9%) | 0.5393 | **0.5583** (+0.019) | N/A | N/A |

### Configuration: ZH -> ZH
| Layers | Model | Base-Plus Acc | Large Acc | Base-Plus F1 | Large F1 | Base Speaker Acc | Large Speaker Acc |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| L6 / L12 | LR | 53.63% | 48.62% (-5.0%) | 0.4411 | 0.0000 (-0.441) | 60.0% | 50.0% (-10.0%) |
| L6 / L12 | SVM-Linear | 55.39% | 48.62% (-6.8%) | 0.4660 | 0.0000 (-0.466) | 70.0% | 50.0% (-20.0%) |
| L6 / L12 | SVM-RBF | 54.76% | 48.62% (-6.1%) | 0.4494 | 0.0000 (-0.449) | 70.0% | 50.0% (-20.0%) |
| L6 / L12 | GRU | 80.00% | 50.00% (-30.0%) | 0.7500 | 0.6154 (-0.135) | 80.0% | 50.0% (-30.0%) |
| L6 / L12 | CLeaD | 55.26% | 51.38% (-3.9%) | 0.4637 | **0.6788** (+0.215) | 70.0% | 50.0% (-20.0%) |
| L7 / L14 | LR | 52.21% | 48.62% (-3.6%) | 0.4521 | 0.0000 (-0.452) | 70.0% | 50.0% (-20.0%) |
| L7 / L14 | SVM-Linear | 53.84% | 48.62% (-5.2%) | 0.4554 | 0.0000 (-0.455) | 80.0% | 50.0% (-30.0%) |
| L7 / L14 | SVM-RBF | 52.72% | 48.62% (-4.1%) | 0.4271 | 0.0000 (-0.427) | 70.0% | 50.0% (-20.0%) |
| L7 / L14 | GRU | 70.00% | 60.00% (-10.0%) | 0.5714 | **0.7143** (+0.143) | 70.0% | 60.0% (-10.0%) |
| L7 / L14 | CLeaD | 55.05% | 51.38% (-3.7%) | 0.4516 | **0.6788** (+0.227) | 70.0% | 50.0% (-20.0%) |
| L8 / L16 | LR | 49.87% | 48.62% (-1.3%) | 0.4112 | 0.0000 (-0.411) | 70.0% | 50.0% (-20.0%) |
| L8 / L16 | SVM-Linear | 51.63% | 48.62% (-3.0%) | 0.4384 | 0.0000 (-0.438) | 70.0% | 50.0% (-20.0%) |
| L8 / L16 | SVM-RBF | 51.71% | 48.62% (-3.1%) | 0.4203 | 0.0000 (-0.420) | 70.0% | 50.0% (-20.0%) |
| L8 / L16 | GRU | 70.00% | 60.00% (-10.0%) | 0.5714 | **0.7143** (+0.143) | 70.0% | 60.0% (-10.0%) |
| L8 / L16 | CLeaD | 50.29% | **51.38%** (+1.1%) | 0.3433 | **0.6788** (+0.336) | 50.0% | 50.0% |
| L9 / L18 | LR | 52.01% | 48.62% (-3.4%) | 0.4573 | 0.0000 (-0.457) | 70.0% | 50.0% (-20.0%) |
| L9 / L18 | SVM-Linear | 52.42% | 48.62% (-3.8%) | 0.4665 | 0.0000 (-0.467) | 70.0% | 50.0% (-20.0%) |
| L9 / L18 | SVM-RBF | 51.04% | 48.62% (-2.4%) | 0.4344 | 0.0000 (-0.434) | 70.0% | 50.0% (-20.0%) |
| L9 / L18 | GRU | 60.00% | 50.00% (-10.0%) | 0.3333 | **0.6154** (+0.282) | 60.0% | 50.0% (-10.0%) |
| L9 / L18 | CLeaD | 52.30% | 51.38% (-0.9%) | 0.4324 | **0.6788** (+0.246) | 70.0% | 50.0% (-20.0%) |

## 3. Mixed-Domain Generalization
Mixed-domain models train on a pooled combination of English and Mandarin speech, then test on monolingual domains to leverage multi-lingual representations.

### Configuration: MIX -> EN
| Layers | Model | Base-Plus Acc | Large Acc | Base-Plus F1 | Large F1 | Base Speaker Acc | Large Speaker Acc |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| L6 / L12 | LR | 66.95% | 61.51% (-5.4%) | 0.5453 | 0.4821 (-0.063) | N/A | N/A |
| L6 / L12 | SVM-Linear | 66.29% | 62.00% (-4.3%) | 0.5375 | 0.4816 (-0.056) | N/A | N/A |
| L6 / L12 | SVM-RBF | 69.83% | 69.32% (-0.5%) | 0.5139 | 0.5110 (-0.003) | N/A | N/A |
| L6 / L12 | GRU | 47.83% | **52.17%** (+4.3%) | 0.4000 | **0.4762** (+0.076) | 47.8% | **52.2%** (+4.3%) |
| L6 / L12 | CLeaD | 65.73% | **68.25%** (+2.5%) | 0.4981 | **0.5694** (+0.071) | N/A | N/A |
| L7 / L14 | LR | 67.90% | **69.72%** (+1.8%) | 0.5646 | **0.5963** (+0.032) | N/A | N/A |
| L7 / L14 | SVM-Linear | 66.24% | **68.94%** (+2.7%) | 0.5354 | **0.5900** (+0.055) | N/A | N/A |
| L7 / L14 | SVM-RBF | 68.25% | **68.84%** (+0.6%) | 0.5005 | **0.5112** (+0.011) | N/A | N/A |
| L7 / L14 | GRU | 43.48% | **52.17%** (+8.7%) | 0.3810 | 0.3529 (-0.028) | 43.5% | **52.2%** (+8.7%) |
| L7 / L14 | CLeaD | 61.59% | **73.59%** (+12.0%) | 0.4664 | **0.6359** (+0.170) | N/A | N/A |
| L8 / L16 | LR | 66.92% | 64.55% (-2.4%) | 0.5643 | 0.5250 (-0.039) | N/A | N/A |
| L8 / L16 | SVM-Linear | 65.17% | **65.45%** (+0.3%) | 0.5396 | **0.5440** (+0.004) | N/A | N/A |
| L8 / L16 | SVM-RBF | 67.37% | 66.77% (-0.6%) | 0.5558 | 0.4874 (-0.068) | N/A | N/A |
| L8 / L16 | GRU | 30.43% | **39.13%** (+8.7%) | 0.3846 | 0.3636 (-0.021) | 30.4% | **39.1%** (+8.7%) |
| L8 / L16 | CLeaD | 60.08% | **70.05%** (+10.0%) | 0.4381 | **0.5794** (+0.141) | N/A | N/A |
| L9 / L18 | LR | 67.55% | 62.43% (-5.1%) | 0.5668 | 0.4780 (-0.089) | N/A | N/A |
| L9 / L18 | SVM-Linear | 65.88% | 61.19% (-4.7%) | 0.5419 | 0.4694 (-0.072) | N/A | N/A |
| L9 / L18 | SVM-RBF | 66.95% | 61.44% (-5.5%) | 0.5266 | 0.3965 (-0.130) | N/A | N/A |
| L9 / L18 | GRU | 39.13% | 34.78% (-4.3%) | 0.3636 | **0.4000** (+0.036) | 39.1% | 34.8% (-4.3%) |
| L9 / L18 | CLeaD | 58.63% | **64.34%** (+5.7%) | 0.4071 | **0.4965** (+0.089) | N/A | N/A |

### Configuration: MIX -> ZH
| Layers | Model | Base-Plus Acc | Large Acc | Base-Plus F1 | Large F1 | Base Speaker Acc | Large Speaker Acc |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| L6 / L12 | LR | 50.38% | **51.38%** (+1.0%) | 0.4272 | **0.6788** (+0.252) | 60.0% | 50.0% (-10.0%) |
| L6 / L12 | SVM-Linear | 51.63% | 51.38% (-0.3%) | 0.4411 | **0.6788** (+0.238) | 70.0% | 50.0% (-20.0%) |
| L6 / L12 | SVM-RBF | 54.64% | 51.38% (-3.3%) | 0.4419 | **0.6788** (+0.237) | 70.0% | 50.0% (-20.0%) |
| L6 / L12 | GRU | 70.00% | 50.00% (-20.0%) | 0.5714 | **0.6667** (+0.095) | 70.0% | 50.0% (-20.0%) |
| L6 / L12 | CLeaD | 55.43% | 51.38% (-4.1%) | 0.5007 | **0.6788** (+0.178) | 90.0% | 50.0% (-40.0%) |
| L7 / L14 | LR | 47.20% | **51.38%** (+4.2%) | 0.4352 | **0.6788** (+0.244) | 70.0% | 50.0% (-20.0%) |
| L7 / L14 | SVM-Linear | 48.87% | **51.38%** (+2.5%) | 0.4396 | **0.6788** (+0.239) | 80.0% | 50.0% (-30.0%) |
| L7 / L14 | SVM-RBF | 51.84% | 51.38% (-0.5%) | 0.4114 | **0.6788** (+0.267) | 70.0% | 50.0% (-20.0%) |
| L7 / L14 | GRU | 80.00% | 50.00% (-30.0%) | 0.7500 | 0.6667 (-0.083) | 80.0% | 50.0% (-30.0%) |
| L7 / L14 | CLeaD | 52.55% | 51.38% (-1.2%) | 0.4915 | **0.6788** (+0.187) | 80.0% | 50.0% (-30.0%) |
| L8 / L16 | LR | 46.16% | **51.38%** (+5.2%) | 0.4339 | **0.6788** (+0.245) | 50.0% | 50.0% |
| L8 / L16 | SVM-Linear | 47.70% | **51.38%** (+3.7%) | 0.4345 | **0.6788** (+0.244) | 50.0% | 50.0% |
| L8 / L16 | SVM-RBF | 46.28% | **51.38%** (+5.1%) | 0.4128 | **0.6788** (+0.266) | 60.0% | 50.0% (-10.0%) |
| L8 / L16 | GRU | 80.00% | 50.00% (-30.0%) | 0.7500 | 0.6667 (-0.083) | 80.0% | 50.0% (-30.0%) |
| L8 / L16 | CLeaD | 49.96% | **51.38%** (+1.4%) | 0.4418 | **0.6788** (+0.237) | 70.0% | 50.0% (-20.0%) |
| L9 / L18 | LR | 46.20% | **51.38%** (+5.2%) | 0.4415 | **0.6788** (+0.237) | 50.0% | 50.0% |
| L9 / L18 | SVM-Linear | 47.20% | **51.38%** (+4.2%) | 0.4485 | **0.6788** (+0.230) | 40.0% | **50.0%** (+10.0%) |
| L9 / L18 | SVM-RBF | 46.62% | **51.38%** (+4.8%) | 0.4089 | **0.6788** (+0.270) | 60.0% | 50.0% (-10.0%) |
| L9 / L18 | GRU | 50.00% | 50.00% | 0.0000 | **0.6667** (+0.667) | 50.0% | 50.0% |
| L9 / L18 | CLeaD | 51.29% | **51.38%** (+0.1%) | 0.4661 | **0.6788** (+0.213) | 80.0% | 50.0% (-30.0%) |

## 4. Key Findings & Insights

- **Max Accuracy Improvement:** The model variant `Large` showed the greatest accuracy gain in **EN -> EN** using **GRU** on **L7 / L14**, improving by **+39.1%** (from 47.8% to 87.0%).
- **Max F1 Score Improvement:** The greatest F1 score gain was in **MIX -> ZH** using **GRU** on **L9 / L18**, improving by **+0.667** (from 0.0000 to 0.6667).
- **Average Segment Accuracy by Classifier:**
  | Classifier | Base-Plus Avg Acc | Large Avg Acc | Gain |
  | :--- | :---: | :---: | :---: |
  | CLeaD | 56.6% | 57.1% | +0.5% |
  | GRU | 57.2% | 53.2% | -4.0% |
  | LR | 56.7% | 57.2% | +0.5% |
  | SVM-Linear | 56.7% | 57.3% | +0.7% |
  | SVM-RBF | 56.8% | 56.5% | -0.3% |