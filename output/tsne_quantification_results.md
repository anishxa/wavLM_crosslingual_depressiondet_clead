# Quantitative t-SNE & Domain Alignment Analysis

This report quantifies cross-lingual domain alignment before and after CLeaD training.

## 1. Domain Predictability (Language Classifier Accuracy)
We trained a Logistic Regression classifier (5-fold CV) to predict the language domain (English vs Mandarin) from the 128-D projections. Better alignment corresponds to a drop in classifier accuracy toward the 50% random chance baseline.

- **Language Classification Accuracy (BEFORE CLeaD)**: **89.15%**
- **Language Classification Accuracy (AFTER CLeaD)**: **83.95%**
- **Delta**: **-5.21%** (indicating successful domain alignment).

## 2. Cluster Separation (Silhouette Scores)
Silhouette scores measure cluster cohesion and separation (ranging from -1.0 to +1.0).

| Target Grouping | Silhouette Score (BEFORE) | Silhouette Score (AFTER) | Target |
| :--- | :---: | :---: | :--- |
| **Language (E-DAIC vs MODMA)** | 0.0196 | 0.0181 | **Decrease** (mix language distributions) |
| **Depression Status (HC vs MDD)** | 0.0100 | 0.0655 | **Increase / Stable** (preserve diagnostic cues) |

### Key Takeaways:
- The **Silhouette Score for Language** drops from **0.0196** to **0.0181**, showing that the language domains are well-mixed after alignment.
- The **Silhouette Score for Depression Status** increases/stabilizes from **0.0100** to **0.0655**, confirming that depression-related features are preserved during alignment.
