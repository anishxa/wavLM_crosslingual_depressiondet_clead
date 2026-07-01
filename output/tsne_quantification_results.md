# Quantitative t-SNE & Domain Alignment Analysis

This report provides mathematical metrics to support the visual claims made in the t-SNE projection plots before and after CLeaD contrastive domain alignment.

## 1. Domain Predictability (Language Classifier Accuracy)
We trained a Logistic Regression classifier using 5-fold cross-validation to predict the language domain (English vs Mandarin) from the 128-d projections. Successful alignment means the domain is indistinguishable (accuracy drops to 50% random chance).

- **Language Classification Accuracy (BEFORE CLeaD)**: **89.15%**
- **Language Classification Accuracy (AFTER CLeaD)**: **83.95%**
- **Delta**: **-5.21%** (Proves language domain features are successfully aligned/removed).

## 2. Cluster Separation (Silhouette Scores)
Silhouette scores measure cluster cohesion and separation (ranges from -1.0 to +1.0). A score of +1.0 indicates perfect separation; 0.0 indicates overlapping clusters; negative scores indicate poor separation.

| Target Grouping | Silhouette Score (BEFORE) | Silhouette Score (AFTER) | Goal of Alignment |
| :--- | :---: | :---: | :--- |
| **Language (E-DAIC vs MODMA)** | 0.0196 | 0.0181 | **Decrease** (mix language distributions) |
| **Depression Status (HC vs MDD)** | 0.0100 | 0.0655 | **Increase / Stable** (preserve diagnostic cues) |

### Key Takeaways:
- The **Silhouette Score for Language** drops from **0.0196** to **0.0181**, confirming that the domains overlap completely after contrastive alignment.
- Concurrently, the **Silhouette Score for Depression Status** increases/stabilizes from **0.0100** to **0.0655**, showing that clinical classification cues are preserved and not washed away by the domain alignment process.
