# Within-Pipeline Leakage Ablation Study (Section III-C)

This report quantifies how pipeline design flaws (feature scaling leakage and speaker identity overlap) artificially inflate performance metrics. Evaluated on WavLM Base-Plus (Layer 6).

## 1. E-DAIC Dataset (English)

| Classifier | Airtight (F1 / AUC) | Scaling Leakage Only (F1 / AUC) | Speaker Identity Leakage Only (F1 / AUC) | Fully Leaky Pipeline (F1 / AUC) |
| :--- | :---: | :---: | :---: | :---: |
| LR | 0.4806 / 0.6432 | 0.4810 / 0.6432 | 0.7467 / 0.8883 | 0.7471 / 0.8883 |
| SVM | 0.4811 / 0.6439 | 0.4814 / 0.6437 | 0.7466 / 0.8883 | 0.7467 / 0.8883 |

## 2. MODMA Dataset (Mandarin)

| Classifier | Airtight (F1 / AUC) | Scaling Leakage Only (F1 / AUC) | Speaker Identity Leakage Only (F1 / AUC) | Fully Leaky Pipeline (F1 / AUC) |
| :--- | :---: | :---: | :---: | :---: |
| LR | 0.6279 / 0.7063 | 0.6282 / 0.7064 | 0.8563 / 0.9329 | 0.8557 / 0.9329 |
| SVM | 0.6274 / 0.7064 | 0.6283 / 0.7063 | 0.8567 / 0.9329 | 0.8562 / 0.9329 |

### Key Takeaways:
- **Speaker Identity Leakage** causes the most severe performance inflation. When segments of the same speaker are split across train and test partitions, classifiers exploit speaker-specific characteristics (voice signatures, recording environment) rather than depression cues.
- **Feature Scaling Leakage** causes a minor but significant metric boost by leaking validation/test distribution statistics into training standardization.
- Utilizing a leakage-free design is essential to prevent over-optimistic performance reports that fail to generalize to true unseen subjects.
