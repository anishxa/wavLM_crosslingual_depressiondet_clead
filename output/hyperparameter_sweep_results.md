# Hyperparameter Sweep Report for CLeaD Contrastive Alignment

This report summarizes the performance of CLeaD under different values of supervised contrastive loss temperature ($\tau$) and loss weighting weight ($\lambda$) on the **MIX -> ZH** (cross-lingual transfer) task.

## 1. WavLM Base-Plus (Layer 7)

| Temperature ($\tau$) | Loss Weight ($\lambda$) | Segment Acc | Segment F1 | Segment AUC | Speaker Vote (MDD/HC) | Speaker Acc |
| :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| 0.05 | 0.3 | 50.84% | 0.4652 | 0.5056 | 2/5 MDD, 4/5 HC | 60.00% |
| 0.05 | 0.5 | 47.45% | 0.3697 | 0.4788 | 1/5 MDD, 4/5 HC | 50.00% |
| 0.05 | 0.7 | 48.08% | 0.4762 | 0.4735 | 3/5 MDD, 4/5 HC | 70.00% |
| 0.1 | 0.3 | 51.59% | 0.4824 | 0.5159 | 3/5 MDD, 4/5 HC | 70.00% |
| 0.1 | 0.5 | 51.25% | 0.5215 | 0.5060 | 4/5 MDD, 4/5 HC | 80.00% |
| 0.1 | 0.7 | 50.25% | 0.4873 | 0.4962 | 3/5 MDD, 4/5 HC | 70.00% |
| 0.2 | 0.3 | 51.00% | 0.5416 | 0.5075 | 4/5 MDD, 3/5 HC | 70.00% |
| 0.2 | 0.5 | 50.67% | 0.4421 | 0.5136 | 2/5 MDD, 4/5 HC | 60.00% |
| 0.2 | 0.7 | 49.96% | 0.4713 | 0.4962 | 3/5 MDD, 4/5 HC | 70.00% |

## 2. WavLM Large (Layer 14)

| Temperature ($\tau$) | Loss Weight ($\lambda$) | Segment Acc | Segment F1 | Segment AUC | Speaker Vote (MDD/HC) | Speaker Acc |
| :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| 0.05 | 0.3 | 46.99% | 0.0335 | 0.4206 | 0/5 MDD, 5/5 HC | 50.00% |
| 0.05 | 0.5 | 47.49% | 0.0172 | 0.4324 | 0/5 MDD, 5/5 HC | 50.00% |
| 0.05 | 0.7 | 46.78% | 0.0304 | 0.4145 | 0/5 MDD, 5/5 HC | 50.00% |
| 0.1 | 0.3 | 44.78% | 0.0908 | 0.4189 | 0/5 MDD, 5/5 HC | 50.00% |
| 0.1 | 0.5 | 45.24% | 0.0915 | 0.4329 | 0/5 MDD, 5/5 HC | 50.00% |
| 0.1 | 0.7 | 47.66% | 0.0173 | 0.4112 | 0/5 MDD, 5/5 HC | 50.00% |
| 0.2 | 0.3 | 44.36% | 0.1155 | 0.4077 | 0/5 MDD, 5/5 HC | 50.00% |
| 0.2 | 0.5 | 43.90% | 0.1694 | 0.4295 | 0/5 MDD, 4/5 HC | 40.00% |
| 0.2 | 0.7 | 46.16% | 0.0459 | 0.4297 | 0/5 MDD, 5/5 HC | 50.00% |
