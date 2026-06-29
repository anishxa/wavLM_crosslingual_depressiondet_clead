# MODMA Leave-One-Speaker-Out (LOSO) Evaluation Report

This report summarizes Leave-One-Speaker-Out cross-validation performance on the **MODMA** dataset (52 unique speakers) across different classification backbones.

## 1. WavLM Base-Plus (Layer 7)

| Model | Segment Accuracy | Segment F1 | Segment AUC | Speaker Accuracy | Speaker F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LR | 65.30% | 0.6352 | 0.7031 | 76.92% | 0.7143 |
| SVM-Linear | 65.23% | 0.6351 | 0.7035 | 80.77% | 0.7619 |
| SVM-RBF | 66.36% | 0.6433 | 0.7176 | 76.92% | 0.7273 |
| CLeaD | 62.15% | 0.6075 | 0.6539 | 65.38% | 0.6400 |
| CLeaD w/o SupCon | 64.32% | 0.6237 | 0.6898 | 67.31% | 0.6222 |

## 2. WavLM Large (Layer 14)

| Model | Segment Accuracy | Segment F1 | Segment AUC | Speaker Accuracy | Speaker F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LR | 65.48% | 0.6298 | 0.7065 | 75.00% | 0.6667 |
| SVM-Linear | 65.02% | 0.6235 | 0.7042 | 73.08% | 0.6500 |
| SVM-RBF | 66.30% | 0.6301 | 0.7187 | 73.08% | 0.6667 |
| CLeaD | 63.32% | 0.6578 | 0.6855 | 67.31% | 0.6909 |
| CLeaD w/o SupCon | 62.73% | 0.6501 | 0.6468 | 65.38% | 0.6667 |
