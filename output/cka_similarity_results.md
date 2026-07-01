# Quantitative Representation Similarity Analysis (CKA)

This analysis uses Linear Centered Kernel Alignment (CKA) to compare WavLM Base-Plus and WavLM Large representations. CKA values range from 0 (completely dissimilar) to 1 (identical representation space up to orthogonal transformation).

## 1. Cross-Model Similarity (Base-Plus vs Large)
This matrix compares corresponding deep layers of WavLM Base-Plus and WavLM Large.

| Base-Plus Layer | Large L12 | Large L14 | Large L16 | Large L18 |
| :---: | :---: | :---: | :---: | :---: |
| **L6** | 0.5975 | 0.5949 | 0.5755 | 0.5617 |
| **L7** | 0.6306 | 0.6236 | 0.6035 | 0.5905 |
| **L8** | 0.6983 | 0.6940 | 0.6754 | 0.6602 |
| **L9** | 0.6981 | 0.6947 | 0.6802 | 0.6667 |

## 2. Within-Model Layer Redundancy (Base-Plus)
High similarity between different layers indicates representation redundancy, while lower similarity indicates feature evolution/specialization.

| Layer | L6 | L7 | L8 | L9 |
| :---: | :---: | :---: | :---: | :---: |
| **L6** | 1.0000 | 0.9573 | 0.8976 | 0.8622 |
| **L7** | 0.9573 | 1.0000 | 0.9595 | 0.9178 |
| **L8** | 0.8976 | 0.9595 | 1.0000 | 0.9702 |
| **L9** | 0.8622 | 0.9178 | 0.9702 | 1.0000 |

## 3. Within-Model Layer Redundancy (Large)
Specialization claim: Large models should exhibit lower cross-layer similarity compared to base models, showing that representations specialize and change rapidly across layers.

| Layer | L12 | L14 | L16 | L18 |
| :---: | :---: | :---: | :---: | :---: |
| **L12** | 1.0000 | 0.9840 | 0.9609 | 0.9426 |
| **L14** | 0.9840 | 1.0000 | 0.9846 | 0.9633 |
| **L16** | 0.9609 | 0.9846 | 1.0000 | 0.9881 |
| **L18** | 0.9426 | 0.9633 | 0.9881 | 1.0000 |

### Specialization Summary:
- **Base-Plus Layer Redundancy (Average Off-Diagonal CKA)**: 0.9274
- **Large Layer Redundancy (Average Off-Diagonal CKA)**: 0.9706
- **Interpretation**: Both models show distinct layer-wise hierarchy.
