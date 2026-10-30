# CardioFuse-4

### A Unified Multi-Modal Late-Fusion Architecture for Cardiovascular Disease Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CardioFuse-4 is a multi-modal late-fusion architecture for cardiovascular disease (CVD) prediction. It combines four specialist streams—lifestyle risk factors, clinical laboratory biomarkers, hospital electronic medical records (EMR), and wearable electrocardiogram (ECG) signals—into a unified prediction using a neural-network meta-classifier.
---

## Contents

- [Research Paper](#research-paper)
- [Architecture](#architecture)
- [Reported Benchmark Results](#reported-benchmark-results)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Repository Structure](#repository-structure)
- [References](#references)

## Research Paper

The research paper provides the detailed explanation of the CardioFuse-4 architecture, methodology, experimental setup, and reported results.

📄 **[Read the CardioFuse-4 Research Paper (PDF)](paper/CardioFuse_research_paper.pdf)**

## Architecture

CardioFuse-4 uses four modality-specific models. Their probability outputs are combined by a two-layer artificial neural network (ANN) meta-classifier with **7 hidden neurons**, which learns non-linear relationships between the specialist predictions.

| Component | Input / modality | Method |
|---|---|---|
| **Stream 1 - Lifestyle** | Lifestyle pivotal factors | XGBoost feature screening (top 15 features), Local Cascade Ensemble (XGB-LCE), SMOTE rebalancing (2:3 ratio), and Negative Correlation Learning |
| **Stream 2 - Clinical labs** | Laboratory biomarkers | Random Forest Gini screening (87 → 30 features), Pearson correlation, Framingham-scale filtering (30 → 26 features), and an improved 1D-CNN with Dynamic Self-Attention |
| **Stream 3 - Hospital EMR** | Demographic variables and clinical notes | Four categorical demographic variables combined with TF-IDF-vectorized unstructured notes, classified using a maximum-margin Linear Support Vector Machine (SVM) |
| **Stream 4 - Wearable ECG** | Continuous ECG; reported raw segment length: 6 seconds | In-sensor Reservoir-Computing Neural Network (RC-NN) with 40 reservoir neurons (`N = 40`, `Gᵢ = 0.7`, `G𝒇 = 0.1`) and a three-layer analog readout ANN |
| **Meta-layer — Late fusion** | Four specialist probability scores | Two-layer ANN with 7 hidden neurons for non-linear fusion across modalities |

### Fusion flow

```text
Lifestyle factors ───────► XGB-LCE ──────────┐
Clinical biomarkers ────► 1D-CNN + Attention ┤
Hospital EMR ────────────► Linear SVM ────────┼──► ANN meta-classifier ──► CVD prediction
Wearable ECG ────────────► RC-NN + readout ──┘       (2 layers, 7 hidden neurons)
```

## Reported Benchmark Results

The following values are the benchmark results reported for the individual streams and the proposed fusion model.

| Model | Primary modality | Accuracy | AUROC | F1-score |
|---|---|---:|---:|---:|
| Stream 1: XGB-LCE | Lifestyle factors | 94.10% | 0.971 | 92.30% |
| Stream 2: 1D-CNN + Self-Attention | Lab biomarkers | 98.72% | — | 97.46% |
| Stream 3: EMR Linear SVM | Demographics and text | 52.00% | 0.550 | 53.00% |
| Stream 4: In-Sensor RC-NN | Raw 6-second ECG signal | — | 0.930 | 92.00% |
| **CardioFuse-4 (proposed fusion)** | **All four modalities** | **98.91%** | **0.994** | **98.75%** |

`-` indicates that a value was not provided in the reported results.

## Getting Started

### 1. Clone the repository

Replace the placeholder below with your actual GitHub repository URL:

```bash
git clone https://github.com/<your-username>/cardiofuse-4.git
cd cardiofuse-4
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\\Scripts\\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python cardiofuse4_system.py
```

This is the execution entry point specified in the project documentation. Ensure that `cardiofuse4_system.py` exists in the repository root.

## Dependencies

Create a `requirements.txt` file in the repository root:

```text
numpy>=1.22.0
pandas>=1.4.0
scikit-learn>=1.0.0
xgboost>=1.6.0
torch>=1.10.0
```

The project badges identify Python 3.8+ and PyTorch 2.0+, while the dependency list records the minimum package versions supplied for the project. Check compatibility in your target environment.

## Repository Structure

A suggested repository layout is:

```text
cardiofuse-4/
├── paper/
│   └── CardioFuse-4_Research_Paper.pdf
├── src/
│   └── cardiofuse4_system.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

If your Python script is currently in the repository root rather than `src/`, either move it into `src/` and update the run command accordingly, or keep the actual structure reflected here.

## References

1. Lin, T., & Fan, M. (2024). “Cardiovascular Disease Risk Prediction Based on Deep Feature Extraction.” *2024 9th International Conference on Intelligent Informatics and Biomedical Sciences (ICIIBMS)*, pp. 210–214. IEEE.
2. Long, Y. (2024). “Using the XGB-LCE Model for Cardiovascular Disease Prediction from Individual Pivotal Factors.” *2024 4th International Conference on Consumer Electronics and Computer Engineering (ICCECE)*, pp. 399–402. IEEE.
3. Sadasivuni, S., Damodaran, V., Banerjee, I., & Sanyal, A. (2022). “Real-time prediction of cardiovascular diseases using reservoir-computing and fusion with electronic medical record.” *2022 IEEE 4th International Conference on Artificial Intelligence Circuits and Systems (AICAS)*, pp. 58–61. IEEE.

