# 🧠 Fuzzy-DDI: A Robust Fuzzy Logic Query Model for Complex Drug–Drug Interaction Prediction

> *"Fuzzy-DDI decomposes DDI predictions into relational projections and logical operations on rough sets during inference. Fuzzy logic makes it more fault-tolerant than binary logic models."*

## 🔬 Overview

This repository contains an **implementation of Fuzzy-DDI**, a novel robust fuzzy logic-based query model for predicting complex drug–drug interactions (DDIs), including:
- Multi-dose DDI prediction
- Noisy and missing sample environments
- Temporal modeling with GRU/Transformer
- Uncertainty estimation via Monte Carlo Dropout
- Integration with DRKG knowledge embeddings
- Visualizations for attention weights and uncertainty distributions

The code builds upon the methodology described in the paper:
> **Cheng, J., Zhang, Y., Zhang, H., & Lu, M. (2025). "Fuzzy-DDI: A robust fuzzy logic query model for complex drug–drug interaction prediction." Artificial Intelligence in Medicine, 164, 103125.**

---

## ✅ Key Features

- ✅ **Temporal Modeling**: Uses GRU or Transformer to encode drug administration sequences.
- ✅ **Uncertainty Estimation**: Implements MC Dropout for confidence scoring.
- ✅ **Attention Visualization**: Highlights which drugs/drug features contribute most to predictions.
- ✅ **Multi-Task Output**: Predicts both presence of DDI and type of mechanism.
- ✅ **Robustness Testing**: Simulates noisy and missing data scenarios.
- ✅ **DRKG Embeddings**: Integrates external biomedical knowledge from DRKG.

---

## 📁 Dataset Descriptions

### 🗂️ Included Datasets

| Dataset | Description |
|--------|-------------|
| **DrugCombDB** | Multi-dose DDI dataset with cell-type and dose information. [Link](http://drugcomb.fimm.fi/) |
| **DrugBank** | FDA-approved drug database with pharmacological relationships. [Link](https://go.drugbank.com/) |
| **TWOSIDES** | Real-world adverse drug reactions from FDA FAERS. [Link](https://tatonetti-lab.gitbook.io/siderography/twosides) |
| **DRKG** | Drug Repurposing Knowledge Graph embeddings. [Link](https://github.com/gnn4dr/DRKG) |

### 📄 Dataset Format

Each dataset should be in CSV format like this:

```csv
drug_sequence,doses,cell_types,label
Fluorouracil;Veliparib,10;5,HepG2;MCF7,1
Loratadine;Ketoconazole,5;5,HEK293;A549,1
```

Each row represents a sequence of drug administrations along with their doses and target cell types.

---

## 📦 Folder Structure

```
fuzzyDDI/
│
├── data/
│   ├── drugcombdb.csv
│   ├── drugbank.csv
│   ├── twosides.csv
│   └── drkg_embeddings.csv
│
├── src/
│   ├── models/
│   │   ├── fuzzy_ddi.py         # Core Fuzzy-DDI model
│   │   ├── temporal_encoder.py  # GRU / Transformer encoder
│   │   └── mc_dropout.py        # Uncertainty layer
│   │
│   ├── datasets/
│   │   ├── ddi_dataset.py       # Custom PyTorch Dataset class
│   │   └── loader.py            # Data loading utilities
│   │
│   ├── utils/
│   │   ├── train_utils.py       # Training loop
│   │   ├── eval_utils.py        # Evaluation metrics
│   │   └── visualize.py         # Attention and uncertainty visualizations
│   │
│   └── config.yaml              # Hyperparameters and settings
│
├── notebooks/
│   └── fuzzyddi_experiments.ipynb   # Example usage and visualization
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### 📦 Required Packages

- `torch==2.0.1`
- `torch_geometric==2.3.1`
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- `rdkit` (for molecular processing)

---

## 🧪 Hyperparameters

Defined in `config.yaml`:

```yaml
embedding_dim: 512
hidden_dim: 1024
num_epochs: 100
batch_size: 32
learning_rate: 0.0001
dropout: 0.3
use_transformer: true
use_mc_dropout: true
device: "cuda" if available else "cpu"
```

You can modify these values based on your hardware and experiment needs.

---

## 🧱 Model Architecture Summary

### 🧠 Core Components

#### 1. **Fuzzy-DDI**
- Combines GCN + MLP for relation projection
- Uses fuzzy logic operations (`min`, `max`) over fuzzy sets
- Supports multi-hop reasoning and complex queries

#### 2. **Temporal Encoder**
- Optional GRU or Transformer module for sequential drug administration
- Encodes time-dependent effects of drug combinations

#### 3. **MC Dropout Layer**
- Adds dropout at inference time for uncertainty estimation

#### 4. **Attention Visualizer**
- Extracts attention weights from Transformer layers
- Visualizes key interactions between drug pairs

---

## 🛠️ Classes and Functions

### 📌 Models

- `class FuzzyDDI_Extended`: Main model combining GNN, fuzzy logic, and MC Dropout
- `class TemporalEncoder`: Handles sequential input using GRU or Transformer
- `class MCDropout`: Enables uncertainty estimation

### 📌 Datasets

- `class DDIDataset`: Loads and preprocesses DDI data
- `def load_drkg_embeddings()`: Integrates DRKG side info

### 📌 Utils

- `train()` function: Trains the model with early stopping
- `evaluate()`: Computes AUC, AUPR, Hits@K
- `predict_with_uncertainty()`: Runs multiple forward passes to estimate prediction confidence
- `visualize_attention()`: Plots attention maps from Transformer

---

## 🧪 Training Instructions

To train the model:

```bash
cd src/
python train.py --dataset drugcombdb --use_transformer --use_mc_dropout
```

### 💡 Options

| Flag | Description |
|------|-------------|
| `--dataset` | Choose from: `drugcombdb`, `drugbank`, `twosides` |
| `--use_transformer` | Use transformer instead of GRU |
| `--use_mc_dropout` | Enable uncertainty estimation |
| `--num_samples` | Number of MC samples for uncertainty |
| `--visualize` | Visualize attention maps after training |

---

## 📈 Evaluation Metrics

- **Hits@K** (1, 3, 10)
- **ROC-AUC**
- **AUPR**
- **Cohen’s Kappa**

All metrics are computed during training and saved to logs.

---

## 🧪 Inference and Uncertainty Estimation

Run inference with uncertainty:

```bash
cd src/
python predict.py --sample_index 10
```

This will:
- Load the best trained model
- Run inference on a test sample
- Print predicted probability and standard deviation

---

## 📊 Visualization Tools

Use the following scripts:

```bash
cd src/utils
python visualize.py --type attention_weights
python visualize.py --type uncertainty_distribution
```

These will generate plots showing:
- **Attention Weights**: Which drug interactions were most influential
- **Uncertainty Distribution**: Confidence intervals across multiple forward passes

---

## 📚 References

### Base Paper

> Cheng, J., Zhang, Y., Zhang, H., & Lu, M. (2025).  
> *Fuzzy-DDI: A robust fuzzy logic query model for complex drug–drug interaction prediction.*  
> Artificial Intelligence in Medicine, 164, 103125.  
> https://doi.org/10.1016/j.artmed.2025.103125

### Datasets

- DrugCombDB: http://drugcomb.fimm.fi/
- DrugBank: https://go.drugbank.com/
- TWOSIDES: https://tatonetti-lab.gitbook.io/siderography/twosides
- DRKG: https://github.com/gnn4dr/DRKG

---

## 🏷️ License

MIT License – see `LICENSE` for details.

---

## 📨 Contact

For questions or collaborations:

- Saeed Iqbal – saeed.iqbal@szu.edu.cn
  
---

## 🎯 Future Work

- Add support for **zero-shot learning**
- Integrate **pathway-level mechanistic interpretation**
- Build a **Streamlit dashboard** for clinical use
- Support **multi-modal inputs**: EHR + genomics + drug embeddings

---

## 🧾 Acknowledgments

This work was supported by the National Natural Science Foundation of China.

---
