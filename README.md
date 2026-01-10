# Drug–Target Interaction Prediction (Davis Dataset)

This repository implements a machine learning pipeline to predict drug–target binding affinities
(IC50 / Kd) using ligand fingerprints and protein language model embeddings.

## Dataset
- Davis kinase inhibitor dataset (TDC)

## Features
- Ligands: MACCS fingerprints
- Proteins:
  - One-hot amino acid composition
  - ProtBERT embeddings
  - ESM-2 embeddings (CPU-safe)

## Models
- Random Forest
- MLP Regressor

## Metrics
- RMSE
- R²
- Matthews Correlation Coefficient (MCC)

## Installation

### Conda (recommended)
```bash
conda env create -f environment.yml
conda activate dti
python main.py

#pip
pip install -r requirements.txt
python main.py
