# Real Data Preprocessing for FairTabGen

## 📋 Overview

This directory contains the real datasets and preprocessing code used in the FairTabGen paper.

## 🚀 Procedure for Reproducing Results

### Environment Setup
```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### Run Preprocessing Notebooks
```python
# Run the COMPAS_Preprocess.ipynb notebook
# This will create compas_cleaned.csv from compas.arff

# Run the Law_Preprocess.ipynb notebook
# This will create bar_pass_prediction (processed version).csv

# Run the MIMIC_Preprocess.ipynb notebook
# This will prepare the MIMIC dataset for analysis
```

## 📁 File Descriptions

### Original Data Files
- `compas.arff`: Original COMPAS dataset in ARFF format
- `bar_pass_prediction.csv`: Original bar exam dataset
- `MIMIC_Preprocess.ipynb`: MIMIC dataset preprocessing

### Processed Data Files
- `compas_cleaned.csv`: Preprocessed criminal justice dataset
- `bar_pass_prediction (processed version).csv`: Processed legal dataset

### Preprocessing Notebooks
- `COMPAS_Preprocess.ipynb`: Criminal justice data preprocessing
- `Law_Preprocess.ipynb`: Legal data preprocessing
- `MIMIC_Preprocess.ipynb`: MIMIC data preprocessing 