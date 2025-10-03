# FairTabGen: AAAI Code Directory

## 📋 Overview

This directory contains the complete implementation for the FairTabGen paper. Each folder corresponds to a specific figure or table from the paper, providing all code, data, and scripts needed to reproduce the results.

## 📁 Directory Structure

### Data Generation (Table 1 and 2)
- **Purpose**: Synthetic data generation experiments
- **Content**: Generation scripts, prompt files, synthetic data files
- **Key Files**: `generate_synthetic_*.py`, `Prompt (Open AI-*)`, `*.csv` data files

### Data Distribution (Fig 2)
- **Purpose**: Data distribution analysis and visualization
- **Content**: Analysis notebook, real and synthetic data files
- **Key Files**: `Data Analysis.ipynb`, `compas_cleaned.csv`, `bar_pass_prediction.csv`

### Model-Based Causal Fairness (Table 3)
- **Purpose**: Causal fairness analysis across multiple ML models
- **Content**: R analysis scripts, prediction files, fairness results
- **Key Files**: `*.R` scripts, `*_predictions_*.csv`, `COMPAS.ipynb`

### Data Quality and Counterfactual Fairness (Fig 3 and 4)
- **Purpose**: Data quality assessment and counterfactual fairness analysis
- **Content**: Analysis notebooks, quality metrics, counterfactual results
- **Key Files**: `COMPAS.ipynb`, `LAW.ipynb`, `MIMIC.ipynb`

### Bias Mitigation Algorithms (Fig 5)
- **Purpose**: Bias mitigation algorithms and fairness decomposition
- **Content**: Bias mitigation notebooks, fairness decomposition plots
- **Key Files**: `*_Visualization.ipynb`, `*_Vis.ipynb`

### Real Data
- **Purpose**: Real dataset preprocessing and preparation
- **Content**: Original datasets and preprocessing code
- **Key Files**: `*_Preprocess.ipynb`, `compas.arff`, `bar_pass_prediction.csv`

## 🚀 Quick Start

1. **Install Dependencies**: See main README.md for setup instructions
2. **Navigate to Specific Folder**: Choose the experiment you want to reproduce
3. **Follow Folder README**: Each folder has specific instructions
4. **Run Scripts/Notebooks**: Execute the code as documented

## 📊 Reproducing Results

- **Table 1 & 2**: Run data generation scripts in respective folders
- **Figure 2**: Execute `Data Analysis.ipynb` in Data Distribution folder
- **Table 3**: Run R scripts and notebooks in Model-Based Causal Fairness folder
- **Figure 3 & 4**: Execute notebooks in Data Quality folder
- **Figure 5**: Run visualization notebooks in Bias Mitigation folder

All data files, scripts, and detailed instructions are provided in each folder.
