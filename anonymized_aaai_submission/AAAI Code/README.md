# FairTabGen: AAAI Code Directory

## 📋 Overview

This directory contains the implementation and experimental results for the FairTabGen paper. The code is organized by figures and tables from the paper for easy reproduction of results.

## 🎯 Paper Information

**Title**: FairTabGen: Unifying Counterfactual and Causal Fairness in Synthetic Tabular Data Generation  
**Conference**: AAAI 2025  
**Authors**: [Anonymized for Review]  
**Repository**: Private repository for AAAI 2025 submission

## 📁 Directory Structure

### 📊 Data Generation (Table 1 and 2)
- **Purpose**: Synthetic data generation experiments
- **Content**: Generation scripts, prompt files, synthetic data files
- **Results**: Table 1 and Table 2 from the paper

### 📈 Data Distribution (Fig 2)
- **Purpose**: Data distribution analysis and visualization
- **Content**: Analysis notebook, real and synthetic data files
- **Results**: Figure 2 from the paper

### 🔬 Model-Based Causal Fairness (Table 3)
- **Purpose**: Causal fairness analysis across multiple ML models
- **Content**: R analysis scripts, prediction files, fairness results
- **Results**: Table 3 from the paper

### 📋 Data Quality and Counterfactual Fairness (Fig 3 and 4)
- **Purpose**: Data quality assessment and counterfactual fairness analysis
- **Content**: Analysis notebooks, quality metrics, counterfactual results
- **Results**: Figure 3 and Figure 4 from the paper

### 📊 Bias Mitigation Algorithms (Fig 5)
- **Purpose**: Bias mitigation algorithms and fairness decomposition
- **Content**: Bias mitigation notebooks, fairness decomposition plots
- **Results**: Figure 5 from the paper

### 📁 Real Data
- **Purpose**: Real dataset preprocessing and preparation
- **Content**: Original datasets and preprocessing code
- **Results**: Cleaned datasets for all experiments

## 🚀 Quick Start Guide

### Step 1: Environment Setup
```bash
# Install Python dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter

# Install R dependencies
R -e "install.packages(c('faircause', 'dplyr', 'ggplot2'))"
```

### Step 2: Data Preprocessing
```bash
# Navigate to Real Data folder
cd "Real Data"

# Run preprocessing notebooks
jupyter notebook COMPAS_Preprocess.ipynb
jupyter notebook Law_Preprocess.ipynb
jupyter notebook MIMIC_Preprocess.ipynb
```

### Step 3: Data Generation
```bash
# Navigate to Data Generation folder
cd "../Data Generation (Table 1 and 2)"

# Run generation scripts
python generate_synthetic_criminal_data.py
python generate_synthetic_legal_data.py
python generate_mimic_decaf_samples.py
```

### Step 4: Analysis and Visualization
```bash
# Run analysis notebooks in each folder
# Each folder contains detailed README files with specific instructions
```

## 📊 Experimental Organization

The results are organized by figures and tables from the paper:

- **Table 1 & 2**: Data generation experiments and results
- **Figure 2**: Data distribution analysis
- **Table 3**: Model-based causal fairness results
- **Figure 3 & 4**: Data quality and counterfactual fairness analysis
- **Figure 5**: Bias mitigation algorithms

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{fairtabgen2025,
  title={FairTabGen: Unifying Counterfactual and Causal Fairness in Synthetic Tabular Data Generation},
  author={[Authors anonymized for review]},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## 🤝 Acknowledgments

We thank the reviewers and the AAAI community for their valuable feedback.

---

**For detailed instructions for each experiment, see the README files in each subfolder.** 