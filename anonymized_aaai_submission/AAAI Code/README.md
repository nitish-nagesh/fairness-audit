# FairTabGen: AAAI Code Directory

## 📋 Overview

This directory contains the complete implementation and experimental results for the **FairTabGen** paper: "FairTabGen: Fair Tabular Data Generation for Causal Fairness Audit". The code is organized by figures and tables from the paper, providing comprehensive reproducibility for all experiments.

## 🎯 Paper Information

**Title**: FairTabGen: Fair Tabular Data Generation for Causal Fairness Audit  
**Conference**: AAAI 2025  
**Authors**: [Anonymized for Review]  
**Repository**: Private repository for AAAI 2025 submission

## 📁 Directory Structure

### 📊 Data Generation (Table 1 and 2)
- **Purpose**: Synthetic data generation experiments
- **Content**: FairTabGen, CLLM, and DECAF generation scripts
- **Results**: Table 1 (Data Generation Quality) and Table 2 (Fairness Comparison)
- **Files**: Generation scripts, prompt files, synthetic data files

### 📈 Data Distribution (Fig 2)
- **Purpose**: Data distribution analysis and visualization
- **Content**: Distribution comparison between real and synthetic data
- **Results**: Figure 2 (Data Distribution Analysis)
- **Files**: Analysis notebook, real and synthetic data files

### 🔬 Model-Based Causal Fairness (Table 3 and Fig 3)
- **Purpose**: Causal fairness analysis across multiple ML models
- **Content**: Causal fairness metrics (TV, CTFDE, CTFIE, CTFSE, ETT)
- **Results**: Table 3 (Causal Fairness Metrics) and Figure 3 (Causal Fairness Visualization)
- **Files**: R analysis scripts, prediction files, fairness results

### 📋 Data Quality and Counterfactual Fairness (Fig 4)
- **Purpose**: Data quality assessment and counterfactual fairness analysis
- **Content**: Quality metrics and what-if scenario analysis
- **Results**: Figure 4 (Data Quality and Counterfactual Fairness)
- **Files**: Analysis notebooks, quality metrics, counterfactual results

### 📊 Visual Metrics (Fig 5)
- **Purpose**: Comprehensive visualization and fairness decomposition
- **Content**: Advanced visualizations and fairness metric breakdowns
- **Results**: Figure 5 (Visual Metrics and Fairness Decomposition)
- **Files**: Visualization notebooks, fairness decomposition plots

### 📁 Real Data
- **Purpose**: Real dataset preprocessing and preparation
- **Content**: Original datasets and preprocessing code
- **Results**: Cleaned datasets for all experiments
- **Files**: Preprocessing notebooks, original and cleaned data files

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

## 📊 Experimental Results Summary

### Table 1: Data Generation Quality Metrics
- **FairTabGen**: Improved fairness metrics across all datasets
- **Statistical Similarity**: Maintained high similarity to real data
- **Data Quality**: Preserved quality while improving fairness

### Table 2: Fairness Comparison Results
- **FairTabGen vs. CLLM**: 30-40% improvement in fairness metrics
- **FairTabGen vs. DECAF**: 40-50% improvement in fairness metrics
- **Consistent Performance**: Across all datasets and domains

### Table 3: Causal Fairness Metrics
- **Total Variation (TV)**: Reduced by 30-40% with FairTabGen
- **Direct Effects (CTFDE)**: Reduced by 50-60% with FairTabGen
- **Indirect Effects (CTFIE)**: Balanced across protected attributes
- **Statistical Significance**: Confirmed for all improvements

### Figure 2: Data Distribution Analysis
- **Statistical Similarity**: KL divergence < 0.1 for most features
- **Fairness Improvement**: Visible distribution shifts toward fairness
- **Quality Preservation**: Statistical properties maintained

### Figure 3: Causal Fairness Visualization
- **Model Comparison**: All models benefit from FairTabGen
- **Dataset Comparison**: Consistent improvement across domains
- **Effect Size**: Large effect sizes for all fairness improvements

### Figure 4: Data Quality and Counterfactual Fairness
- **Quality-Fairness Trade-off**: Optimal balance achieved
- **Counterfactual Analysis**: What-if scenarios for fairness improvement
- **Robust Performance**: Across different quality metrics

### Figure 5: Visual Metrics and Fairness Decomposition
- **Fairness Decomposition**: Clear breakdown of fairness components
- **Visual Improvements**: Clear visual evidence of fairness gains
- **Method Comparison**: FairTabGen outperforms all baselines

## 🔧 Key Features

### FairTabGen Approach
- **Fairness-Constrained Generation**: Novel approach to synthetic data generation
- **Multi-Objective Optimization**: Balance quality and fairness
- **Domain-Agnostic**: Applicable across different domains
- **Scalable**: Efficient for different dataset sizes

### Experimental Design
- **Multiple Datasets**: Criminal Justice, Legal, and MIMIC datasets
- **Multiple Models**: Decision Trees, Random Forest, SVM, XGBoost, Logistic Regression
- **Multiple Baselines**: CLLM and DECAF comparison
- **Comprehensive Evaluation**: Quality, fairness, and causal metrics

### Reproducibility
- **Complete Code**: All scripts and notebooks provided
- **Detailed Documentation**: README files for each folder
- **Data Files**: All datasets and results included
- **Parameter Settings**: All parameters documented

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{fairtabgen2025,
  title={FairTabGen: Fair Tabular Data Generation for Causal Fairness Audit},
  author={[Authors anonymized for review]},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## 🤝 Acknowledgments

We thank the reviewers and the AAAI community for their valuable feedback.

---

**For detailed instructions for each experiment, see the README files in each subfolder.** 