# FairTabGen: Unifying Counterfactual and Causal Fairness in Synthetic Tabular Data Generation

## 📋 Paper Information

**Title**: FairTabGen: Unifying Counterfactual and Causal Fairness in Synthetic Tabular Data Generation  
**Conference**: AAAI 2025  
**Authors**: [Anonymized for Review]  
**Repository**: Private repository for AAAI 2025 submission

## 🎯 Research Overview

This repository contains the complete implementation and experimental results for **FairTabGen**, a novel approach that unifies counterfactual and causal fairness in synthetic tabular data generation. Our method addresses the critical challenge of generating fair synthetic data while maintaining statistical properties and improving fairness across multiple domains.

## 🔬 Key Contributions

### 1. **Unified Fairness Framework**
- **Counterfactual Fairness**: What-if analysis for individual and group fairness
- **Causal Fairness**: Causal effect decomposition and bias mitigation
- **Synthetic Data Generation**: Fairness-constrained generation approach

### 2. **Multi-Domain Evaluation**
- **Criminal Justice Dataset**: Recidivism prediction fairness
- **Legal Dataset**: Bar exam pass prediction fairness
- **MIMIC Dataset**: Healthcare outcome fairness

### 3. **Comprehensive Experimental Design**
- **Multiple ML Models**: Decision Trees, Random Forest, SVM, XGBoost, Logistic Regression
- **Multiple Baselines**: CLLM and DECAF comparison
- **Multiple Metrics**: Quality, fairness, and causal metrics

## 📁 Repository Structure

### 📊 AAAI Code/
Complete implementation organized by figures and tables:

- **Data Generation (Table 1 and 2)**: Synthetic data generation experiments
- **Data Distribution (Fig 2)**: Distribution analysis and visualization
- **Model-Based Causal Fairness (Table 3)**: Causal fairness metrics table
- **Data Quality and Counterfactual Fairness (Fig 3 and 4)**: Quality and counterfactual analysis
- **Bias Mitigation Algorithms (Fig 5)**: Bias mitigation techniques
- **Real Data**: Dataset preprocessing and preparation

## 🚀 Quick Start

### Environment Setup
```bash
# Install Python dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter streamlit openai

# Install R dependencies
R -e "install.packages(c('faircause', 'dplyr', 'ggplot2'))"
```

### Reproducing Results
1. **Data Preprocessing**: Run notebooks in `Real Data/`
2. **Data Generation**: Execute scripts in `Data Generation (Table 1 and 2)/`
3. **Analysis**: Follow README files in each experiment folder
4. **Visualization**: Run notebooks for figures and tables

## 📊 Experimental Results

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

### Figure 3: Data Quality Analysis
- **Quality Metrics**: Completeness, consistency, accuracy assessment
- **Domain Comparison**: Quality across different datasets
- **Method Comparison**: FairTabGen vs. baseline quality performance
- **Statistical Validation**: Quality improvement significance

### Figure 4: Counterfactual Fairness Analysis
- **Individual Counterfactuals**: What-if scenarios for individual cases
- **Group Counterfactuals**: What-if scenarios for protected groups
- **Fairness Optimization**: Quality vs. fairness trade-off analysis
- **Robust Performance**: Across different counterfactual scenarios

### Figure 5: Bias Mitigation Algorithms
- **Pre-processing Algorithms**: Data-level bias mitigation techniques
- **In-processing Algorithms**: Model-level bias mitigation approaches
- **Post-processing Algorithms**: Prediction-level bias mitigation methods
- **Hybrid Approaches**: Combination of multiple bias mitigation techniques

## 🔧 Key Features

### FairTabGen Approach
- **Unified Fairness**: Combines counterfactual and causal fairness
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
  title={FairTabGen: Unifying Counterfactual and Causal Fairness in Synthetic Tabular Data Generation},
  author={[Authors anonymized for review]},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## 🤝 Acknowledgments

We thank the reviewers and the AAAI community for their valuable feedback.

---

**For detailed instructions for each experiment, see the README files in the AAAI Code directory.**
