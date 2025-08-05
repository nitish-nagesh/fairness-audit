# Data Quality and Counterfactual Fairness Analysis for FairTabGen (Figure 3 and Figure 4)

## 📋 Overview

This directory contains the implementation and analysis for Figure 3 and Figure 4 of the FairTabGen paper, which presents data quality and counterfactual fairness analysis.

## 🚀 Procedure for Reproducing Results

### Environment Setup
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
R -e "install.packages(c('faircause', 'dplyr', 'ggplot2'))"
```

### Run Analysis Notebooks
```python
# Run the COMPAS.ipynb notebook

# Run the LAW.ipynb notebook

# Run the MIMIC.ipynb notebook

```

## 📁 File Descriptions

### Analysis Notebooks
- `COMPAS.ipynb`: Criminal justice data quality and counterfactual analysis
- `LAW.ipynb`: Legal dataset quality and counterfactual analysis
- `MIMIC.ipynb`: MIMIC dataset quality and counterfactual analysis

### Data Files
- `compas_cleaned.csv`: Real criminal justice dataset
- `bar_pass_prediction (processed version).csv`: Real legal dataset
- `compas_synthetic_data_1000_200_epochs.csv`: Our approach criminal justice data
- `synthetic_law_data_decaf.csv`: Our approach legal data
- `mimic_synthetic_data_*.csv`: Our approach MIMIC data variants
- `generated_data_Our_prompt_*.csv`: Our approach generated data
- `generated_data_CLLM_prompt_*.csv`: CLLM baseline data

### Results Files
- `fairness_results_*.csv`: Fairness metrics for each dataset and method 
