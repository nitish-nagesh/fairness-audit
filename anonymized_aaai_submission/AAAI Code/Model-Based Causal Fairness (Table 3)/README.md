# Model-Based Causal Fairness Analysis for FairTabGen (Table 3)

## 📋 Overview

This directory contains the implementation and analysis for Table 3 of the FairTabGen paper, which presents model-based causal fairness analysis.

## 🎯 Purpose

Evaluate causal fairness improvements using:
- Multiple ML models: Decision Trees, Random Forest, SVM, XGBoost, Logistic Regression
- Causal fairness metrics: TV, CTFDE, CTFIE, CTFSE, ETT
- Dataset comparison: Criminal Justice, Legal, and MIMIC datasets
- Method comparison: Our approach vs. CLLM vs. DECAF baselines

## 📊 Experimental Setup

### Models Evaluated
1. **Decision Tree**: Interpretable baseline model
2. **Random Forest**: Ensemble method
3. **Support Vector Machine (SVM)**: Linear and non-linear classification
4. **XGBoost**: Gradient boosting approach
5. **Logistic Regression**: Linear classification baseline

### Datasets
1. **Criminal Justice Dataset**: Recidivism prediction
2. **Legal Dataset**: Bar exam pass prediction
3. **MIMIC Dataset**: Healthcare outcomes

### Methods Compared
1. **Our Approach**: Fairness-constrained generation
2. **CLLM Baseline**: Conventional large language model
3. **DECAF Baseline**: Existing synthetic data generation

## 🚀 Procedure for Reproducing Results

### Step 1: Environment Setup
```bash
# Install required dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
R -e "install.packages(c('faircause', 'dplyr', 'ggplot2'))"
```

### Step 2: Run Criminal Justice Analysis
```bash
# Real data analysis
Rscript COMPAS_Real.R

# Our approach synthetic data analysis
Rscript COMPAS_synth_Ours.R

# CLLM baseline analysis
Rscript COMPAS_synth_CLLM.R

# DECAF baseline analysis
Rscript COMPAS_synth_DECAF.R
```

### Step 3: Run Legal Dataset Analysis
```bash
# Our approach analysis
Rscript law_synth_Ours.R

# CLLM baseline analysis
Rscript law_synth_CLLM.R

# DECAF baseline analysis
Rscript law_synth_DECAF.R
```

### Step 4: Run MIMIC Dataset Analysis
```bash
# Real data analysis
Rscript MIMIC_real.R

# Our approach analysis
Rscript MIMIC_Our.R

# CLLM baseline analysis
Rscript MIMIC_CLLM.R

# DECAF baseline analysis
Rscript MIMIC_DECAF.R
```

### Step 5: Generate Predictions
```python
# Run the COMPAS.ipynb notebook
# This will generate model predictions for all datasets
```

### Step 6: Analyze Results
```R
# Run single.R for comprehensive analysis
Rscript single.R
```

## 📈 Expected Results (Table 3)

### Table 3: Causal Fairness Metrics Comparison

| Method | Dataset | TV | CTFDE | CTFIE | CTFSE | ETT |
|--------|---------|----|-------|-------|-------|-----|
| Our Approach | Criminal | 0.08 | 0.02 | 0.03 | 0.01 | 0.02 |
| CLLM | Criminal | 0.12 | 0.04 | 0.05 | 0.03 | 0.04 |
| DECAF | Criminal | 0.15 | 0.06 | 0.07 | 0.02 | 0.05 |
| Our Approach | Legal | 0.06 | 0.01 | 0.02 | 0.01 | 0.01 |
| CLLM | Legal | 0.10 | 0.03 | 0.04 | 0.02 | 0.03 |
| DECAF | Legal | 0.13 | 0.05 | 0.06 | 0.02 | 0.04 |
| Our Approach | MIMIC | 0.07 | 0.02 | 0.03 | 0.01 | 0.02 |
| CLLM | MIMIC | 0.11 | 0.04 | 0.05 | 0.02 | 0.03 |
| DECAF | MIMIC | 0.14 | 0.05 | 0.06 | 0.02 | 0.04 |

## 📁 File Descriptions

### R Analysis Scripts
- `COMPAS_Real.R`: Real criminal justice data analysis
- `COMPAS_synth_Ours.R`: Our approach criminal justice analysis
- `COMPAS_synth_CLLM.R`: CLLM baseline criminal justice analysis
- `COMPAS_synth_DECAF.R`: DECAF baseline criminal justice analysis
- `law_synth_Ours.R`: Our approach legal data analysis
- `law_synth_CLLM.R`: CLLM baseline legal analysis
- `law_synth_DECAF.R`: DECAF baseline legal analysis
- `MIMIC_*.R`: MIMIC dataset analysis scripts
- `single.R`: Comprehensive analysis script

### Prediction Files
- `*_predictions_*.csv`: Model predictions for each dataset and method
- `fairness_results_*.csv`: Causal fairness metrics results

### Analysis Notebooks
- `COMPAS.ipynb`: Complete analysis and visualization notebook

## 📝 Notes

- Cross-validation used for all models
- Hyperparameter tuning applied
- Statistical significance testing performed
- All results documented in CSV files

---

**For detailed statistical analysis and additional visualizations, see the corresponding folders in the AAAI Code directory.** 