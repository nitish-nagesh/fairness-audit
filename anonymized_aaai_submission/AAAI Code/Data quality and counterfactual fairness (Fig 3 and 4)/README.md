# Data Quality and Counterfactual Fairness Analysis for FairTabGen (Figure 3 and Figure 4)

## 📋 Overview

This directory contains the implementation and analysis for Figure 3 and Figure 4 of the FairTabGen paper, which presents data quality and counterfactual fairness analysis.

## 🎯 Purpose

Evaluate data quality and counterfactual fairness using:
- Data quality metrics: Completeness, consistency, accuracy, timeliness
- Counterfactual fairness: What-if analysis for fairness improvement
- Dataset comparison: Criminal Justice, Legal, and MIMIC datasets
- Method comparison: Our approach vs. CLLM vs. DECAF baselines

## 📊 Experimental Setup

### Quality Metrics Evaluated
1. **Completeness**: Percentage of non-missing values
2. **Consistency**: Data format and value consistency
3. **Accuracy**: Statistical similarity to real data
4. **Timeliness**: Data freshness and relevance
5. **Validity**: Data range and domain constraints

### Counterfactual Analysis
1. **Individual Counterfactuals**: What-if scenarios for individual cases
2. **Group Counterfactuals**: What-if scenarios for protected groups
3. **Fairness Counterfactuals**: What-if scenarios for fairness improvement

### Datasets
1. **Criminal Justice Dataset**: Recidivism prediction
2. **Legal Dataset**: Bar exam pass prediction
3. **MIMIC Dataset**: Healthcare outcomes

## 🚀 Procedure for Reproducing Results

### Step 1: Environment Setup
```bash
# Install required dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
R -e "install.packages(c('faircause', 'dplyr', 'ggplot2'))"
```

### Step 2: Run Criminal Justice Analysis
```python
# Run the COMPAS.ipynb notebook
# This will generate Figure 3A: Criminal Justice data quality and counterfactual analysis
```

### Step 3: Run Legal Dataset Analysis
```python
# Run the LAW.ipynb notebook
# This will generate Figure 3B: Legal data quality and counterfactual analysis
```

### Step 4: Run MIMIC Dataset Analysis
```python
# Run the MIMIC.ipynb notebook
# This will generate Figure 3C: MIMIC data quality and counterfactual analysis
```

### Step 5: Generate Counterfactual Scenarios
```python
# In each notebook, run counterfactual analysis:
# 1. Generate individual counterfactuals
# 2. Analyze group-level counterfactuals
# 3. Evaluate fairness counterfactuals
```

## 📈 Expected Results (Figure 3 and Figure 4)

### Figure 3: Data Quality Analysis

**Panel A: Criminal Justice Dataset**
- Data quality metrics assessment
- Counterfactual analysis scenarios
- Fairness improvement visualization

**Panel B: Legal Dataset**
- Legal domain-specific quality measures
- Legal domain counterfactual scenarios
- Bar exam prediction fairness analysis

**Panel C: MIMIC Dataset**
- Healthcare-specific quality measures
- Healthcare counterfactual scenarios
- Healthcare outcome fairness analysis

### Figure 4: Counterfactual Fairness Analysis

**Panel A: Individual Counterfactuals**
- What-if scenarios for individual cases
- Individual-level fairness improvements
- Statistical significance assessment

**Panel B: Group Counterfactuals**
- Group-level counterfactual analysis
- Group-specific fairness measures
- Effect size visualization

**Panel C: Overall Comparison**
- Quality vs. fairness trade-off analysis
- Method comparison across baselines
- Statistical significance assessment

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

## 📝 Notes

- Automated quality checks implemented for all datasets
- Domain-specific validation rules applied
- Statistical significance testing performed
- All results documented in CSV files

---

**For detailed statistical analysis and additional visualizations, see the corresponding folders in the AAAI Code directory.** 