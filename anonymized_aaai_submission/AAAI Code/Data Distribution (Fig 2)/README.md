# Data Distribution Analysis for FairTabGen (Figure 2)

## 📋 Overview

This directory contains the implementation and analysis for Figure 2 of the FairTabGen paper, which presents data distribution analysis across different datasets and synthetic data generation methods.

## 🎯 Purpose

Analyze and visualize data distributions to compare:
- Statistical similarity between real and synthetic data
- Distribution changes for fairness improvement
- Method comparison across different approaches

## 📊 Datasets Analyzed

### Real Datasets
1. **Criminal Justice Dataset** (`compas_cleaned.csv`): Recidivism prediction data
2. **Legal Dataset** (`bar_pass_prediction (processed version).csv`): Bar exam results
3. **MIMIC Dataset**: Healthcare outcomes (processed)

### Synthetic Datasets
1. **Our Approach Generated** (`generated_data_Our_prompt_*.csv`): Our method
2. **Baseline Generated**: Comparison with existing methods

## 🚀 Procedure for Reproducing Results

### Step 1: Environment Setup
```bash
# Install required dependencies
pip install pandas numpy matplotlib seaborn jupyter
```

### Step 2: Load and Prepare Data
```python
# In Data Analysis.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load real datasets
compas_real = pd.read_csv("compas_cleaned.csv")
legal_real = pd.read_csv("bar_pass_prediction (processed version).csv")

# Load synthetic datasets
compas_synth = pd.read_csv("generated_data_Our_prompt_COMPAS.csv")
legal_synth = pd.read_csv("generated_data_Our_prompts_Law.csv")
mimic_synth = pd.read_csv("generated_data_Our_prompts_MIMIC.csv")
```

### Step 3: Distribution Analysis
```python
# Run the Data Analysis.ipynb notebook
# This will generate Figure 2 visualizations
```

### Step 4: Generate Figure 2
The notebook will create:
- Distribution comparisons between real and synthetic data
- Fairness metrics before and after our approach
- Statistical tests for distribution similarity
- Multi-panel figure showing distributions

## 📈 Expected Results (Figure 2)

### Panel A: Criminal Justice Dataset
- Real data distribution vs. synthetic data distribution
- Distribution similarity assessment
- Fairness improvement visualization

### Panel B: Legal Dataset
- Legal domain-specific distribution analysis
- Bar exam data distribution comparison
- Method comparison in legal domain

### Panel C: MIMIC Dataset
- Healthcare-specific distribution analysis
- Healthcare outcome distribution comparison
- Method comparison in healthcare domain

### Panel D: Overall Comparison
- Cross-dataset distribution comparison
- Statistical significance assessment
- Effect size visualization

## 📁 File Descriptions

### Data Files
- `compas_cleaned.csv`: Real criminal justice dataset
- `bar_pass_prediction (processed version).csv`: Real legal dataset
- `generated_data_Our_prompt_COMPAS.csv`: Our approach criminal justice data
- `generated_data_Our_prompts_Law.csv`: Our approach legal data
- `generated_data_Our_prompts_MIMIC.csv`: Our approach MIMIC data

### Analysis Files
- `Data Analysis.ipynb`: Complete analysis and visualization notebook

## 📝 Notes

- All datasets normalized for fair comparison
- Missing values handled consistently
- Categorical variables encoded appropriately
- Statistical tests applied for significance
- High-resolution output for publication

---

**For detailed statistical analysis and additional visualizations, see the corresponding folders in the AAAI Code directory.** 