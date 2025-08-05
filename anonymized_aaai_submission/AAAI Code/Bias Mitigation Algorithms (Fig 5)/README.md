# Bias Mitigation Algorithms for FairTabGen (Figure 5)

## 📋 Overview

This directory contains the implementation and analysis for Figure 5 of the FairTabGen paper, which presents bias mitigation algorithms and fairness decomposition analysis.

## 🎯 Purpose

Implement and evaluate bias mitigation algorithms to demonstrate:
- Algorithmic bias mitigation techniques
- Fairness decomposition analysis
- Distribution comparisons between real and synthetic data
- Method comparison across different approaches

## 📊 Experimental Setup

### Bias Mitigation Algorithms
1. **Pre-processing Algorithms**: Data-level bias mitigation
2. **In-processing Algorithms**: Model-level bias mitigation
3. **Post-processing Algorithms**: Prediction-level bias mitigation
4. **Hybrid Approaches**: Combination of multiple techniques

### Visualization Types
1. **Fairness Decomposition Plots**: Breakdown of fairness metrics
2. **Distribution Comparison Plots**: Real vs. synthetic data distributions
3. **Fairness Metrics Plots**: Before vs. after our approach
4. **Method Comparison Plots**: Our approach vs. baseline approaches

### Datasets Visualized
1. **Criminal Justice Dataset**: Recidivism prediction visualizations
2. **Legal Dataset**: Bar exam prediction visualizations
3. **MIMIC Dataset**: Healthcare outcome visualizations

### Visualization Methods
1. **Histograms**: Distribution comparisons
2. **Box Plots**: Fairness metric distributions
3. **Scatter Plots**: Correlation analysis
4. **Heatmaps**: Fairness decomposition matrices

## 🚀 Procedure for Reproducing Results

### Step 1: Environment Setup
```bash
# Install required dependencies
pip install pandas numpy matplotlib seaborn plotly jupyter
```

### Step 2: Run Criminal Justice Bias Mitigation
```python
# Run the COMPAS_Visualization.ipynb notebook
# This will generate Figure 5A: Criminal Justice bias mitigation algorithms
```

### Step 3: Run Legal Dataset Bias Mitigation
```python
# Run the Law_Vis.ipynb notebook
# This will generate Figure 5B: Legal dataset bias mitigation algorithms
```

### Step 4: Run MIMIC Dataset Bias Mitigation
```python
# Run the MIMIC_Vis.ipynb notebook
# This will generate Figure 5C: MIMIC dataset bias mitigation algorithms
```

### Step 5: Generate Combined Visualizations
```python
# In each notebook, create:
# 1. Bias mitigation algorithm comparisons
# 2. Distribution comparison plots
# 3. Method comparison plots
# 4. Statistical significance plots
```

## 📈 Expected Results (Figure 5)

### Panel A: Criminal Justice Dataset
- Bias mitigation algorithms comparison
- Distribution comparison between real and synthetic data
- Method comparison across different approaches
- Fairness improvement visualization

### Panel B: Legal Dataset
- Legal domain-specific bias mitigation algorithms
- Bar exam data distribution analysis
- Legal domain method comparison
- Bias mitigation improvement assessment

### Panel C: MIMIC Dataset
- Healthcare-specific bias mitigation algorithms
- Healthcare outcome distribution analysis
- Healthcare domain method comparison
- Healthcare-specific bias mitigation gains

### Panel D: Overall Comparison
- Cross-dataset bias mitigation comparison
- Statistical significance assessment
- Effect size visualization
- Method performance comparison

## 📁 File Descriptions

### Bias Mitigation Notebooks
- `COMPAS_Visualization.ipynb`: Criminal justice bias mitigation algorithms
- `Law_Vis.ipynb`: Legal dataset bias mitigation algorithms
- `MIMIC_Vis.ipynb`: MIMIC dataset bias mitigation algorithms

### Data Files
- `compas_cleaned.csv`: Real criminal justice dataset
- `bar_pass_prediction (processed version).csv`: Real legal dataset
- `compas_synthetic_data_1000_200_epochs.csv`: Our approach criminal justice data
- `synthetic_law_data_decaf.csv`: Our approach legal data
- `mimic_synthetic_data_3400_samples_DECAF.csv`: Our approach MIMIC data
- `generated_data_Our_prompt_*.csv`: Our approach generated data
- `generated_data_CLLM_prompt_*.csv`: CLLM baseline data

## 📝 Notes

- Algorithm selection based on domain requirements
- Parameter tuning applied for optimal performance
- Evaluation metrics appropriate for each domain
- Cross-validation used for robust results

---

**For detailed statistical analysis and additional visualizations, see the corresponding folders in the AAAI Code directory.** 