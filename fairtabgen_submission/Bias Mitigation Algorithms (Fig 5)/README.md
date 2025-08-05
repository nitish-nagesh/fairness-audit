# Bias Mitigation Algorithms for FairTabGen (Figure 5)

## 📋 Overview

This directory contains the implementation and analysis for Figure 5 of the FairTabGen paper, which presents bias mitigation algorithms and fairness decomposition analysis.

## 🚀 Procedure for Reproducing Results

### Environment Setup
```bash
pip install pandas numpy matplotlib seaborn plotly jupyter
```

### Run Visualization Notebooks
```python
# Run the COMPAS_Visualization.ipynb notebook

# Run the Law_Vis.ipynb notebook

# Run the MIMIC_Vis.ipynb notebook
```

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
