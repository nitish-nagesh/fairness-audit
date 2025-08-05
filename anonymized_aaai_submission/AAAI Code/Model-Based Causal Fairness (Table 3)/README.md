# Model-Based Causal Fairness Analysis for FairTabGen (Table 3)

## 📋 Overview

This directory contains the implementation and analysis for Table 3 of the FairTabGen paper, which presents model-based causal fairness analysis.

## 🚀 Procedure for Reproducing Results

### Environment Setup
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
R -e "install.packages(c('faircause', 'dplyr', 'ggplot2'))"
```

### Run R Analysis Scripts
```bash
Rscript COMPAS_Real.R
Rscript COMPAS_synth_Ours.R
Rscript COMPAS_synth_CLLM.R
Rscript COMPAS_synth_DECAF.R
Rscript law_synth_Ours.R
Rscript law_synth_CLLM.R
Rscript law_synth_DECAF.R
Rscript MIMIC_real.R
Rscript MIMIC_Our.R
Rscript MIMIC_CLLM.R
Rscript MIMIC_DECAF.R
Rscript single.R
```

### Generate Predictions
```python
# Run the COMPAS.ipynb notebook
```

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
