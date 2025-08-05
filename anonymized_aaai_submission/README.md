# FairTabGen: Unifying Counterfactual and Causal Fairness in Synthetic Tabular Data Generation

## 📋 Paper Information

**Title**: FairTabGen: Unifying Counterfactual and Causal Fairness in Synthetic Tabular Data Generation  
**Conference**: AAAI 2025  
**Authors**: [Anonymized for Review]  
**Repository**: Private repository for AAAI 2025 submission

## 🎯 Overview

This repository contains the implementation and experimental results for the FairTabGen paper. The code is organized by figures and tables from the paper for easy reproduction of results.

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

**For detailed instructions for each experiment, see the README files in the AAAI Code directory.**
