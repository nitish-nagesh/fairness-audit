# FairTabGen: Fair Tabular Data Generation for Causal Fairness Audit

## 📋 Paper Information

**Title**: FairTabGen: Fair Tabular Data Generation for Causal Fairness Audit  
**Conference**: AAAI 2025  
**Authors**: [Anonymized for Review]  
**Repository**: Private repository for AAAI 2025 submission

## 🎯 Overview

This repository contains the implementation and experimental results for our paper "FairTabGen: Fair Tabular Data Generation for Causal Fairness Audit". The work focuses on generating fair synthetic tabular data for causal fairness auditing, addressing the critical need for diverse and representative datasets in fairness research.

## 📁 Repository Structure

```
fairness-audit/
├── AAAI Code/                           # Main research code organized by figures and tables
│   ├── Data Generation (Table 1 and 2)/ # Synthetic data generation experiments
│   ├── Data Distribution (Fig 2)/       # Data distribution analysis
│   ├── Model-Based Causal Fairness (Table 3 and Fig 3)/ # Causal fairness metrics
│   ├── Data quality and counterfactual fairness (Fig 4)/ # Counterfactual analysis
│   ├── Visual metrics (Fig 5)/          # Visualization and metrics
│   └── Real Data/                       # Real dataset preprocessing
├── requirements.txt                      # Python dependencies
├── TEST_REPORT.md                       # Comprehensive testing report
└── README.md                            # This file
```

## 🔬 Research Contributions

### 1. **Fair Synthetic Data Generation**
- Novel approach to generating fair tabular data
- Integration of fairness constraints in data generation
- Support for multiple fairness metrics

### 2. **Causal Fairness Audit Framework**
- Comprehensive causal fairness analysis
- Multiple fairness decomposition methods
- Model-agnostic fairness evaluation

### 3. **Experimental Validation**
- Extensive experiments on multiple datasets
- Comparison with state-of-the-art methods
- Robust evaluation metrics

## 📊 Experimental Results

### Datasets Used
- **Criminal Justice Dataset**: Recidivism prediction
- **Legal Dataset**: Bar exam pass prediction  
- **MIMIC Dataset**: Healthcare outcomes

### Key Findings
- Improved fairness metrics across all datasets
- Better representation of minority groups
- Enhanced causal fairness understanding

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running Experiments
1. Navigate to specific experiment directories in `AAAI Code/`
2. Follow the README files in each directory
3. Execute the provided scripts

## 📈 Results Organization

The results are organized by figures and tables from the paper:

- **Table 1 & 2**: Data generation experiments and results
- **Figure 2**: Data distribution analysis
- **Table 3 & Figure 3**: Model-based causal fairness results
- **Figure 4**: Data quality and counterfactual fairness analysis
- **Figure 5**: Visual metrics and fairness decomposition

## 🔒 Repository Status

- **Privacy**: Private repository for AAAI 2025 submission
- **Anonymization**: Complete - all identifying information removed
- **Code Quality**: Verified and tested
- **Dependencies**: All required packages installed

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

**Note**: This repository is prepared for AAAI 2025 submission. All identifying information has been removed for the review process.
