# Causal Fairness Audit Framework

This repository contains a comprehensive framework for conducting causal fairness audits on machine learning models.

## Overview

The framework implements:
- Causal fairness decomposition analysis
- Synthetic data generation with fairness constraints
- Automated fairness explanation generation
- Multi-agent debate and critique systems
- Interactive Streamlit interface for fairness auditing

## Structure

- `main_app.py`: Main Streamlit application
- `synthetic_data_generator.py`: Synthetic data generation with fairness constraints
- `audit_runner.py`: Automated fairness audit pipeline
- `AAAI Code/`: Organized research code by figures and tables
  - `Data Generation (Table 1 and 2)/`: Data generation experiments
  - `Data Distribution (Fig 2)/`: Data distribution analysis
  - `Model-Based Causal Fairness (Table 3 and Fig 3)/`: Model-based fairness analysis
  - `Data quality and counterfactual fairness (Fig 4)/`: Data quality analysis
  - `Visual metrics (Fig 5)/`: Visual metrics and evaluation
  - `Real Data/`: Real dataset analysis
- `checkpoints/`: Model checkpoints and saved states

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Run the main application: `streamlit run main_app.py`
3. Upload dataset or use synthetic data generation
4. Conduct fairness audits and explanations

## Requirements

- Python 3.8+
- R with faircause package
- OpenAI API key for explanation generation
- Streamlit for web interface

## Citation

This work is submitted to AAAI 2024. Please cite appropriately if used.
