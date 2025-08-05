# Data Generation for FairTabGen (Table 1 and 2)

## 📋 Overview

This directory contains the implementation and data for the synthetic data generation experiments presented in Table 1 and 2 of the FairTabGen paper.

## 🚀 Procedure for Reproducing Results

### Environment Setup
```bash
pip install pandas numpy matplotlib seaborn openai
```

### Run Generation Scripts
```bash
python generate_synthetic_criminal_data.py
python generate_synthetic_legal_data.py
python generate_mimic_decaf_samples.py
```

### Run Prompt Scripts
```bash
python "Prompt (Open AI-Our Prompt With Fairness) Criminal.py"
python "Prompt (Open AI-Our Prompt With Fairness) Legal.py"
python "Prompt (Open AI-Our Prompt With Fairness) MIMIC.py"
python "Prompt (Open AI-CLLM Prompt) Criminal.py"
python "Prompt (Open AI-CLLM Prompt_Legal).py"
python "Prompt (Open AI-CLLM Prompt_MIMIC).py"
```

## 📁 File Descriptions

### Generation Scripts
- `generate_synthetic_criminal_data.py`: Criminal justice data generation
- `generate_synthetic_legal_data.py`: Legal data generation
- `generate_mimic_decaf_samples.py`: MIMIC dataset generation

### Prompt Scripts
- `Prompt (Open AI-Our Prompt With Fairness) *.py`: Our approach
- `Prompt (Open AI-CLLM Prompt) *.py`: CLLM baseline approach

### Data Files
- `criminal_synthetic_data_1000_200_epochs.csv`: Criminal justice synthetic data
- `synthetic_legal_data_decaf.csv`: Legal synthetic data
- `mimic_synthetic_data_*.csv`: MIMIC synthetic data variants
- `generated_data_Our_prompt_*.csv`: Our approach generated data
- `generated_data_CLLM_prompt_*.csv`: CLLM generated data 