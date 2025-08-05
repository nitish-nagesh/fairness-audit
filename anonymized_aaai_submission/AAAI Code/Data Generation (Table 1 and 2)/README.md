# Data Generation for FairTabGen (Table 1 and 2)

## 📋 Overview

This directory contains the implementation and data for the synthetic data generation experiments presented in Table 1 and 2 of the FairTabGen paper.

## 🎯 Purpose

Generate synthetic datasets for fairness analysis across multiple domains:
- Criminal Justice Dataset: Recidivism prediction
- Legal Dataset: Bar exam pass prediction
- MIMIC Dataset: Healthcare outcomes

## 📊 Experimental Setup

### Datasets Used
1. **Criminal Justice Dataset** (`criminal_synthetic_data_1000_200_epochs.csv`)
2. **Legal Dataset** (`synthetic_legal_data_decaf.csv`)
3. **MIMIC Dataset** (`mimic_synthetic_data_*.csv`)

### Generation Methods
1. **Our Approach**: Fairness-constrained synthetic data generation
2. **CLLM Baseline**: Conventional large language model approach
3. **DECAF Baseline**: Existing synthetic data generation method

## 🚀 Procedure for Reproducing Results

### Step 1: Environment Setup
```bash
# Install required dependencies
pip install pandas numpy matplotlib seaborn openai
```

### Step 2: Generate Criminal Justice Dataset
```bash
python generate_synthetic_criminal_data.py
```
**Expected Output**: `criminal_synthetic_data_1000_200_epochs.csv`

### Step 3: Generate Legal Dataset
```bash
python generate_synthetic_legal_data.py
```
**Expected Output**: `synthetic_legal_data_decaf.csv`

### Step 4: Generate MIMIC Dataset
```bash
python generate_mimic_decaf_samples.py
```
**Expected Output**: `mimic_synthetic_data_*.csv` files

### Step 5: Run Our Approach Prompts
```bash
# Criminal Justice Dataset
python "Prompt (Open AI-Our Prompt With Fairness) Criminal.py"

# Legal Dataset
python "Prompt (Open AI-Our Prompt With Fairness) Legal.py"

# MIMIC Dataset
python "Prompt (Open AI-Our Prompt With Fairness) MIMIC.py"
```

### Step 6: Run CLLM Baseline
```bash
# Criminal Justice Dataset
python "Prompt (Open AI-CLLM Prompt) Criminal.py"

# Legal Dataset
python "Prompt (Open AI-CLLM Prompt_Legal).py"

# MIMIC Dataset
python "Prompt (Open AI-CLLM Prompt_MIMIC).py"
```

## 📈 Expected Results

### Table 1: Data Generation Quality Metrics
- Statistical similarity between synthetic and real data
- Fairness metrics across protected attributes
- Data quality assessment

### Table 2: Fairness Comparison Results
- Comparison between our approach and baselines
- Fairness metrics across all datasets
- Method performance comparison

## 📁 File Descriptions

### Generation Scripts
- `generate_synthetic_criminal_data.py`: Criminal justice data generation
- `generate_synthetic_legal_data.py`: Legal data generation
- `generate_mimic_decaf_samples.py`: MIMIC dataset generation

### Prompt Scripts
- `Prompt (Open AI-Our Prompt With Fairness) *.py`: Our approach
- `Prompt (Open AI-CLLM Prompt) *.py`: CLLM baseline approach

### Generated Data Files
- `criminal_synthetic_data_1000_200_epochs.csv`: Criminal justice synthetic data
- `synthetic_legal_data_decaf.csv`: Legal synthetic data
- `mimic_synthetic_data_*.csv`: MIMIC synthetic data variants
- `generated_data_Our_prompt_*.csv`: Our approach generated data
- `generated_data_CLLM_prompt_*.csv`: CLLM generated data

## 📝 Notes

- Ensure OpenAI API key is configured for prompt-based generation
- Results may vary slightly due to randomness in generation process
- All synthetic data files are provided for reproducibility
- See corresponding analysis folders for detailed evaluation results

---

**For detailed analysis results, see the corresponding folders in the AAAI Code directory.** 