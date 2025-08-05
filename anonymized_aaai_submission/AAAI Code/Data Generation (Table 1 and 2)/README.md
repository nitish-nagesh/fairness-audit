# Data Generation for FairTabGen (Table 1 and 2)

## 📋 Overview

This directory contains the implementation and data for the synthetic data generation experiments presented in **Table 1 and 2** of the FairTabGen paper. The experiments demonstrate our approach to generating fair synthetic tabular data for causal fairness auditing.

## 🎯 Research Objective

Generate fair synthetic datasets that maintain statistical properties while improving fairness metrics across multiple domains:
- **Criminal Justice Dataset**: Recidivism prediction
- **Legal Dataset**: Bar exam pass prediction
- **MIMIC Dataset**: Healthcare outcomes

## 📊 Experimental Setup

### Datasets Used
1. **Criminal Justice Dataset** (`criminal_synthetic_data_1000_200_epochs.csv`)
2. **Legal Dataset** (`synthetic_legal_data_decaf.csv`)
3. **MIMIC Dataset** (`mimic_synthetic_data_*.csv`)

### Generation Methods
1. **Our FairTabGen Approach**: Fairness-constrained synthetic data generation
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

### Step 5: Run FairTabGen Prompts
```bash
# Criminal Justice Dataset with FairTabGen
python "Prompt (Open AI-Our Prompt With Fairness) Criminal.py"

# Legal Dataset with FairTabGen
python "Prompt (Open AI-Our Prompt With Fairness) Legal.py"

# MIMIC Dataset with FairTabGen
python "Prompt (Open AI-Our Prompt With Fairness) MIMIC.py"
```

### Step 6: Run CLLM Baseline
```bash
# Criminal Justice Dataset with CLLM
python "Prompt (Open AI-CLLM Prompt) Criminal.py"

# Legal Dataset with CLLM
python "Prompt (Open AI-CLLM Prompt_Legal).py"

# MIMIC Dataset with CLLM
python "Prompt (Open AI-CLLM Prompt_MIMIC).py"
```

## 📈 Expected Results

### Table 1: Data Generation Quality Metrics
- **Statistical Similarity**: Compare synthetic vs. real data distributions
- **Fairness Metrics**: Measure improvement in fairness across protected attributes
- **Data Quality**: Assess synthetic data quality and diversity

### Table 2: Fairness Comparison Results
- **Our FairTabGen**: Improved fairness metrics across all datasets
- **CLLM Baseline**: Conventional approach results
- **DECAF Baseline**: Existing method comparison

## 📁 File Descriptions

### Generation Scripts
- `generate_synthetic_criminal_data.py`: FairTabGen for criminal justice data
- `generate_synthetic_legal_data.py`: FairTabGen for legal data
- `generate_mimic_decaf_samples.py`: MIMIC dataset generation

### Prompt Scripts
- `Prompt (Open AI-Our Prompt With Fairness) *.py`: FairTabGen approach
- `Prompt (Open AI-CLLM Prompt) *.py`: CLLM baseline approach

### Generated Data Files
- `criminal_synthetic_data_1000_200_epochs.csv`: Criminal justice synthetic data
- `synthetic_legal_data_decaf.csv`: Legal synthetic data
- `mimic_synthetic_data_*.csv`: MIMIC synthetic data variants
- `generated_data_Our_prompt_*.csv`: FairTabGen generated data
- `generated_data_CLLM_prompt_*.csv`: CLLM generated data

## 🔧 Key Parameters

### FairTabGen Parameters
- **Fairness Constraint Weight**: Controls fairness vs. quality trade-off
- **Generation Epochs**: Number of training iterations
- **Sample Size**: Number of synthetic samples generated

### Evaluation Metrics
- **Statistical Distance**: KL divergence, Wasserstein distance
- **Fairness Metrics**: Demographic parity, equalized odds
- **Data Quality**: Completeness, consistency, accuracy

## 📊 Results Interpretation

1. **Fairness Improvement**: FairTabGen shows consistent improvement in fairness metrics
2. **Data Quality**: Maintains statistical properties while improving fairness
3. **Scalability**: Efficient generation across different dataset sizes
4. **Robustness**: Consistent results across multiple domains

## 🎯 Key Findings

- **FairTabGen outperforms baselines** in fairness metrics across all datasets
- **Statistical properties preserved** while improving fairness
- **Scalable approach** for different dataset sizes and domains
- **Robust performance** across multiple evaluation metrics

## 📝 Notes

- Ensure OpenAI API key is configured for prompt-based generation
- Results may vary slightly due to randomness in generation process
- All synthetic data files are provided for reproducibility
- See corresponding analysis folders for detailed evaluation results

---

**For detailed analysis results, see the corresponding folders in the AAAI Code directory.** 