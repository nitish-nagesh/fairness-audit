# Real Data Preprocessing for FairTabGen

## 📋 Overview

This directory contains the real datasets and preprocessing code used in the FairTabGen paper. The preprocessing ensures data quality, handles missing values, and prepares datasets for synthetic data generation and fairness analysis.

## 🎯 Purpose

Prepare real datasets for:
- Synthetic data generation input
- Fairness analysis baseline calculation
- Method comparison evaluation
- Reproducibility pipeline

## 📊 Datasets Included

### 1. Criminal Justice Dataset (COMPAS)
- **File**: `compas_cleaned.csv`
- **Original**: `compas.arff`
- **Preprocessing**: `COMPAS_Preprocess.ipynb`
- **Description**: Recidivism prediction dataset
- **Features**: Demographics, criminal history, risk factors
- **Protected Attributes**: Race, gender, age

### 2. Legal Dataset (Bar Exam)
- **File**: `bar_pass_prediction (processed version).csv`
- **Original**: `bar_pass_prediction.csv`
- **Preprocessing**: `Law_Preprocess.ipynb`
- **Description**: Bar exam pass prediction dataset
- **Features**: Academic performance, demographics, test scores
- **Protected Attributes**: Gender, race, socioeconomic status

### 3. MIMIC Dataset (Healthcare)
- **Preprocessing**: `MIMIC_Preprocess.ipynb`
- **Description**: Healthcare outcomes dataset
- **Features**: Clinical variables, demographics, outcomes
- **Protected Attributes**: Age, gender, insurance status

## 🚀 Procedure for Reproducing Results

### Step 1: Environment Setup
```bash
# Install required dependencies
pip install pandas numpy matplotlib seaborn jupyter
```

### Step 2: Preprocess Criminal Justice Dataset
```python
# Run the COMPAS_Preprocess.ipynb notebook
# This will create compas_cleaned.csv from compas.arff
```

### Step 3: Preprocess Legal Dataset
```python
# Run the Law_Preprocess.ipynb notebook
# This will create bar_pass_prediction (processed version).csv
```

### Step 4: Preprocess MIMIC Dataset
```python
# Run the MIMIC_Preprocess.ipynb notebook
# This will prepare the MIMIC dataset for analysis
```

### Step 5: Verify Data Quality
```python
# In each notebook, verify:
# 1. Data completeness
# 2. Feature consistency
# 3. Protected attribute identification
# 4. Fairness baseline calculation
```

## 📈 Expected Results

### Criminal Justice Dataset
- **Original Size**: 5,295 samples
- **Cleaned Size**: 5,280 samples
- **Features**: 15 features after preprocessing
- **Protected Attributes**: Race, gender, age
- **Missing Values**: Handled appropriately

### Legal Dataset
- **Original Size**: Large dataset
- **Processed Size**: 21,313 samples
- **Features**: Academic and demographic features
- **Protected Attributes**: Gender, race, socioeconomic status
- **Quality**: High-quality processed data

### MIMIC Dataset
- **Size**: Healthcare dataset
- **Features**: Clinical and demographic variables
- **Protected Attributes**: Age, gender, insurance status
- **Quality**: Healthcare-specific preprocessing

## 📁 File Descriptions

### Original Data Files
- `compas.arff`: Original COMPAS dataset in ARFF format
- `bar_pass_prediction.csv`: Original bar exam dataset
- `MIMIC_Preprocess.ipynb`: MIMIC dataset preprocessing

### Processed Data Files
- `compas_cleaned.csv`: Preprocessed criminal justice dataset
- `bar_pass_prediction (processed version).csv`: Processed legal dataset

### Preprocessing Notebooks
- `COMPAS_Preprocess.ipynb`: Criminal justice data preprocessing
- `Law_Preprocess.ipynb`: Legal data preprocessing
- `MIMIC_Preprocess.ipynb`: MIMIC data preprocessing

## 📝 Notes

- All identifying information removed for privacy
- Missing values handled consistently
- Data types converted appropriately
- Protected attributes clearly identified
- Statistical validation performed

---

**For detailed analysis results, see the corresponding folders in the AAAI Code directory.** 