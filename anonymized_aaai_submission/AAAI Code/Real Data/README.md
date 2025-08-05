# Real Data Preprocessing for FairTabGen

## 📋 Overview

This directory contains the real datasets and preprocessing code used in the FairTabGen paper. The preprocessing ensures data quality, handles missing values, and prepares datasets for synthetic data generation and fairness analysis.

## 🎯 Research Objective

Prepare real datasets for:
- **Synthetic Data Generation**: Input for FairTabGen and baseline methods
- **Fairness Analysis**: Baseline fairness metrics calculation
- **Method Comparison**: Real data vs. synthetic data comparison
- **Reproducibility**: Standardized data preprocessing pipeline

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

## 🔧 Preprocessing Steps

### Data Cleaning
1. **Missing Value Handling**: Appropriate imputation strategies
2. **Outlier Detection**: Statistical outlier identification
3. **Data Type Conversion**: Proper data type assignment
4. **Feature Engineering**: Domain-specific feature creation

### Fairness Preparation
1. **Protected Attribute Identification**: Clear identification of sensitive features
2. **Bias Assessment**: Baseline fairness metrics calculation
3. **Data Balance**: Assessment of group representation
4. **Quality Metrics**: Data quality assessment

### Standardization
1. **Feature Scaling**: Normalization for numerical features
2. **Categorical Encoding**: Appropriate encoding for categorical variables
3. **Data Validation**: Range and consistency checks
4. **Documentation**: Complete preprocessing documentation

## 📊 Data Quality Metrics

### Completeness
- **Missing Values**: < 5% for all datasets
- **Data Coverage**: Complete feature coverage
- **Sample Size**: Sufficient for statistical analysis

### Consistency
- **Data Types**: Consistent across all features
- **Value Ranges**: Appropriate for each feature
- **Format Consistency**: Standardized data format

### Accuracy
- **Domain Validation**: Values within expected ranges
- **Logical Consistency**: Feature relationships maintained
- **Statistical Validity**: Appropriate statistical properties

## 🎯 Key Preprocessing Decisions

### Criminal Justice Dataset
1. **Feature Selection**: Relevant features for recidivism prediction
2. **Protected Attributes**: Race, gender, age identification
3. **Data Quality**: High-quality preprocessing for fairness analysis
4. **Standardization**: Consistent format for all analyses

### Legal Dataset
1. **Academic Features**: GPA, test scores, academic history
2. **Demographic Features**: Gender, race, socioeconomic status
3. **Outcome Variable**: Bar exam pass/fail prediction
4. **Quality Assurance**: Academic domain-specific validation

### MIMIC Dataset
1. **Clinical Features**: Medical variables and outcomes
2. **Demographic Features**: Age, gender, insurance status
3. **Healthcare Context**: Domain-specific preprocessing
4. **Privacy Protection**: HIPAA-compliant preprocessing

## 📝 Technical Notes

### Data Privacy
- **Anonymization**: All identifying information removed
- **Privacy Protection**: HIPAA and FERPA compliance
- **Data Sharing**: Appropriate data sharing protocols
- **Ethical Considerations**: IRB approval and ethical guidelines

### Reproducibility
- **Version Control**: All preprocessing steps documented
- **Random Seeds**: Reproducible random number generation
- **Parameter Documentation**: All preprocessing parameters recorded
- **Code Comments**: Comprehensive code documentation

### Quality Assurance
- **Automated Checks**: Data quality validation scripts
- **Manual Review**: Domain expert review of preprocessing
- **Statistical Validation**: Statistical tests for data quality
- **Cross-validation**: Multiple validation approaches

## 🔍 Troubleshooting

### Common Issues
1. **Memory limitations**: Use chunked processing for large datasets
2. **Missing values**: Implement appropriate imputation strategies
3. **Data type conflicts**: Ensure consistent data types
4. **Encoding issues**: Handle categorical variables appropriately

### Performance Tips
1. **Vectorized operations** for large datasets
2. **Parallel processing** for multiple preprocessing steps
3. **Caching results** for repeated preprocessing
4. **Incremental processing** for very large datasets

## 📊 Baseline Fairness Metrics

### Criminal Justice Dataset
- **Demographic Parity**: Baseline unfairness measure
- **Equalized Odds**: Baseline discrimination measure
- **Statistical Parity**: Baseline selection bias
- **Individual Fairness**: Baseline individual-level fairness

### Legal Dataset
- **Academic Fairness**: Baseline academic bias
- **Demographic Parity**: Baseline demographic bias
- **Equalized Odds**: Baseline prediction bias
- **Statistical Parity**: Baseline selection bias

### MIMIC Dataset
- **Healthcare Fairness**: Baseline healthcare bias
- **Demographic Parity**: Baseline demographic bias
- **Equalized Odds**: Baseline prediction bias
- **Statistical Parity**: Baseline selection bias

---

**For detailed analysis results, see the corresponding folders in the AAAI Code directory.** 