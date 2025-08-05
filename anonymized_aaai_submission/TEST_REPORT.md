# AAAI Code Testing Report

## Overview
This report summarizes the testing of all files in the AAAI Code directory.

## Test Results
- All Python files syntax checked
- All R files syntax checked  
- All Jupyter notebooks validated
- Data files verified

## Directory Structure
```
AAAI Code/
├── Data Generation (Table 1 and 2)/
│   ├── generate_synthetic_criminal_data.py
│   ├── generate_synthetic_legal_data.py
│   ├── generate_mimic_decaf_samples.py
│   ├── Prompt (Open AI-Our Prompt With Fairness) Criminal.py
│   ├── Prompt (Open AI-Our Prompt With Fairness) Legal.py
│   ├── Prompt (Open AI-Our Prompt With Fairness) MIMIC.py
│   ├── Prompt (Open AI-CLLM Prompt) Criminal.py
│   ├── Prompt (Open AI-CLLM Prompt_Legal).py
│   ├── Prompt (Open AI-CLLM Prompt_MIMIC).py
│   └── [Data files]
├── Data Distribution (Fig 2)/
│   ├── Data Analysis.ipynb
│   └── [Data files]
├── Model-Based Causal Fairness (Table 3 and Fig 3)/
│   ├── Various .R files
│   ├── COMPAS.ipynb
│   └── [Prediction files]
├── Data quality and counterfactual fairness (Fig 4)/
├── Visual metrics (Fig 5)/
└── Real Data/

## Testing Status
- ✅ All files anonymized
- ✅ All file references updated
- ✅ All syntax validated
- ✅ Ready for submission

## Next Steps
1. Provide required datasets (MIMIC, etc.)
2. Run specific experiments as needed
3. Verify outputs match expected results
4. Submit to AAAI

## Notes
- Large data files (>10MB) are present and may need to be provided separately
- MIMIC dataset not included due to size constraints
- All code is anonymized and ready for submission
