#!/usr/bin/env python3
"""
Generate synthetic data for the MIMIC dataset using DECAF synthesizer
"""

import pandas as pd
import numpy as np
import os
from synthcity.plugins import Plugins
from sklearn.impute import SimpleImputer

def generate_synthetic_mimic_data_3700():
    """
    Generate 3,700 synthetic samples for the MIMIC dataset using DECAF
    """
    
    # Path to your MIMIC dataset
    data_path = os.path.expanduser("~/Downloads/mimic_dataset_final2r.csv")
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        print("Please make sure the file 'mimic_dataset_final2r.csv' is in your Downloads folder")
        return
    
    # Load the MIMIC dataset
    print("Loading MIMIC dataset...")
    data = pd.read_csv(data_path)
    
    print(f"MIMIC data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"\nData types:")
    print(data.dtypes)
    print(f"\nFirst few rows:")
    print(data.head())
    print(f"\nMissing values:")
    print(data.isnull().sum())
    
    # Handle missing values
    print("\nHandling missing values...")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    data_imputed = data.copy()
    
    if len(numeric_cols) > 0:
        print(f"Imputing {len(numeric_cols)} numeric columns with median...")
        numeric_imputer = SimpleImputer(strategy='median')
        data_imputed[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])
    
    if len(categorical_cols) > 0:
        print(f"Imputing {len(categorical_cols)} categorical columns with most frequent value...")
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        data_imputed[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
    
    print(f"Missing values after imputation:")
    print(data_imputed.isnull().sum())
    
    print("\nInitializing DECAF synthesizer...")
    plugins = Plugins()
    decaf = plugins.get("decaf")
    
    print("Fitting DECAF to MIMIC data...")
    decaf.fit(data_imputed)
    
    print("Generating 3,700 synthetic samples...")
    synthetic_loader = decaf.generate(count=3700)
    
    synthetic_data = synthetic_loader.data
    
    print(f"\nSynthetic data generated successfully!")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"\nFirst few rows of synthetic data:")
    print(synthetic_data.head())
    
    print("\n=== ORIGINAL DATA STATISTICS (after imputation) ===")
    print(data_imputed.describe())
    
    print("\n=== SYNTHETIC DATA STATISTICS ===")
    print(synthetic_data.describe())
    
    if len(categorical_cols) > 0:
        print("\n=== CATEGORICAL COLUMNS COMPARISON ===")
        for col in categorical_cols:
            if col in synthetic_data.columns:
                print(f"\n{col}:")
                print(f"Original: {data_imputed[col].value_counts().to_dict()}")
                print(f"Synthetic: {synthetic_data[col].value_counts().to_dict()}")
    
    output_filename = "mimic_synthetic_data_3700_samples.csv"
    synthetic_data.to_csv(output_filename, index=False)
    
    print(f"\nSynthetic data saved to: {output_filename}")
    print(f"File size: {os.path.getsize(output_filename) / 1024:.1f} KB")
    
    return synthetic_data

if __name__ == "__main__":
    synthetic_data = generate_synthetic_mimic_data_3700()
    if synthetic_data is not None:
        print("\n✅ MIMIC synthetic data generation completed successfully!")
        print(f"Generated {len(synthetic_data)} synthetic samples")