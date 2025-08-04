#!/usr/bin/env python3
"""
Generate synthetic data for the COMPAS dataset using DECAF synthesizer
"""

import pandas as pd
import os
from synthcity.plugins import Plugins

def generate_synthetic_compas_data():
    """
    Generate synthetic data for the COMPAS dataset using DECAF
    """
    
    # Path to your COMPAS dataset
    data_path = os.path.expanduser("~/Downloads/compas_selected (1).csv")
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        print("Please make sure the file 'compas_selected (1).csv' is in your Downloads folder")
        return
    
    # Load the COMPAS dataset
    print("Loading COMPAS dataset...")
    data = pd.read_csv(data_path)
    
    print(f"COMPAS data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"\nData types:")
    print(data.dtypes)
    print(f"\nFirst few rows:")
    print(data.head())
    print(f"\nMissing values:")
    print(data.isnull().sum())
    
    # Initialize DECAF synthesizer
    print("\nInitializing DECAF synthesizer...")
    plugins = Plugins()
    decaf = plugins.get("decaf")
    
    # Fit the model to the COMPAS data
    print("Fitting DECAF to COMPAS data...")
    decaf.fit(data)
    
    # Generate 1000 synthetic samples
    print("Generating 1000 synthetic samples...")
    synthetic_loader = decaf.generate(count=1000)
    
    # Extract the synthetic data
    synthetic_data = synthetic_loader.data
    
    print(f"\nSynthetic data generated successfully!")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"\nFirst few rows of synthetic data:")
    print(synthetic_data.head())
    
    # Compare statistics
    print("\n=== ORIGINAL DATA STATISTICS ===")
    print(data.describe())
    
    print("\n=== SYNTHETIC DATA STATISTICS ===")
    print(synthetic_data.describe())
    
    # Compare categorical columns if any
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("\n=== CATEGORICAL COLUMNS COMPARISON ===")
        for col in categorical_cols:
            if col in synthetic_data.columns:
                print(f"\n{col}:")
                print(f"Original: {data[col].value_counts().to_dict()}")
                print(f"Synthetic: {synthetic_data[col].value_counts().to_dict()}")
    
    # Save synthetic data
    output_filename = "compas_synthetic_data_1000_samples.csv"
    synthetic_data.to_csv(output_filename, index=False)
    
    print(f"\nSynthetic data saved to: {output_filename}")
    print(f"File size: {os.path.getsize(output_filename) / 1024:.1f} KB")
    
    return synthetic_data

if __name__ == "__main__":
    synthetic_data = generate_synthetic_compas_data()
    if synthetic_data is not None:
        print("\n✅ COMPAS synthetic data generation completed successfully!")
        print(f"Generated {len(synthetic_data)} synthetic samples") 