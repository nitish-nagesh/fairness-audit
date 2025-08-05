#!/usr/bin/env python3
"""
Generate synthetic data for the legal dataset using DECAF synthesizer
"""

import pandas as pd
import os
from synthcity.plugins import Plugins

def generate_synthetic_legal_data():
    """
    Generate synthetic data for the legal dataset using DECAF
    """
    
    # Path to your legal dataset
    data_path = os.path.expanduser("~/Downloads/bar_pass_prediction (processed version).csv")
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        print("Please make sure the file 'bar_pass_prediction (processed version).csv' is in your Downloads folder")
        return
    
    # Load the legal dataset
    print("Loading legal dataset...")
    data = pd.read_csv(data_path)
    
    print(f"Original data shape: {data.shape}")
    print(f"Original data columns: {list(data.columns)}")
    print("\nFirst few rows of original data:")
    print(data.head())
    
    # Display data info
    print("\nData types:")
    print(data.dtypes)
    
    print("\nMissing values:")
    print(data.isnull().sum())
    
    # Initialize DECAF synthesizer with default parameters
    print("\nInitializing DECAF synthesizer...")
    plugins = Plugins()
    decaf_synthesizer = plugins.get("decaf")
    
    # Fit the synthesizer to your data
    print("Fitting DECAF synthesizer...")
    decaf_synthesizer.fit(data)
    
    # Generate 1000 synthetic samples
    print("Generating 1000 synthetic samples...")
    synthetic_data_loader = decaf_synthesizer.generate(count=1000)
    
    # Access the synthetic data
    synthetic_data = synthetic_data_loader.data
    
    print(f"\nSynthetic data shape: {synthetic_data.shape}")
    print("\nFirst few rows of synthetic data:")
    print(synthetic_data.head())
    
    # Compare basic statistics
    print("\nComparing basic statistics:")
    print("Original data statistics:")
    print(data.describe())
    print("\nSynthetic data statistics:")
    print(synthetic_data.describe())
    
    # Save synthetic data
    output_path = "synthetic_legal_data_decaf.csv"
    synthetic_data.to_csv(output_path, index=False)
    print(f"\nSynthetic data saved to '{output_path}'")
    
    return synthetic_data

if __name__ == "__main__":
    print("Generating synthetic data for legal dataset using DECAF...")
    synthetic_data = generate_synthetic_legal_data() 