"""
Main Pipeline Script for Hotel Customer Churn Analysis

This script demonstrates the complete data science pipeline for analyzing
hotel customer churn data, including EDA, feature engineering, and preprocessing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import (
    load_data, get_data_info, get_feature_types, 
    prepare_features_target, split_data, handle_missing_values, convert_date_columns
)
from eda import comprehensive_eda
from feature_engineering import comprehensive_feature_engineering
from data_transformation import comprehensive_transformation_pipeline
from outlier_detection import comprehensive_outlier_analysis, get_outlier_summary


def main():
    """
    Main function to run the complete data science pipeline.
    """
    print("=== Hotel Customer Churn Analysis Pipeline ===\n")
    
    # Step 1: Load and explore data
    print("Step 1: Loading and exploring data...")
    df = load_data('customer_churn_data_more_missing.csv')
    
    # Get basic data information
    data_info = get_data_info(df)
    print(f"Dataset shape: {data_info['shape']}")
    print(f"Missing values:\n{data_info['missing_values']}")
    print(f"Duplicate rows: {data_info['duplicates']}")
    print(f"Memory usage: {data_info['memory_usage']} bytes")
    
    # Step 2: Handle missing values
    print("\nStep 2: Handling missing values...")
    df_processed = handle_missing_values(df)
    print("Missing values after processing:")
    print(df_processed.isnull().sum())
    
    # Step 3: Convert date columns
    print("\nStep 3: Converting date columns...")
    df_processed = convert_date_columns(df_processed)
    print("Date columns converted and features extracted")
    
    # Step 4: Exploratory Data Analysis
    print("\nStep 4: Performing Exploratory Data Analysis...")
    comprehensive_eda(df_processed)
    
    # Step 5: Prepare features and target
    print("\nStep 5: Preparing features and target...")
    X, y = prepare_features_target(df_processed)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Step 6: Split data
    print("\nStep 6: Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Step 7: Feature Engineering
    print("\nStep 7: Applying feature engineering...")
    float_columns = ['MonthlyCharges', 'ServiceUsage', 'Age', 'TotalTransactions']
    feature_results = comprehensive_feature_engineering(X_train, X_test, float_columns)
    print("Feature engineering completed")
    
    # Step 8: Outlier Detection
    print("\nStep 8: Detecting outliers...")
    outlier_results = comprehensive_outlier_analysis(
        X_train, float_columns, 
        methods=['zscore', 'iqr', 'isolation_forest']
    )
    
    # Get outlier summary
    outlier_summary = get_outlier_summary(outlier_results)
    print("\nOutlier Detection Summary:")
    print(outlier_summary)
    
    # Step 9: Data Transformation
    print("\nStep 9: Applying data transformations...")
    transformation_results = comprehensive_transformation_pipeline(
        feature_results['X_train_transformed'], 
        feature_results['X_test_transformed'],
        scaling_method='standard',
        apply_pca_flag=True,
        n_components=3
    )
    print("Data transformation completed")
    
    # Step 10: Final Data Preparation
    print("\nStep 10: Preparing final datasets...")
    
    # Get the final processed data
    X_train_final = transformation_results['X_train_scaled']
    X_test_final = transformation_results['X_test_scaled']
    
    if 'X_train_pca' in transformation_results:
        X_train_final = transformation_results['X_train_pca']
        X_test_final = transformation_results['X_test_pca']
        print(f"Final training set shape (after PCA): {X_train_final.shape}")
        print(f"Final test set shape (after PCA): {X_test_final.shape}")
    else:
        print(f"Final training set shape: {X_train_final.shape}")
        print(f"Final test set shape: {X_test_final.shape}")
    
    # Step 11: Save processed data
    print("\nStep 11: Saving processed data...")
    
    # Convert to DataFrames for easier handling
    if 'pca' in transformation_results:
        feature_names = [f'PC_{i+1}' for i in range(X_train_final.shape[1])]
    else:
        feature_names = feature_results['column_transformer'].get_feature_names_out()
    
    X_train_df = pd.DataFrame(X_train_final, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_final, columns=feature_names, index=X_test.index)
    
    # Save to CSV
    X_train_df.to_csv('X_train_processed.csv')
    X_test_df.to_csv('X_test_processed.csv')
    y_train.to_csv('y_train.csv')
    y_test.to_csv('y_test.csv')
    
    print("Processed data saved to CSV files")
    
    # Step 12: Summary
    print("\n=== Pipeline Summary ===")
    print(f"Original dataset shape: {df.shape}")
    print(f"Final training set shape: {X_train_final.shape}")
    print(f"Final test set shape: {X_test_final.shape}")
    print(f"Features used: {len(feature_names)}")
    print(f"Missing values in final data: {X_train_df.isnull().sum().sum()}")
    
    # Print outlier detection summary
    print("\nOutlier Detection Results:")
    for method, result in outlier_results.items():
        print(f"{method}: {result['n_outliers']} outliers detected")
    
    print("\n=== Pipeline Complete ===")
    
    return {
        'original_data': df,
        'processed_data': df_processed,
        'X_train': X_train_final,
        'X_test': X_test_final,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'outlier_results': outlier_results,
        'transformation_results': transformation_results
    }


def run_quick_analysis():
    """
    Run a quick analysis for demonstration purposes.
    """
    print("=== Quick Analysis Mode ===\n")
    
    # Load data
    df = load_data('customer_churn_data_more_missing.csv')
    print(f"Dataset loaded: {df.shape}")
    
    # Basic info
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Handle missing values
    df_processed = handle_missing_values(df)
    print(f"Missing values after processing: {df_processed.isnull().sum().sum()}")
    
    # Get feature types
    float_features, object_features, _, _ = get_feature_types(df_processed)
    print(f"Float features: {float_features}")
    print(f"Object features: {object_features}")
    
    # Prepare data
    X, y = prepare_features_target(df_processed)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    print("\n=== Quick Analysis Complete ===")


if __name__ == "__main__":
    # Run the complete pipeline
    try:
        results = main()
        print("\nPipeline executed successfully!")
    except FileNotFoundError:
        print("Error: 'customer_churn_data_more_missing.csv' file not found.")
        print("Please ensure the data file is in the current directory.")
        print("\nRunning quick analysis instead...")
        run_quick_analysis()
    except Exception as e:
        print(f"Error during pipeline execution: {str(e)}")
        print("\nRunning quick analysis instead...")
        run_quick_analysis()
