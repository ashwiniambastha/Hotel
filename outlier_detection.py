"""
Outlier Detection Module

This module contains functions for detecting and handling outliers
in the dataset using various statistical methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def detect_outliers_zscore(data, columns, threshold=3):
    """
    Detect outliers using Z-score method.
    
    Args:
        data (pd.DataFrame): Input data
        columns (list): List of column names to check for outliers
        threshold (float): Z-score threshold for outlier detection
        
    Returns:
        tuple: (outlier_mask, outlier_data, outlier_indices)
    """
    outlier_mask = pd.DataFrame(False, index=data.index, columns=columns)
    
    for col in columns:
        if col in data.columns:
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            outlier_mask[col] = z_scores > threshold
    
    # Get rows with any outlier
    any_outlier = outlier_mask.any(axis=1)
    outlier_data = data[any_outlier]
    outlier_indices = data.index[any_outlier].tolist()
    
    return outlier_mask, outlier_data, outlier_indices


def detect_outliers_iqr(data, columns, factor=1.5):
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Args:
        data (pd.DataFrame): Input data
        columns (list): List of column names to check for outliers
        factor (float): IQR factor for outlier detection
        
    Returns:
        tuple: (outlier_mask, outlier_data, outlier_indices)
    """
    outlier_mask = pd.DataFrame(False, index=data.index, columns=columns)
    
    for col in columns:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outlier_mask[col] = (data[col] < lower_bound) | (data[col] > upper_bound)
    
    # Get rows with any outlier
    any_outlier = outlier_mask.any(axis=1)
    outlier_data = data[any_outlier]
    outlier_indices = data.index[any_outlier].tolist()
    
    return outlier_mask, outlier_data, outlier_indices


def detect_outliers_isolation_forest(data, columns, contamination=0.1):
    """
    Detect outliers using Isolation Forest algorithm.
    
    Args:
        data (pd.DataFrame): Input data
        columns (list): List of column names to check for outliers
        contamination (float): Expected proportion of outliers
        
    Returns:
        tuple: (outlier_mask, outlier_data, outlier_indices, model)
    """
    # Select only numeric columns
    numeric_data = data[columns].select_dtypes(include=[np.number])
    
    # Handle missing values
    numeric_data = numeric_data.fillna(numeric_data.median())
    
    # Fit Isolation Forest
    model = IsolationForest(contamination=contamination, random_state=42)
    outlier_predictions = model.fit_predict(numeric_data)
    
    # Convert predictions to boolean mask
    outlier_mask = pd.Series(outlier_predictions == -1, index=data.index)
    outlier_data = data[outlier_mask]
    outlier_indices = data.index[outlier_mask].tolist()
    
    return outlier_mask, outlier_data, outlier_indices, model


def detect_outliers_lof(data, columns, n_neighbors=20, contamination=0.1):
    """
    Detect outliers using Local Outlier Factor (LOF) algorithm.
    
    Args:
        data (pd.DataFrame): Input data
        columns (list): List of column names to check for outliers
        n_neighbors (int): Number of neighbors for LOF
        contamination (float): Expected proportion of outliers
        
    Returns:
        tuple: (outlier_mask, outlier_data, outlier_indices, model)
    """
    # Select only numeric columns
    numeric_data = data[columns].select_dtypes(include=[np.number])
    
    # Handle missing values
    numeric_data = numeric_data.fillna(numeric_data.median())
    
    # Fit LOF
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outlier_predictions = model.fit_predict(numeric_data)
    
    # Convert predictions to boolean mask
    outlier_mask = pd.Series(outlier_predictions == -1, index=data.index)
    outlier_data = data[outlier_mask]
    outlier_indices = data.index[outlier_mask].tolist()
    
    return outlier_mask, outlier_data, outlier_indices, model


def plot_outlier_analysis(data, columns, outlier_mask=None, method_name="Outlier Detection"):
    """
    Plot outlier analysis for specified columns.
    
    Args:
        data (pd.DataFrame): Input data
        columns (list): List of column names to analyze
        outlier_mask (pd.Series): Boolean mask for outliers
        method_name (str): Name of the detection method
    """
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, col in enumerate(columns):
        row = i // n_cols
        col_idx = i % n_cols
        
        if n_rows == 1 and n_cols == 1:
            ax = axes
        elif n_rows == 1:
            ax = axes[col_idx]
        elif n_cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col_idx]
        
        # Box plot
        ax.boxplot(data[col].dropna())
        ax.set_title(f'{col} - {method_name}')
        ax.set_ylabel('Value')
        
        # Highlight outliers if mask provided
        if outlier_mask is not None:
            outlier_data = data[outlier_mask][col].dropna()
            if len(outlier_data) > 0:
                ax.scatter([1] * len(outlier_data), outlier_data, 
                          color='red', alpha=0.6, s=20, label='Outliers')
                ax.legend()
    
    # Hide empty subplots
    for i in range(len(columns), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        if n_rows == 1 and n_cols == 1:
            axes.set_visible(False)
        elif n_rows == 1:
            axes[col_idx].set_visible(False)
        elif n_cols == 1:
            axes[row].set_visible(False)
        else:
            axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def remove_outliers(data, outlier_indices):
    """
    Remove outliers from the dataset.
    
    Args:
        data (pd.DataFrame): Input data
        outlier_indices (list): List of indices to remove
        
    Returns:
        pd.DataFrame: Data with outliers removed
    """
    return data.drop(index=outlier_indices)


def cap_outliers(data, columns, method='iqr', factor=1.5):
    """
    Cap outliers instead of removing them.
    
    Args:
        data (pd.DataFrame): Input data
        columns (list): List of column names to cap
        method (str): Method for capping ('iqr' or 'zscore')
        factor (float): Factor for capping
        
    Returns:
        pd.DataFrame: Data with outliers capped
    """
    data_capped = data.copy()
    
    for col in columns:
        if col in data_capped.columns:
            if method == 'iqr':
                Q1 = data_capped[col].quantile(0.25)
                Q3 = data_capped[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                data_capped[col] = data_capped[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == 'zscore':
                mean = data_capped[col].mean()
                std = data_capped[col].std()
                
                lower_bound = mean - factor * std
                upper_bound = mean + factor * std
                
                data_capped[col] = data_capped[col].clip(lower=lower_bound, upper=upper_bound)
    
    return data_capped


def comprehensive_outlier_analysis(data, columns, methods=['zscore', 'iqr', 'isolation_forest']):
    """
    Perform comprehensive outlier analysis using multiple methods.
    
    Args:
        data (pd.DataFrame): Input data
        columns (list): List of column names to analyze
        methods (list): List of methods to use
        
    Returns:
        dict: Dictionary containing results from all methods
    """
    results = {}
    
    for method in methods:
        print(f"\n=== {method.upper()} Method ===")
        
        if method == 'zscore':
            outlier_mask, outlier_data, outlier_indices = detect_outliers_zscore(data, columns)
            results[method] = {
                'outlier_mask': outlier_mask,
                'outlier_data': outlier_data,
                'outlier_indices': outlier_indices,
                'n_outliers': len(outlier_indices)
            }
            
        elif method == 'iqr':
            outlier_mask, outlier_data, outlier_indices = detect_outliers_iqr(data, columns)
            results[method] = {
                'outlier_mask': outlier_mask,
                'outlier_data': outlier_data,
                'outlier_indices': outlier_indices,
                'n_outliers': len(outlier_indices)
            }
            
        elif method == 'isolation_forest':
            outlier_mask, outlier_data, outlier_indices, model = detect_outliers_isolation_forest(data, columns)
            results[method] = {
                'outlier_mask': outlier_mask,
                'outlier_data': outlier_data,
                'outlier_indices': outlier_indices,
                'n_outliers': len(outlier_indices),
                'model': model
            }
            
        elif method == 'lof':
            outlier_mask, outlier_data, outlier_indices, model = detect_outliers_lof(data, columns)
            results[method] = {
                'outlier_mask': outlier_mask,
                'outlier_data': outlier_data,
                'outlier_indices': outlier_indices,
                'n_outliers': len(outlier_indices),
                'model': model
            }
        
        print(f"Number of outliers detected: {results[method]['n_outliers']}")
        
        # Plot analysis
        plot_outlier_analysis(data, columns, outlier_mask, method)
    
    return results


def get_outlier_summary(results):
    """
    Get summary of outlier detection results.
    
    Args:
        results (dict): Results from comprehensive outlier analysis
        
    Returns:
        pd.DataFrame: Summary of outlier detection results
    """
    summary_data = []
    
    for method, result in results.items():
        summary_data.append({
            'Method': method,
            'Number of Outliers': result['n_outliers'],
            'Percentage': (result['n_outliers'] / len(result['outlier_mask'])) * 100
        })
    
    return pd.DataFrame(summary_data)
