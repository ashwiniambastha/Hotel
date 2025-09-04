"""
Data Transformation Module

This module contains functions for data transformation including
scaling, normalization, and dimensionality reduction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats


def apply_standard_scaling(X_train, X_test):
    """
    Apply standard scaling to the data.
    
    Args:
        X_train (np.array): Training features
        X_test (np.array): Testing features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def apply_minmax_scaling(X_train, X_test):
    """
    Apply min-max scaling to the data.
    
    Args:
        X_train (np.array): Training features
        X_test (np.array): Testing features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def apply_robust_scaling(X_train, X_test):
    """
    Apply robust scaling to the data.
    
    Args:
        X_train (np.array): Training features
        X_test (np.array): Testing features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def apply_pca(X_train, X_test, n_components=None, explained_variance_threshold=0.95):
    """
    Apply Principal Component Analysis to the data.
    
    Args:
        X_train (np.array): Training features
        X_test (np.array): Testing features
        n_components (int, optional): Number of components to keep
        explained_variance_threshold (float): Threshold for cumulative explained variance
        
    Returns:
        tuple: (X_train_pca, X_test_pca, pca, explained_variance_ratio)
    """
    if n_components is None:
        # Find number of components that explain the threshold variance
        pca_temp = PCA()
        pca_temp.fit(X_train)
        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= explained_variance_threshold) + 1
    
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    
    return X_train_pca, X_test_pca, pca, explained_variance_ratio


def plot_pca_variance(pca, title="PCA Explained Variance"):
    """
    Plot PCA explained variance ratio.
    
    Args:
        pca (PCA): Fitted PCA object
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Individual explained variance
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Individual Explained Variance')
    
    # Cumulative explained variance
    plt.subplot(1, 2, 2)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum) + 1), cumsum, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_scaling_comparison(X_original, X_scaled, feature_names=None, scaler_name="StandardScaler"):
    """
    Plot comparison of original and scaled data.
    
    Args:
        X_original (np.array): Original data
        X_scaled (np.array): Scaled data
        feature_names (list): List of feature names
        scaler_name (str): Name of the scaler used
    """
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X_original.shape[1])]
    
    n_features = min(4, X_original.shape[1])  # Plot max 4 features
    
    fig, axes = plt.subplots(2, n_features, figsize=(4 * n_features, 8))
    if n_features == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_features):
        # Original data
        axes[0, i].hist(X_original[:, i], bins=30, alpha=0.7, color='blue')
        axes[0, i].set_title(f'{feature_names[i]} (Original)')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Frequency')
        
        # Scaled data
        axes[1, i].hist(X_scaled[:, i], bins=30, alpha=0.7, color='red')
        axes[1, i].set_title(f'{feature_names[i]} ({scaler_name})')
        axes[1, i].set_xlabel('Value')
        axes[1, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()


def apply_log_transformation(X, columns=None):
    """
    Apply log transformation to specified columns.
    
    Args:
        X (pd.DataFrame): Input data
        columns (list): List of column names to transform
        
    Returns:
        pd.DataFrame: Data with log-transformed columns
    """
    X_transformed = X.copy()
    
    if columns is None:
        # Apply to all numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        columns = numeric_columns.tolist()
    
    for col in columns:
        if col in X_transformed.columns:
            # Add small constant to avoid log(0)
            X_transformed[col] = np.log1p(X_transformed[col])
    
    return X_transformed


def apply_sqrt_transformation(X, columns=None):
    """
    Apply square root transformation to specified columns.
    
    Args:
        X (pd.DataFrame): Input data
        columns (list): List of column names to transform
        
    Returns:
        pd.DataFrame: Data with sqrt-transformed columns
    """
    X_transformed = X.copy()
    
    if columns is None:
        # Apply to all numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        columns = numeric_columns.tolist()
    
    for col in columns:
        if col in X_transformed.columns:
            X_transformed[col] = np.sqrt(X_transformed[col])
    
    return X_transformed


def comprehensive_transformation_pipeline(X_train, X_test, scaling_method='standard', 
                                        apply_pca_flag=True, n_components=None):
    """
    Apply comprehensive transformation pipeline.
    
    Args:
        X_train (np.array): Training features
        X_test (np.array): Testing features
        scaling_method (str): Scaling method ('standard', 'minmax', 'robust')
        apply_pca_flag (bool): Whether to apply PCA
        n_components (int): Number of PCA components
        
    Returns:
        dict: Dictionary containing all transformed data and transformers
    """
    results = {}
    
    # 1. Apply scaling
    if scaling_method == 'standard':
        X_train_scaled, X_test_scaled, scaler = apply_standard_scaling(X_train, X_test)
    elif scaling_method == 'minmax':
        X_train_scaled, X_test_scaled, scaler = apply_minmax_scaling(X_train, X_test)
    elif scaling_method == 'robust':
        X_train_scaled, X_test_scaled, scaler = apply_robust_scaling(X_train, X_test)
    else:
        raise ValueError("scaling_method must be 'standard', 'minmax', or 'robust'")
    
    results['scaler'] = scaler
    results['X_train_scaled'] = X_train_scaled
    results['X_test_scaled'] = X_test_scaled
    
    # 2. Apply PCA if requested
    if apply_pca_flag:
        X_train_pca, X_test_pca, pca, explained_variance_ratio = apply_pca(
            X_train_scaled, X_test_scaled, n_components
        )
        results['pca'] = pca
        results['X_train_pca'] = X_train_pca
        results['X_test_pca'] = X_test_pca
        results['explained_variance_ratio'] = explained_variance_ratio
        
        # Plot PCA variance
        plot_pca_variance(pca)
    
    return results


def get_transformation_summary(results):
    """
    Get summary of transformation results.
    
    Args:
        results (dict): Results from transformation pipeline
        
    Returns:
        dict: Summary of transformations
    """
    summary = {}
    
    if 'scaler' in results:
        summary['scaling_method'] = type(results['scaler']).__name__
        summary['scaling_params'] = results['scaler'].get_params()
    
    if 'pca' in results:
        summary['pca_components'] = results['pca'].n_components_
        summary['explained_variance_ratio'] = results['explained_variance_ratio']
        summary['cumulative_variance'] = np.cumsum(results['explained_variance_ratio'])
    
    return summary
