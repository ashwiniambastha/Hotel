"""
Feature Engineering Module

This module contains functions for feature engineering including
encoding, transformation, and feature extraction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer


def ordinal_encode_subscription_type(X_train, X_test):
    """
    Apply ordinal encoding to SubscriptionType column.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        
    Returns:
        tuple: (X_train_encoded, X_test_encoded, encoder)
    """
    oe = OrdinalEncoder(categories=[['Basic', 'Standard', 'Premium']])
    
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    X_train_encoded['SubscriptionType'] = oe.fit_transform(X_train[['SubscriptionType']])
    X_test_encoded['SubscriptionType'] = oe.transform(X_test[['SubscriptionType']])
    
    return X_train_encoded, X_test_encoded, oe


def one_hot_encode_gender(X_train, X_test):
    """
    Apply one-hot encoding to Gender column.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        
    Returns:
        tuple: (X_train_encoded, X_test_encoded, encoder)
    """
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    
    X_train_encoded = ohe.fit_transform(X_train[['Gender']])
    X_test_encoded = ohe.transform(X_test[['Gender']])
    
    return X_train_encoded, X_test_encoded, ohe


def apply_column_transformer(X_train, X_test):
    """
    Apply column transformer for mixed data types.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        
    Returns:
        tuple: (X_train_transformed, X_test_transformed, transformer)
    """
    transformer = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(categories=[['Basic', 'Standard', 'Premium']]), ['SubscriptionType']),
            ('onehot', OneHotEncoder(sparse_output=False, drop='first'), ['Gender'])
        ],
        remainder='passthrough'
    )
    
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)
    
    return X_train_transformed, X_test_transformed, transformer


def get_transformed_dataframe(X_transformed, transformer, original_index):
    """
    Convert transformed array back to DataFrame with proper column names.
    
    Args:
        X_transformed (np.array): Transformed feature array
        transformer (ColumnTransformer): Fitted transformer
        original_index (pd.Index): Original DataFrame index
        
    Returns:
        pd.DataFrame: Transformed data as DataFrame
    """
    feature_names = transformer.get_feature_names_out()
    df_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=original_index)
    
    return df_transformed


def apply_function_transformation(X, column, func=np.log1p):
    """
    Apply function transformation to a specific column.
    
    Args:
        X (pd.DataFrame): Input data
        column (str): Column name to transform
        func (callable): Function to apply (default: np.log1p)
        
    Returns:
        tuple: (transformed_data, transformer)
    """
    transformer = FunctionTransformer(func=func)
    transformed_data = transformer.fit_transform(X[[column]])
    
    return transformed_data, transformer


def plot_transformation_comparison(X, column, transformed_data, func_name="log1p"):
    """
    Plot comparison of original and transformed data.
    
    Args:
        X (pd.DataFrame): Original data
        column (str): Column name
        transformed_data (np.array): Transformed data
        func_name (str): Name of transformation function
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    plt.figure(figsize=(14, 8))
    
    # Original data histogram
    plt.subplot(221)
    sns.histplot(X[column], kde=True)
    plt.xlabel(f'PDF of {column} (before transformation)')
    plt.title(f'Original {column}')
    
    # Original data Q-Q plot
    plt.subplot(222)
    stats.probplot(X[column].dropna(), dist="norm", plot=plt)
    plt.xlabel(f'Q-Q plot of {column} (before transformation)')
    plt.title(f'Q-Q plot of {column} (before transformation)')
    
    # Transformed data histogram
    plt.subplot(223)
    sns.histplot(transformed_data, kde=True)
    plt.xlabel(f'PDF of {column} (after {func_name} transformation)')
    plt.title(f'Transformed {column}')
    
    # Transformed data Q-Q plot
    plt.subplot(224)
    stats.probplot(transformed_data.flatten(), dist="norm", plot=plt)
    plt.xlabel(f'Q-Q plot of {column} (after {func_name} transformation)')
    plt.title(f'Q-Q plot of {column} (after {func_name} transformation)')
    
    plt.tight_layout()
    plt.show()


def apply_knn_imputation(X, columns_to_impute):
    """
    Apply KNN imputation to specified columns.
    
    Args:
        X (pd.DataFrame): Input data
        columns_to_impute (list): List of column names to impute
        
    Returns:
        tuple: (X_imputed, imputer)
    """
    imputer = KNNImputer()
    X_imputed = X.copy()
    X_imputed[columns_to_impute] = imputer.fit_transform(X[columns_to_impute])
    
    return X_imputed, imputer


def extract_date_features(df, date_column='ContractEndDate'):
    """
    Extract features from date column.
    
    Args:
        df (pd.DataFrame): Input data
        date_column (str): Name of date column
        
    Returns:
        pd.DataFrame: Data with extracted date features
    """
    df_processed = df.copy()
    
    # Convert to datetime if not already
    df_processed[date_column] = pd.to_datetime(df_processed[date_column])
    
    # Extract year and day of week
    df_processed['ContractYear'] = df_processed[date_column].dt.year
    df_processed['ContractDayOfWeek'] = df_processed[date_column].dt.dayofweek
    
    return df_processed


def comprehensive_feature_engineering(X_train, X_test, float_columns=None):
    """
    Apply comprehensive feature engineering pipeline.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        float_columns (list): List of float columns for imputation
        
    Returns:
        dict: Dictionary containing all transformed data and transformers
    """
    if float_columns is None:
        float_columns = ['MonthlyCharges', 'ServiceUsage', 'Age', 'TotalTransactions']
    
    results = {}
    
    # 1. Apply column transformer for mixed data types
    X_train_trf, X_test_trf, column_transformer = apply_column_transformer(X_train, X_test)
    results['column_transformer'] = column_transformer
    results['X_train_transformed'] = X_train_trf
    results['X_test_transformed'] = X_test_trf
    
    # 2. Convert to DataFrame with proper column names
    X_train_df = get_transformed_dataframe(X_train_trf, column_transformer, X_train.index)
    X_test_df = get_transformed_dataframe(X_test_trf, column_transformer, X_test.index)
    results['X_train_df'] = X_train_df
    results['X_test_df'] = X_test_df
    
    # 3. Apply KNN imputation to float columns
    X_train_imputed, knn_imputer = apply_knn_imputation(X_train, float_columns)
    X_test_imputed, _ = apply_knn_imputation(X_test, float_columns)
    results['knn_imputer'] = knn_imputer
    results['X_train_imputed'] = X_train_imputed
    results['X_test_imputed'] = X_test_imputed
    
    return results
