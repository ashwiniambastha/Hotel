"""
Data Loading and Preprocessing Module

This module handles loading and basic preprocessing of the hotel customer churn dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(file_path='customer_churn_data_more_missing.csv'):
    """
    Load the customer churn dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset with CustomerID as index
    """
    df = pd.read_csv(file_path, index_col='CustomerID')
    return df


def get_data_info(df):
    """
    Get basic information about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Dictionary containing dataset information
    """
    info = {
        'shape': df.shape,
        'missing_values': df.isnull().sum(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes,
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    return info


def get_feature_types(df):
    """
    Separate features by data type.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        tuple: (float_features, object_features, float_df, object_df)
    """
    float_features = df.select_dtypes(include='float').columns.tolist()
    object_features = df.select_dtypes(include='object').columns.tolist()
    
    float_df = df.select_dtypes(include='float')
    object_df = df.select_dtypes(include='object')
    
    return float_features, object_features, float_df, object_df


def prepare_features_target(df):
    """
    Prepare features (X) and target (y) variables.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        tuple: (X, y) - Features and target arrays
    """
    X = df.iloc[:, 0:7]  # All columns except the last one (Churn)
    y = df.iloc[:, -1:]  # Last column (Churn)
    
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.DataFrame): Target variable
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with missing values handled
    """
    # Fill missing values in SubscriptionType with mode
    df['SubscriptionType'] = df['SubscriptionType'].fillna(df['SubscriptionType'].mode()[0])
    
    # Fill missing values in Gender with mode
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    
    return df


def convert_date_columns(df):
    """
    Convert date columns to datetime format and extract features.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with date features extracted
    """
    df['ContractEndDate'] = pd.to_datetime(df['ContractEndDate'])
    
    # Extract year and day of week
    df['ContractYear'] = df['ContractEndDate'].dt.year
    df['ContractDayOfWeek'] = df['ContractEndDate'].dt.dayofweek
    
    return df
