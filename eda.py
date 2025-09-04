"""
Exploratory Data Analysis Module

This module contains functions for performing exploratory data analysis
on the hotel customer churn dataset.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from scipy import stats


def plot_distributions(df, float_features):
    """
    Plot distribution plots for all float features.
    
    Args:
        df (pd.DataFrame): Input dataset
        float_features (list): List of float feature column names
    """
    for feature in float_features:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[feature], kde=True)
        plt.title(f'PDF of {feature}')
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


def calculate_skewness(df, float_features):
    """
    Calculate and print skewness for all float features.
    
    Args:
        df (pd.DataFrame): Input dataset
        float_features (list): List of float feature column names
    """
    for feature in float_features:
        skew = df[feature].skew()
        print(f'The skewness of {feature} is {skew}')


def plot_boxplots(df, float_features):
    """
    Plot boxplots for all float features.
    
    Args:
        df (pd.DataFrame): Input dataset
        float_features (list): List of float feature column names
    """
    for feature in float_features:
        plt.figure(figsize=(6, 4))
        sns.boxplot(df[feature])
        plt.title(f'Boxplot of {feature}')
        plt.xlabel(feature)
        plt.show()


def plot_pairplot(df, float_features, hue_column='Churn'):
    """
    Create pairplot for float features with hue.
    
    Args:
        df (pd.DataFrame): Input dataset
        float_features (list): List of float feature column names
        hue_column (str): Column to use for hue in pairplot
    """
    df_float = df[float_features + [hue_column]].dropna()
    sns.pairplot(df_float, hue=hue_column)


def plot_crosstabs(df, object_features):
    """
    Plot crosstab heatmaps for all pairs of object features.
    
    Args:
        df (pd.DataFrame): Input dataset
        object_features (list): List of object feature column names
    """
    for i, j in combinations(object_features, 2):
        plt.figure(figsize=(20, 4))
        crosstab = pd.crosstab(df[i], df[j])
        print(f"\nCrosstab between {i} and {j}:\n")
        print(crosstab)
        sns.heatmap(crosstab)
        plt.title(f'Crosstab between {i} and {j}')
        plt.show()


def plot_correlation_matrix(df, columns=None):
    """
    Plot correlation matrix for specified columns.
    
    Args:
        df (pd.DataFrame): Input dataset
        columns (list, optional): List of columns to include in correlation matrix
    """
    if columns is None:
        columns = ['MonthlyCharges', 'ServiceUsage', 'Age', 'TotalTransactions', 'Churn']
    
    corr_matrix = df[columns].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()
    
    return corr_matrix


def plot_specific_crosstab(df, col1, col2):
    """
    Plot crosstab heatmap for two specific columns.
    
    Args:
        df (pd.DataFrame): Input dataset
        col1 (str): First column name
        col2 (str): Second column name
    """
    plt.figure(figsize=(8, 6))
    crosstab = pd.crosstab(df[col1], df[col2])
    sns.heatmap(crosstab, annot=True, fmt='d')
    plt.title(f'Crosstab between {col1} and {col2}')
    plt.show()
    
    return crosstab


def analyze_feature_distributions(df, float_features):
    """
    Comprehensive analysis of feature distributions.
    
    Args:
        df (pd.DataFrame): Input dataset
        float_features (list): List of float feature column names
    """
    print("=== Feature Distribution Analysis ===")
    
    # Plot distributions
    plot_distributions(df, float_features)
    
    # Calculate skewness
    calculate_skewness(df, float_features)
    
    # Plot boxplots
    plot_boxplots(df, float_features)
    
    # Plot pairplot
    plot_pairplot(df, float_features)


def analyze_categorical_relationships(df, object_features):
    """
    Analyze relationships between categorical features.
    
    Args:
        df (pd.DataFrame): Input dataset
        object_features (list): List of object feature column names
    """
    print("=== Categorical Feature Analysis ===")
    
    # Plot crosstabs
    plot_crosstabs(df, object_features)
    
    # Specific crosstab for SubscriptionType and Gender
    plot_specific_crosstab(df, 'SubscriptionType', 'Gender')


def comprehensive_eda(df):
    """
    Perform comprehensive exploratory data analysis.
    
    Args:
        df (pd.DataFrame): Input dataset
    """
    print("=== COMPREHENSIVE EDA ===")
    
    # Get feature types
    float_features = df.select_dtypes(include='float').columns.tolist()
    object_features = df.select_dtypes(include='object').columns.tolist()
    
    # Analyze distributions
    analyze_feature_distributions(df, float_features)
    
    # Analyze categorical relationships
    analyze_categorical_relationships(df, object_features)
    
    # Plot correlation matrix
    plot_correlation_matrix(df)
    
    print("=== EDA Complete ===")


