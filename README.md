# Hotel Customer Churn Analysis

A comprehensive data science pipeline for analyzing hotel customer churn data, including exploratory data analysis, feature engineering, and data preprocessing.

## 📁 Project Structure

```
Hotel_V1/
├── main.py                    # Main pipeline script
├── data_loader.py            # Data loading and preprocessing
├── eda.py                    # Exploratory data analysis
├── feature_engineering.py    # Feature engineering and encoding
├── data_transformation.py    # Data scaling and transformation
├── outlier_detection.py      # Outlier detection and handling
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── notebook.py               # Original notebook (for reference)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Required packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Hotel
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your data file:
   - Ensure `customer_churn_data_more_missing.csv` is in the project directory

### Running the Pipeline

#### Complete Pipeline
```bash
python main.py
```

#### Quick Analysis
```bash
python -c "from main import run_quick_analysis; run_quick_analysis()"
```

## 📊 Features

### Data Loading & Preprocessing (`data_loader.py`)
- Load CSV data with proper indexing
- Handle missing values using mode imputation
- Convert date columns and extract temporal features
- Split data into training and testing sets

### Exploratory Data Analysis (`eda.py`)
- Distribution analysis for numerical features
- Skewness calculation and visualization
- Box plot analysis for outlier detection
- Pair plots for feature relationships
- Crosstab analysis for categorical features
- Correlation matrix visualization

### Feature Engineering (`feature_engineering.py`)
- Ordinal encoding for subscription types
- One-hot encoding for categorical variables
- Column transformer for mixed data types
- Function transformations (log, sqrt)
- KNN imputation for missing values
- Date feature extraction

### Data Transformation (`data_transformation.py`)
- Standard scaling
- Min-max scaling
- Robust scaling
- Principal Component Analysis (PCA)
- Log and square root transformations
- Comprehensive transformation pipeline

### Outlier Detection (`outlier_detection.py`)
- Z-score method
- Interquartile Range (IQR) method
- Isolation Forest algorithm
- Local Outlier Factor (LOF)
- Outlier capping and removal options
- Comprehensive outlier analysis

## 🔧 Usage Examples

### Basic Data Loading
```python
from data_loader import load_data, get_data_info

# Load data
df = load_data('customer_churn_data_more_missing.csv')

# Get basic information
info = get_data_info(df)
print(f"Dataset shape: {info['shape']}")
```

### Exploratory Data Analysis
```python
from eda import comprehensive_eda

# Run complete EDA
comprehensive_eda(df)
```

### Feature Engineering
```python
from feature_engineering import comprehensive_feature_engineering

# Apply feature engineering
X_train, X_test = prepare_features_target(df)
results = comprehensive_feature_engineering(X_train, X_test)
```

### Outlier Detection
```python
from outlier_detection import comprehensive_outlier_analysis

# Detect outliers using multiple methods
outlier_results = comprehensive_outlier_analysis(
    X_train, 
    ['MonthlyCharges', 'ServiceUsage', 'Age', 'TotalTransactions'],
    methods=['zscore', 'iqr', 'isolation_forest']
)
```

## 📈 Pipeline Output

The pipeline generates several outputs:

1. **Processed Data Files:**
   - `X_train_processed.csv` - Processed training features
   - `X_test_processed.csv` - Processed test features
   - `y_train.csv` - Training target variable
   - `y_test.csv` - Test target variable

2. **Visualizations:**
   - Distribution plots for all numerical features
   - Box plots for outlier analysis
   - Pair plots for feature relationships
   - Correlation matrices
   - PCA variance plots

3. **Analysis Reports:**
   - Missing value analysis
   - Outlier detection summary
   - Feature transformation results
   - Data quality metrics

## 🛠️ Customization

### Adding New Features
1. Modify `feature_engineering.py` to add new feature creation functions
2. Update the main pipeline in `main.py` to include new features

### Changing Outlier Detection Methods
1. Add new methods to `outlier_detection.py`
2. Update the `comprehensive_outlier_analysis` function
3. Modify the main pipeline to use new methods

### Custom Transformations
1. Add new transformation functions to `data_transformation.py`
2. Update the `comprehensive_transformation_pipeline` function

## 📋 Requirements

- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- scipy >= 1.9.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Support

For questions or issues, please open an issue in the GitHub repository.

## 🔄 Version History

- **v1.0.0** - Initial release with complete data science pipeline
  - Data loading and preprocessing
  - Exploratory data analysis
  - Feature engineering
  - Data transformation
  - Outlier detection
  - Main pipeline script

---

**Note:** This project was created by splitting a Jupyter notebook into modular Python files for better organization and reusability in a GitHub repository.
