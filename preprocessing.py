"""
Medical Insurance Cost Prediction - Data Preprocessing
This script cleans and preprocesses the insurance dataset.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# STEP 1: Load the Dataset
# ============================================================
print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('data/insurance.csv')

# Display basic information about the dataset
print(f"\nDataset Shape: {df.shape}")
print(f"Number of Rows: {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")

# ============================================================
# STEP 2: Check First Few Rows
# ============================================================
print("\n" + "=" * 60)
print("FIRST 5 ROWS OF DATASET")
print("=" * 60)
print(df.head())

# ============================================================
# STEP 3: Check Data Types
# ============================================================
print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

# ============================================================
# STEP 4: Check for Missing Values
# ============================================================
print("\n" + "=" * 60)
print("MISSING VALUES CHECK")
print("=" * 60)

# Count missing values in each column
missing_values = df.isnull().sum()
print(missing_values)

# Check if there are any missing values
if missing_values.sum() == 0:
    print("\n[OK] No missing values found! Dataset is clean.")
else:
    print(f"\n✗ Found {missing_values.sum()} missing values!")

# ============================================================
# STEP 5: Check for Duplicate Rows
# ============================================================
print("\n" + "=" * 60)
print("DUPLICATE ROWS CHECK")
print("=" * 60)

# Count duplicate rows
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

if duplicates > 0:
    print(f"\nRemoving {duplicates} duplicate rows...")
    df = df.drop_duplicates()
    print(f"New dataset shape: {df.shape}")
else:
    print("[OK] No duplicate rows found!")

# ============================================================
# STEP 6: Check for Outliers using IQR Method
# ============================================================
print("\n" + "=" * 60)
print("OUTLIER DETECTION (IQR Method)")
print("=" * 60)

# Function to detect outliers using IQR
def detect_outliers_iqr(data, column):
    """Detect outliers using Interquartile Range (IQR) method"""
    Q1 = data[column].quantile(0.25)  # 25th percentile
    Q3 = data[column].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Interquartile Range

    # Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    return len(outliers), lower_bound, upper_bound

# Check outliers in numerical columns
numerical_cols = ['age', 'bmi', 'children', 'charges']

for col in numerical_cols:
    count, lower, upper = detect_outliers_iqr(df, col)
    print(f"\n{col}:")
    print(f"  Lower Bound: {lower:.2f}")
    print(f"  Upper Bound: {upper:.2f}")
    print(f"  Outliers Found: {count}")

# ============================================================
# STEP 7: Statistical Summary
# ============================================================
print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())

# ============================================================
# STEP 8: Check Unique Values in Categorical Columns
# ============================================================
print("\n" + "=" * 60)
print("UNIQUE VALUES IN CATEGORICAL COLUMNS")
print("=" * 60)

categorical_cols = ['sex', 'smoker', 'region']

for col in categorical_cols:
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].unique()}")
    print(f"  Count: {df[col].value_counts().to_dict()}")

# ============================================================
# STEP 9: Save Cleaned Data
# ============================================================
print("\n" + "=" * 60)
print("SAVING CLEANED DATA")
print("=" * 60)

# Save the cleaned dataset
df.to_csv('data/insurance_cleaned.csv', index=False)
print("[OK] Cleaned data saved to 'data/insurance_cleaned.csv'")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE!")
print("=" * 60)
print(f"Final Dataset Shape: {df.shape}")
print("Ready for Exploratory Data Analysis (EDA)!")
