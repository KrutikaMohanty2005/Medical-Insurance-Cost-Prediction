"""
Medical Insurance Cost Prediction - Feature Engineering
This script encodes categorical variables and prepares data for ML models.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# ============================================================
# STEP 1: Load the Cleaned Dataset
# ============================================================
print("=" * 60)
print("STEP 1: LOADING CLEANED DATASET")
print("=" * 60)

df = pd.read_csv('data/insurance_cleaned.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# ============================================================
# STEP 2: Encode Binary Categorical Variables (Label Encoding)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: LABEL ENCODING (Binary Variables)")
print("=" * 60)

# Create a copy for encoding
df_encoded = df.copy()

# Initialize Label Encoders (separate instance for each column)
le_sex = LabelEncoder()
le_smoker = LabelEncoder()

# Encode 'sex' column: female=0, male=1
df_encoded['sex_encoded'] = le_sex.fit_transform(df_encoded['sex'])
print("\nSex Encoding:")
print("  female -> 0")
print("  male -> 1")

# Encode 'smoker' column: no=0, yes=1
df_encoded['smoker_encoded'] = le_smoker.fit_transform(df_encoded['smoker'])
print("\nSmoker Encoding:")
print("  no -> 0")
print("  yes -> 1")

# Show encoded values
print("\nEncoded values (first 5 rows):")
print(df_encoded[['sex', 'sex_encoded', 'smoker', 'smoker_encoded']].head())

# ============================================================
# STEP 3: One-Hot Encoding for Region
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: ONE-HOT ENCODING (Region)")
print("=" * 60)

# One-Hot Encode 'region' column
# This creates 4 new columns: region_northeast, region_northwest, region_southeast, region_southwest
region_dummies = pd.get_dummies(df_encoded['region'], prefix='region', dtype=int)

# Add dummy columns to dataframe
df_encoded = pd.concat([df_encoded, region_dummies], axis=1)

print("\nRegion Encoding:")
print("  region_northeast = 1 if northeast, else 0")
print("  region_northwest = 1 if northwest, else 0")
print("  region_southeast = 1 if southeast, else 0")
print("  region_southwest = 1 if southwest, else 0")

# Show encoded region values
print("\nRegion encoded values (first 5 rows):")
print(df_encoded[['region', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']].head())

# ============================================================
# STEP 4: Select Final Features for Model
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: SELECTING FINAL FEATURES")
print("=" * 60)

# Define feature columns (X) and target variable (y)
feature_columns = [
    'age',                    # Age of person
    'sex_encoded',           # Gender (0=female, 1=male)
    'bmi',                   # Body Mass Index
    'children',              # Number of children
    'smoker_encoded',        # Smoker status (0=no, 1=yes)
    'region_northeast',      # Region indicator
    'region_northwest',      # Region indicator
    'region_southeast',      # Region indicator
    'region_southwest'       # Region indicator
]

target_column = 'charges'   # What we want to predict

# Create X (features) and y (target)
X = df_encoded[feature_columns]
y = df_encoded[target_column]

print("\nFeature columns (X):")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i}. {col}")

print(f"\nTarget column (y): {target_column}")
print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

# ============================================================
# STEP 5: Feature Scaling (Standardization)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: FEATURE SCALING (Standardization)")
print("=" * 60)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for readability
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

print("\nFeatures scaled using StandardScaler")
print("This ensures all features have mean=0 and std=1")
print("\nScaled features (first 5 rows):")
print(X_scaled_df.head())

# ============================================================
# STEP 6: Save Encoded Data and Scaler
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: SAVING ENCODED DATA AND SCALER")
print("=" * 60)

# Save the encoded dataset
df_encoded.to_csv('data/insurance_encoded.csv', index=False)
print("[OK] Encoded dataset saved to 'data/insurance_encoded.csv'")

# Save the scaler for later use in predictions
joblib.dump(scaler, 'models/scaler.pkl')
print("[OK] Scaler saved to 'models/scaler.pkl'")

# Save feature columns list
joblib.dump(feature_columns, 'models/feature_columns.pkl')
print("[OK] Feature columns list saved to 'models/feature_columns.pkl'")

# Save label encoders
joblib.dump({'sex': le_sex, 'smoker': le_smoker}, 'models/label_encoder.pkl')
print("[OK] Label encoders saved to 'models/label_encoder.pkl'")

# ============================================================
# STEP 7: Summary
# ============================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 60)

print(f"\nFinal Features: {len(feature_columns)} columns")
print(f"Target Variable: {target_column}")
print(f"Total Samples: {len(X)}")

print("\nFiles Created:")
print("  1. data/insurance_encoded.csv - Full encoded dataset")
print("  2. models/scaler.pkl - Scaler for predictions")
print("  3. models/feature_columns.pkl - Feature names list")
print("  4. models/label_encoder.pkl - Label encoder for categories")

print("\nReady for Model Training!")
