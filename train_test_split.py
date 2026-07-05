"""
Medical Insurance Cost Prediction - Train/Test Split
This script splits the dataset into training and testing sets.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# ============================================================
# STEP 1: Load the Encoded Dataset
# ============================================================
print("=" * 60)
print("STEP 1: LOADING ENCODED DATASET")
print("=" * 60)

# Load encoded dataset
df = pd.read_csv('data/insurance_encoded.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Load saved feature columns
feature_columns = joblib.load('models/feature_columns.pkl')
print(f"\nFeature columns: {feature_columns}")

# ============================================================
# STEP 2: Define Features (X) and Target (y)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: DEFINING FEATURES AND TARGET")
print("=" * 60)

# X = Features (input to model)
X = df[feature_columns]

# y = Target (what we want to predict)
y = df['charges']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

print("\nFirst 5 rows of X:")
print(X.head())

print("\nFirst 5 values of y:")
print(y.head())

# ============================================================
# STEP 3: Split Data into Train and Test Sets
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: SPLITTING DATA (80% Train, 20% Test)")
print("=" * 60)

# Split the data
# test_size=0.2 means 20% for testing, 80% for training
# random_state=42 ensures reproducibility (same split every time)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% test data
    random_state=42      # For reproducibility
)

print(f"\nTraining Set:")
print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  Percentage: {len(X_train)/len(X)*100:.1f}%")

print(f"\nTesting Set:")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_test shape: {y_test.shape}")
print(f"  Percentage: {len(X_test)/len(X)*100:.1f}%")

# ============================================================
# STEP 4: Verify the Split
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: VERIFYING THE SPLIT")
print("=" * 60)

print("\nTraining set statistics:")
print(f"  Age range: {X_train['age'].min()} - {X_train['age'].max()}")
print(f"  BMI range: {X_train['bmi'].min():.1f} - {X_train['bmi'].max():.1f}")
print(f"  Charges range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
print(f"  Average charges: ${y_train.mean():,.0f}")

print("\nTesting set statistics:")
print(f"  Age range: {X_test['age'].min()} - {X_test['age'].max()}")
print(f"  BMI range: {X_test['bmi'].min():.1f} - {X_test['bmi'].max():.1f}")
print(f"  Charges range: ${y_test.min():,.0f} - ${y_test.max():,.0f}")
print(f"  Average charges: ${y_test.mean():,.0f}")

# ============================================================
# STEP 5: Save Train/Test Sets
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: SAVING TRAIN/TEST SETS")
print("=" * 60)

# Save training data
joblib.dump(X_train, 'models/X_train.pkl')
joblib.dump(y_train, 'models/y_train.pkl')
print("[OK] Training data saved")

# Save testing data
joblib.dump(X_test, 'models/X_test.pkl')
joblib.dump(y_test, 'models/y_test.pkl')
print("[OK] Testing data saved")

# ============================================================
# STEP 6: Summary
# ============================================================
print("\n" + "=" * 60)
print("TRAIN/TEST SPLIT COMPLETE!")
print("=" * 60)

print(f"\nTotal Dataset: {len(X)} samples")
print(f"Training Set:  {len(X_train)} samples (80%)")
print(f"Testing Set:   {len(X_test)} samples (20%)")

print("\nFiles Created:")
print("  1. models/X_train.pkl - Training features")
print("  2. models/y_train.pkl - Training target")
print("  3. models/X_test.pkl  - Testing features")
print("  4. models/y_test.pkl  - Testing target")

print("\nReady for Model Training!")
