"""
Medical Insurance Cost Prediction - Enhanced Model Training
Models: Linear Regression, Decision Tree, Random Forest, XGBoost, Neural Network (MLP)
Dataset: Enhanced with medical history & lifestyle features

Author: Krutika Mohanty
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MEDICAL INSURANCE COST PREDICTION - ENHANCED MODEL TRAINING")
print("=" * 70)

# ============================================================
# STEP 1: LOAD ENHANCED DATASET
# ============================================================
print("\n[STEP 1] Loading enhanced dataset...")

df = pd.read_csv('data/insurance_enhanced.csv')
print(f"  Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("\n[STEP 2] Feature Engineering...")

# Encode categorical variables
df_encoded = df.copy()

# Label encode binary variables
le_sex = LabelEncoder()
df_encoded['sex_encoded'] = le_sex.fit_transform(df_encoded['sex'])

le_smoker = LabelEncoder()
df_encoded['smoker_encoded'] = le_smoker.fit_transform(df_encoded['smoker'])

# One-hot encode region
region_dummies = pd.get_dummies(df_encoded['region'], prefix='region', dtype=int)
df_encoded = pd.concat([df_encoded, region_dummies], axis=1)

# One-hot encode BMI category
bmi_dummies = pd.get_dummies(df_encoded['bmi_category'], prefix='bmi_cat', dtype=int)
df_encoded = pd.concat([df_encoded, bmi_dummies], axis=1)

# Define features
feature_columns = [
    'age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded',
    'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest',
    'hospitalizations', 'chronic_diseases', 'pre_existing', 'family_history', 'surgeries',
    'exercise_freq', 'alcohol', 'smoking_years', 'diet_quality', 'stress_level',
    'sleep_hours', 'water_intake', 'sugar_intake', 'health_risk_score',
    'bmi_cat_normal', 'bmi_cat_obese', 'bmi_cat_overweight', 'bmi_cat_underweight'
]

X = df_encoded[feature_columns]
y = df_encoded['charges']

print(f"  Features: {X.shape[1]} columns")
print(f"  Target: charges")

# ============================================================
# STEP 3: TRAIN/TEST SPLIT & SCALING
# ============================================================
print("\n[STEP 3] Train/Test Split & Scaling...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Testing set: {X_test.shape[0]} samples")

# ============================================================
# STEP 4: DEFINE MODELS
# ============================================================
print("\n[STEP 4] Defining models...")

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=5),
    'Neural Network (MLP)': MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
}

# ============================================================
# STEP 5: TRAIN & EVALUATE
# ============================================================
print("\n[STEP 5] Training and evaluating models...")

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")

    # Use scaled data for Neural Network, original for tree-based
    if name == 'Neural Network (MLP)':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    elif name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2,
        'predictions': y_pred
    }

    print(f"    MAE:  ${mae:,.2f}")
    print(f"    RMSE: ${rmse:,.2f}")
    print(f"    R2:   {r2:.4f}")

# ============================================================
# STEP 6: MODEL COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("[STEP 6] MODEL COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'MAE': [results[m]['MAE'] for m in results],
    'RMSE': [results[m]['RMSE'] for m in results],
    'R2 Score': [results[m]['R2'] for m in results]
})

print("\n" + comparison_df.to_string(index=False))

best_model_name = comparison_df.loc[comparison_df['R2 Score'].idxmax(), 'Model']
best_r2 = comparison_df['R2 Score'].max()

print(f"\n  BEST MODEL: {best_model_name}")
print(f"  R2 Score: {best_r2:.4f}")

# ============================================================
# STEP 7: VISUALIZATION
# ============================================================
print("\n[STEP 7] Creating visualization charts...")

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

colors = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6', '#F39C12']

# Plot 1: R2 Score Comparison
axes[0, 0].bar(comparison_df['Model'], comparison_df['R2 Score'], color=colors, edgecolor='black')
axes[0, 0].set_title('R2 Score Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('R2 Score')
axes[0, 0].set_ylim(0, 1)
for i, v in enumerate(comparison_df['R2 Score']):
    axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold', fontsize=9)
axes[0, 0].tick_params(axis='x', rotation=30)

# Plot 2: MAE Comparison
axes[0, 1].bar(comparison_df['Model'], comparison_df['MAE'], color=colors, edgecolor='black')
axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('MAE ($)')
for i, v in enumerate(comparison_df['MAE']):
    axes[0, 1].text(i, v + 100, f'${v:,.0f}', ha='center', fontsize=9, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=30)

# Plot 3: RMSE Comparison
axes[0, 2].bar(comparison_df['Model'], comparison_df['RMSE'], color=colors, edgecolor='black')
axes[0, 2].set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
axes[0, 2].set_ylabel('RMSE ($)')
for i, v in enumerate(comparison_df['RMSE']):
    axes[0, 2].text(i, v + 100, f'${v:,.0f}', ha='center', fontsize=9, fontweight='bold')
axes[0, 2].tick_params(axis='x', rotation=30)

# Plot 4: Actual vs Predicted (Best Model)
best_predictions = results[best_model_name]['predictions']
axes[1, 0].scatter(y_test, best_predictions, alpha=0.5, color='steelblue', edgecolors='black', linewidths=0.5)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[1, 0].set_xlabel('Actual Charges ($)')
axes[1, 0].set_ylabel('Predicted Charges ($)')
axes[1, 0].set_title(f'Actual vs Predicted ({best_model_name})', fontsize=14, fontweight='bold')

# Plot 5: Residuals Distribution
residuals = y_test - best_predictions
axes[1, 1].hist(residuals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Residual ($)')

# Plot 6: Feature Importance (if available)
if hasattr(models[best_model_name], 'feature_importances_'):
    importances = models[best_model_name].feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    axes[1, 2].barh(range(10), importances[indices][::-1], color='steelblue', edgecolor='black')
    axes[1, 2].set_yticks(range(10))
    axes[1, 2].set_yticklabels([feature_columns[i] for i in indices][::-1])
    axes[1, 2].set_title(f'Top 10 Feature Importance ({best_model_name})', fontsize=14, fontweight='bold')
else:
    # For MLP, show loss curve
    if hasattr(models[best_model_name], 'loss_curve_'):
        axes[1, 2].plot(models[best_model_name].loss_curve_, color='steelblue', linewidth=2)
        axes[1, 2].set_title('Neural Network Training Loss', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')

plt.tight_layout()
plt.savefig('screenshots/model_comparison_enhanced.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: screenshots/model_comparison_enhanced.png")

# ============================================================
# STEP 8: SAVE BEST MODEL
# ============================================================
print("\n[STEP 8] Saving the best model...")

best_model = models[best_model_name]

# For neural network, save with scaler
if best_model_name == 'Neural Network (MLP)':
    joblib.dump(best_model, 'models/insurance_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
else:
    # Retrain best model on full data for production
    best_model.fit(X, y)
    joblib.dump(best_model, 'models/insurance_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

joblib.dump(feature_columns, 'models/feature_columns.pkl')

# Save all model comparison data
model_metrics = {
    'best_model': best_model_name,
    'models': {}
}
for name in results:
    model_metrics['models'][name] = {
        'MAE': float(results[name]['MAE']),
        'RMSE': float(results[name]['RMSE']),
        'R2': float(results[name]['R2'])
    }

import json
with open('models/model_metrics.json', 'w') as f:
    json.dump(model_metrics, f, indent=2)

print(f"  Best model saved: models/insurance_model.pkl")
print(f"  Metrics saved: models/model_metrics.json")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE - ENHANCED MODEL SUMMARY")
print("=" * 70)

for name in results:
    print(f"\n  {name}:")
    print(f"    MAE:  ${results[name]['MAE']:,.2f}")
    print(f"    RMSE: ${results[name]['RMSE']:,.2f}")
    print(f"    R2:   {results[name]['R2']:.4f}")

print(f"\n  BEST MODEL: {best_model_name}")
print(f"  R2 SCORE: {best_r2:.4f}")
print("\nReady for deployment!")
