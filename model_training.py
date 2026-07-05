"""
Medical Insurance Cost Prediction - Model Training and Comparison
This script trains multiple ML models and compares their performance.

Models Used:
1. Linear Regression - Simple, interpretable model
2. Decision Tree Regressor - Tree-based model
3. Random Forest Regressor - Ensemble of multiple trees

Author: Krutika Mohanty
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MEDICAL INSURANCE COST PREDICTION - MODEL TRAINING")
print("=" * 70)

# ============================================================
# STEP 1: LOAD TRAINING AND TESTING DATA
# ============================================================
print("\n[STEP 1] Loading training and testing data...")

# Load the saved train/test sets
X_train = joblib.load('models/X_train.pkl')
y_train = joblib.load('models/y_train.pkl')
X_test = joblib.load('models/X_test.pkl')
y_test = joblib.load('models/y_test.pkl')

print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Testing set: {X_test.shape[0]} samples")
print(f"  Features: {X_train.shape[1]}")

# ============================================================
# STEP 2: DEFINE MODELS
# ============================================================
print("\n[STEP 2] Defining machine learning models...")

# Create a dictionary of models to train
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

print("  Models defined:")
for name in models.keys():
    print(f"    - {name}")

# ============================================================
# STEP 3: TRAIN AND EVALUATE MODELS
# ============================================================
print("\n[STEP 3] Training and evaluating models...")

# Dictionary to store results
results = {}

# Train each model and calculate metrics
for name, model in models.items():
    print(f"\n  Training {name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)       # Mean Absolute Error
    mse = mean_squared_error(y_test, y_pred)         # Mean Squared Error
    rmse = np.sqrt(mse)                              # Root Mean Squared Error
    r2 = r2_score(y_test, y_pred)                    # R-squared Score

    # Store results
    results[name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'predictions': y_pred
    }

    print(f"    MAE:  ${mae:,.2f}")
    print(f"    RMSE: ${rmse:,.2f}")
    print(f"    R2:   {r2:.4f}")

# ============================================================
# STEP 4: COMPARE MODEL PERFORMANCE
# ============================================================
print("\n" + "=" * 70)
print("[STEP 4] MODEL COMPARISON")
print("=" * 70)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'MAE': [results[m]['MAE'] for m in results],
    'RMSE': [results[m]['RMSE'] for m in results],
    'R2 Score': [results[m]['R2'] for m in results]
})

print("\n" + comparison_df.to_string(index=False))

# Find best model based on R2 Score
best_model_name = comparison_df.loc[comparison_df['R2 Score'].idxmax(), 'Model']
best_r2 = comparison_df['R2 Score'].max()

print(f"\n  BEST MODEL: {best_model_name}")
print(f"  R2 Score: {best_r2:.4f}")

# ============================================================
# STEP 5: VISUALIZE MODEL COMPARISON
# ============================================================
print("\n[STEP 5] Creating visualization charts...")

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

# Plot 1: R2 Score Comparison
plt.subplot(2, 2, 1)
colors = ['#3498DB', '#E74C3C', '#2ECC71']
bars = plt.bar(comparison_df['Model'], comparison_df['R2 Score'], color=colors, edgecolor='black')
plt.title('R2 Score Comparison', fontsize=14, fontweight='bold')
plt.ylabel('R2 Score')
plt.ylim(0, 1)
for bar, score in zip(bars, comparison_df['R2 Score']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.4f}', ha='center', fontweight='bold')

# Plot 2: MAE Comparison
plt.subplot(2, 2, 2)
bars = plt.bar(comparison_df['Model'], comparison_df['MAE'], color=colors, edgecolor='black')
plt.title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
plt.ylabel('MAE ($)')
for bar, mae in zip(bars, comparison_df['MAE']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             f'${mae:,.0f}', ha='center', fontweight='bold')

# Plot 3: RMSE Comparison
plt.subplot(2, 2, 3)
bars = plt.bar(comparison_df['Model'], comparison_df['RMSE'], color=colors, edgecolor='black')
plt.title('Root Mean Squared Error Comparison', fontsize=14, fontweight='bold')
plt.ylabel('RMSE ($)')
for bar, rmse in zip(bars, comparison_df['RMSE']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             f'${rmse:,.0f}', ha='center', fontweight='bold')

# Plot 4: Actual vs Predicted (Best Model)
plt.subplot(2, 2, 4)
best_predictions = results[best_model_name]['predictions']
plt.scatter(y_test, best_predictions, alpha=0.5, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Charges ($)')
plt.ylabel('Predicted Charges ($)')
plt.title(f'Actual vs Predicted ({best_model_name})', fontsize=14, fontweight='bold')
plt.legend()

plt.tight_layout()
plt.savefig('static/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: static/model_comparison.png")

# ============================================================
# STEP 6: SAVE THE BEST MODEL
# ============================================================
print("\n[STEP 6] Saving the best model...")

# Get the best model
best_model = models[best_model_name]

# Save the model using joblib
joblib.dump(best_model, 'models/insurance_model.pkl')
print(f"  Model saved: models/insurance_model.pkl")
print(f"  Model type: {best_model_name}")

# ============================================================
# STEP 7: CREATE RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("[STEP 7] TRAINING COMPLETE - SUMMARY")
print("=" * 70)

print(f"""
MODEL PERFORMANCE SUMMARY:
-------------------------
1. Linear Regression:
   - MAE:  ${results['Linear Regression']['MAE']:,.2f}
   - RMSE: ${results['Linear Regression']['RMSE']:,.2f}
   - R2:   {results['Linear Regression']['R2']:.4f}

2. Decision Tree:
   - MAE:  ${results['Decision Tree']['MAE']:,.2f}
   - RMSE: ${results['Decision Tree']['RMSE']:,.2f}
   - R2:   {results['Decision Tree']['R2']:.4f}

3. Random Forest:
   - MAE:  ${results['Random Forest']['MAE']:,.2f}
   - RMSE: ${results['Random Forest']['RMSE']:,.2f}
   - R2:   {results['Random Forest']['R2']:.4f}

BEST MODEL: {best_model_name}
R2 SCORE: {best_r2:.4f}
""")

print("FILES CREATED:")
print("  1. models/insurance_model.pkl - Best trained model")
print("  2. static/model_comparison.png - Comparison chart")

print("\nReady for Flask Web Application!")
