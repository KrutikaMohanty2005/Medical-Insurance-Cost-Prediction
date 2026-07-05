"""
Generate additional screenshots and visualizations for the project.
Creates enhanced EDA plots and feature importance charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("=" * 60)
print("GENERATING ADDITIONAL SCREENSHOTS")
print("=" * 60)

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv('data/insurance_cleaned.csv')
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# 1. Feature Importance Chart
# ============================================================
print("\n[1] Creating Feature Importance chart...")

# Load model and get feature importance
model = joblib.load('models/insurance_model.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

# Get feature importances from Random Forest
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_names = feature_columns
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    colors = ['#2ECC71' if i == indices[0] else '#3498DB' for i in range(len(indices))]
    bars = plt.bar(range(len(importances)), importances[indices], color=colors, edgecolor='black')
    plt.title('Feature Importance (Random Forest)', fontsize=16, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    
    # Add value labels
    for bar, imp in zip(bars, importances[indices]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{imp:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('static/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: feature_importance.png")

# ============================================================
# 2. BMI vs Charges by Smoker (Enhanced)
# ============================================================
print("\n[2] Creating BMI vs Charges by Smoker (enhanced)...")

plt.figure(figsize=(12, 7))
scatter = sns.scatterplot(x='bmi', y='charges', hue='smoker', style='smoker',
                         data=df, palette=['#2ECC71', '#E74C3C'], s=80, alpha=0.7)

# Add trend lines
for smoker_val, color in zip(['yes', 'no'], ['#E74C3C', '#2ECC71']):
    subset = df[df['smoker'] == smoker_val]
    z = np.polyfit(subset['bmi'], subset['charges'], 1)
    p = np.poly1d(z)
    plt.plot(subset['bmi'], p(subset['bmi']), color=color, linestyle='--', linewidth=2, alpha=0.8)

plt.title('BMI vs Insurance Charges by Smoking Status', fontsize=16, fontweight='bold')
plt.xlabel('BMI', fontsize=12)
plt.ylabel('Charges ($)', fontsize=12)
plt.legend(title='Smoker', fontsize=11)
plt.tight_layout()
plt.savefig('static/bmi_vs_charges_smoker.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: bmi_vs_charges_smoker.png")

# ============================================================
# 3. Age Distribution by Smoker Status
# ============================================================
print("\n[3] Creating Age Distribution by Smoker...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram
for smoker_val, color, label in zip(['yes', 'no'], ['#E74C3C', '#2ECC71'], ['Smoker', 'Non-Smoker']):
    subset = df[df['smoker'] == smoker_val]
    axes[0].hist(subset['age'], bins=25, color=color, edgecolor='black', alpha=0.7, label=label)
axes[0].set_title('Age Distribution by Smoking Status', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Age', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].legend()

# KDE plot
for smoker_val, color, label in zip(['yes', 'no'], ['#E74C3C', '#2ECC71'], ['Smoker', 'Non-Smoker']):
    subset = df[df['smoker'] == smoker_val]
    axes[1].hist(subset['charges'], bins=25, color=color, edgecolor='black', alpha=0.6, label=label, density=True)
    sns.kdeplot(data=subset['charges'], ax=axes[1], color=color, linewidth=2)
axes[1].set_title('Charges Distribution by Smoking Status', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Charges ($)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.savefig('static/age_charges_by_smoker.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: age_charges_by_smoker.png")

# ============================================================
# 4. Actual vs Predicted (Standalone)
# ============================================================
print("\n[4] Creating Actual vs Predicted chart...")

X_test = joblib.load('models/X_test.pkl')
y_test = joblib.load('models/y_test.pkl')
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='black', linewidths=0.5, s=60)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction (y=x)')
plt.fill_between([y_test.min(), y_test.max()],
                  [y_test.min() - 5000, y_test.max() - 5000],
                  [y_test.min() + 5000, y_test.max() + 5000],
                  alpha=0.1, color='green', label='±$5,000 Range')

r2 = model.score(X_test, y_test)
plt.xlabel('Actual Charges ($)', fontsize=12)
plt.ylabel('Predicted Charges ($)', fontsize=12)
plt.title(f'Actual vs Predicted Insurance Charges (R² = {r2:.4f})', fontsize=16, fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('static/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: actual_vs_predicted.png")

# ============================================================
# 5. Residuals Distribution
# ============================================================
print("\n[5] Creating Residuals Distribution...")

residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Residuals histogram
axes[0].hist(residuals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Residual ($)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)

# Residuals vs Predicted
axes[1].scatter(y_pred, residuals, alpha=0.6, color='steelblue', edgecolors='black', linewidths=0.5)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Charges ($)', fontsize=12)
axes[1].set_ylabel('Residual ($)', fontsize=12)

plt.tight_layout()
plt.savefig('static/residuals_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: residuals_analysis.png")

# ============================================================
# 6. Charges Distribution by Region and Smoker
# ============================================================
print("\n[6] Creating Charges by Region and Smoker...")

plt.figure(figsize=(14, 7))
sns.barplot(x='region', y='charges', hue='smoker', data=df,
            palette=['#2ECC71', '#E74C3C'], ci=95, edgecolor='black')
plt.title('Average Insurance Charges by Region and Smoking Status', fontsize=16, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Average Charges ($)', fontsize=12)
plt.legend(title='Smoker', fontsize=11)

# Add value labels
ax = plt.gca()
for container in ax.containers:
    ax.bar_label(container, fmt='$%,.0f', fontsize=9, padding=3)

plt.tight_layout()
plt.savefig('static/charges_region_smoker.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: charges_region_smoker.png")

# ============================================================
# 7. Children vs Charges
# ============================================================
print("\n[7] Creating Children vs Charges...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot
sns.boxplot(x='children', y='charges', data=df, palette='Set2', ax=axes[0])
axes[0].set_title('Insurance Charges by Number of Children', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Children', fontsize=12)
axes[0].set_ylabel('Charges ($)', fontsize=12)

# Bar plot with mean
children_mean = df.groupby('children')['charges'].mean().sort_index()
bars = axes[1].bar(children_mean.index.astype(str), children_mean.values, 
                   color=sns.color_palette('Set2', len(children_mean)), edgecolor='black')
axes[1].set_title('Average Charges by Number of Children', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Number of Children', fontsize=12)
axes[1].set_ylabel('Average Charges ($)', fontsize=12)

for bar, val in zip(bars, children_mean.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'${val:,.0f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('static/charges_by_children.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: charges_by_children.png")

# ============================================================
# 8. Correlation Heatmap (Enhanced)
# ============================================================
print("\n[8] Creating Enhanced Correlation Heatmap...")

# Encode categorical variables for full correlation
df_encoded = df.copy()
df_encoded['sex_encoded'] = (df_encoded['sex'] == 'male').astype(int)
df_encoded['smoker_encoded'] = (df_encoded['smoker'] == 'yes').astype(int)
df_encoded['region_encoded'] = df_encoded['region'].map({'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3})

plt.figure(figsize=(10, 8))
corr_cols = ['age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded', 'region_encoded', 'charges']
corr_matrix = df_encoded[corr_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
            mask=mask, square=True, linewidths=1, fmt='.2f',
            cbar_kws={"shrink": 0.8, "label": "Correlation"})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('static/correlation_heatmap_enhanced.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: correlation_heatmap_enhanced.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SCREENSHOT GENERATION COMPLETE!")
print("=" * 60)
print("\nNew screenshots saved to 'static/':")
print("  1. feature_importance.png")
print("  2. bmi_vs_charges_smoker.png")
print("  3. age_charges_by_smoker.png")
print("  4. actual_vs_predicted.png")
print("  5. residuals_analysis.png")
print("  6. charges_region_smoker.png")
print("  7. charges_by_children.png")
print("  8. correlation_heatmap_enhanced.png")
