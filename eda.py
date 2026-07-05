"""
Medical Insurance Cost Prediction - Exploratory Data Analysis (EDA)
This script creates visualizations to understand data patterns.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better looking graphs
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# ============================================================
# STEP 1: Load the Cleaned Dataset
# ============================================================
print("Loading cleaned dataset...")
df = pd.read_csv('data/insurance_cleaned.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# STEP 2: Distribution of Age
# ============================================================
print("\nCreating Age Distribution plot...")

plt.figure(figsize=(10, 6))
plt.hist(df['age'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Age', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.axvline(df['age'].mean(), color='red', linestyle='--', label=f"Mean: {df['age'].mean():.1f}")
plt.axvline(df['age'].median(), color='green', linestyle='--', label=f"Median: {df['age'].median():.1f}")
plt.legend()
plt.tight_layout()
plt.savefig('static/age_distribution.png')
plt.close()
print("  Saved: age_distribution.png")

# ============================================================
# STEP 3: Distribution of BMI
# ============================================================
print("Creating BMI Distribution plot...")

plt.figure(figsize=(10, 6))
plt.hist(df['bmi'], bins=30, color='coral', edgecolor='black', alpha=0.7)
plt.title('Distribution of BMI', fontsize=16, fontweight='bold')
plt.xlabel('BMI', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.axvline(df['bmi'].mean(), color='red', linestyle='--', label=f"Mean: {df['bmi'].mean():.1f}")
plt.axvline(df['bmi'].median(), color='green', linestyle='--', label=f"Median: {df['bmi'].median():.1f}")
plt.legend()
plt.tight_layout()
plt.savefig('static/bmi_distribution.png')
plt.close()
print("  Saved: bmi_distribution.png")

# ============================================================
# STEP 4: Distribution of Charges (Target Variable)
# ============================================================
print("Creating Charges Distribution plot...")

plt.figure(figsize=(10, 6))
plt.hist(df['charges'], bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
plt.title('Distribution of Insurance Charges', fontsize=16, fontweight='bold')
plt.xlabel('Charges ($)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.axvline(df['charges'].mean(), color='red', linestyle='--', label=f"Mean: ${df['charges'].mean():,.0f}")
plt.axvline(df['charges'].median(), color='green', linestyle='--', label=f"Median: ${df['charges'].median():,.0f}")
plt.legend()
plt.tight_layout()
plt.savefig('static/charges_distribution.png')
plt.close()
print("  Saved: charges_distribution.png")

# ============================================================
# STEP 5: Gender Distribution
# ============================================================
print("Creating Gender Distribution plot...")

plt.figure(figsize=(8, 6))
gender_counts = df['sex'].value_counts()
colors = ['#FF6B6B', '#4ECDC4']
plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, explode=(0.05, 0))
plt.title('Gender Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('static/gender_distribution.png')
plt.close()
print("  Saved: gender_distribution.png")

# ============================================================
# STEP 6: Smoker Distribution
# ============================================================
print("Creating Smoker Distribution plot...")

plt.figure(figsize=(8, 6))
smoker_counts = df['smoker'].value_counts()
colors = ['#2ECC71', '#E74C3C']
plt.pie(smoker_counts.values, labels=smoker_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, explode=(0.05, 0.05))
plt.title('Smoker Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('static/smoker_distribution.png')
plt.close()
print("  Saved: smoker_distribution.png")

# ============================================================
# STEP 7: Region Distribution
# ============================================================
print("Creating Region Distribution plot...")

plt.figure(figsize=(10, 6))
region_counts = df['region'].value_counts()
colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
bars = plt.bar(region_counts.index, region_counts.values, color=colors, edgecolor='black')
plt.title('Region Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Count', fontsize=12)
for bar, count in zip(bars, region_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(count), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('static/region_distribution.png')
plt.close()
print("  Saved: region_distribution.png")

# ============================================================
# STEP 8: Charges by Smoker Status
# ============================================================
print("Creating Charges by Smoker plot...")

plt.figure(figsize=(10, 6))
sns.boxplot(x='smoker', y='charges', data=df, palette=['#2ECC71', '#E74C3C'])
plt.title('Insurance Charges by Smoking Status', fontsize=16, fontweight='bold')
plt.xlabel('Smoker', fontsize=12)
plt.ylabel('Charges ($)', fontsize=12)
plt.tight_layout()
plt.savefig('static/charges_by_smoker.png')
plt.close()
print("  Saved: charges_by_smoker.png")

# ============================================================
# STEP 9: Charges by Gender
# ============================================================
print("Creating Charges by Gender plot...")

plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='charges', data=df, palette=['#FF6B6B', '#4ECDC4'])
plt.title('Insurance Charges by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Charges ($)', fontsize=12)
plt.tight_layout()
plt.savefig('static/charges_by_gender.png')
plt.close()
print("  Saved: charges_by_gender.png")

# ============================================================
# STEP 10: Charges by Region
# ============================================================
print("Creating Charges by Region plot...")

plt.figure(figsize=(12, 6))
sns.boxplot(x='region', y='charges', data=df, palette='Set2')
plt.title('Insurance Charges by Region', fontsize=16, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Charges ($)', fontsize=12)
plt.tight_layout()
plt.savefig('static/charges_by_region.png')
plt.close()
print("  Saved: charges_by_region.png")

# ============================================================
# STEP 11: Age vs Charges (Scatter Plot)
# ============================================================
print("Creating Age vs Charges scatter plot...")

plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='charges', hue='smoker', data=df, palette=['#2ECC71', '#E74C3C'], alpha=0.6)
plt.title('Age vs Insurance Charges', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Charges ($)', fontsize=12)
plt.legend(title='Smoker')
plt.tight_layout()
plt.savefig('static/age_vs_charges.png')
plt.close()
print("  Saved: age_vs_charges.png")

# ============================================================
# STEP 12: BMI vs Charges (Scatter Plot)
# ============================================================
print("Creating BMI vs Charges scatter plot...")

plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df, palette=['#2ECC71', '#E74C3C'], alpha=0.6)
plt.title('BMI vs Insurance Charges', fontsize=16, fontweight='bold')
plt.xlabel('BMI', fontsize=12)
plt.ylabel('Charges ($)', fontsize=12)
plt.legend(title='Smoker')
plt.tight_layout()
plt.savefig('static/bmi_vs_charges.png')
plt.close()
print("  Saved: bmi_vs_charges.png")

# ============================================================
# STEP 13: Correlation Heatmap
# ============================================================
print("Creating Correlation Heatmap...")

plt.figure(figsize=(10, 8))
# Select only numerical columns for correlation
numerical_df = df[['age', 'bmi', 'children', 'charges']]
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('static/correlation_heatmap.png')
plt.close()
print("  Saved: correlation_heatmap.png")

# ============================================================
# STEP 14: Print Key Insights
# ============================================================
print("\n" + "=" * 60)
print("KEY INSIGHTS FROM EDA")
print("=" * 60)

print("""
1. SMOKER IMPACT: Smokers pay significantly higher insurance charges
   - Average charges for smokers: ${:,.0f}
   - Average charges for non-smokers: ${:,.0f}
   - Difference: ${:,.0f} ({:.0f}% more)

2. AGE CORRELATION: Older people tend to pay more
   - Correlation between age and charges: {:.2f}

3. BMI IMPACT: Higher BMI leads to higher charges (especially for smokers)
   - Correlation between BMI and charges: {:.2f}

4. REGION: Southeast region has slightly higher charges
   - This may be due to higher obesity rates in that region

5. GENDER: Minimal difference in charges between male and female
   - Male average: ${:,.0f}
   - Female average: ${:,.0f}
""".format(
    df[df['smoker'] == 'yes']['charges'].mean(),
    df[df['smoker'] == 'no']['charges'].mean(),
    df[df['smoker'] == 'yes']['charges'].mean() - df[df['smoker'] == 'no']['charges'].mean(),
    ((df[df['smoker'] == 'yes']['charges'].mean() - df[df['smoker'] == 'no']['charges'].mean()) / df[df['smoker'] == 'no']['charges'].mean()) * 100,
    correlation_matrix.loc['age', 'charges'],
    correlation_matrix.loc['bmi', 'charges'],
    df[df['sex'] == 'male']['charges'].mean(),
    df[df['sex'] == 'female']['charges'].mean()
))

print("=" * 60)
print("EDA COMPLETE!")
print("=" * 60)
print("All graphs saved to 'static/' folder")
print("Ready for Feature Engineering!")
