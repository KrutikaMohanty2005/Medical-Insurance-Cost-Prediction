"""
Medical Insurance Cost Prediction - Enhanced Dataset Generator
Adds medical history and lifestyle features to the original dataset.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

print("=" * 60)
print("GENERATING ENHANCED DATASET WITH NEW FEATURES")
print("=" * 60)

df = pd.read_csv('data/insurance_cleaned.csv')
print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")

n = len(df)

# ============================================================
# MEDICAL HISTORY FEATURES
# ============================================================
print("\nAdding medical history features...")

# 1. Number of hospitalizations in last 5 years (0-5)
df['hospitalizations'] = np.random.choice(
    [0, 1, 2, 3, 4, 5], size=n,
    p=[0.50, 0.25, 0.12, 0.07, 0.04, 0.02]
)

# 2. Chronic diseases (0-4: diabetes, hypertension, heart disease, asthma)
df['chronic_diseases'] = np.random.choice(
    [0, 1, 2, 3, 4], size=n,
    p=[0.40, 0.30, 0.15, 0.10, 0.05]
)
# Correlate chronic diseases with age and smoker status
older_mask = df['age'] > 45
smoker_mask = df['smoker'] == 'yes'
df.loc[older_mask, 'chronic_diseases'] = df.loc[older_mask, 'chronic_diseases'].apply(
    lambda x: min(x + 1, 4)
)
df.loc[smoker_mask, 'chronic_diseases'] = df.loc[smoker_mask, 'chronic_diseases'].apply(
    lambda x: min(x + 1, 4)
)

# 3. Pre-existing conditions (binary: 0=no, 1=yes)
df['pre_existing'] = ((df['chronic_diseases'] >= 1) | (df['age'] > 50)).astype(int)

# 4. Family medical history (0=no conditions, 1=one condition, 2=multiple)
df['family_history'] = np.random.choice([0, 1, 2], size=n, p=[0.55, 0.30, 0.15])

# 5. Number of surgeries in lifetime (0-3)
df['surgeries'] = np.random.choice([0, 1, 2, 3], size=n, p=[0.60, 0.25, 0.10, 0.05])
# Older people more likely to have had surgery
df.loc[df['age'] > 50, 'surgeries'] = df.loc[df['age'] > 50, 'surgeries'].apply(
    lambda x: min(x + 1, 3)
)

# ============================================================
# LIFESTYLE FEATURES
# ============================================================
print("Adding lifestyle features...")

# 6. Exercise frequency (0=never, 1=rarely, 2=sometimes, 3=regular, 4=daily)
df['exercise_freq'] = np.random.choice(
    [0, 1, 2, 3, 4], size=n,
    p=[0.15, 0.20, 0.30, 0.20, 0.15]
)
# Smokers tend to exercise less
df.loc[smoker_mask, 'exercise_freq'] = df.loc[smoker_mask, 'exercise_freq'].apply(
    lambda x: max(x - 1, 0)
)

# 7. Alcohol consumption (0=none, 1=occasional, 2=moderate, 3=heavy)
df['alcohol'] = np.random.choice(
    [0, 1, 2, 3], size=n,
    p=[0.30, 0.35, 0.25, 0.10]
)

# 8. Smoking years (0 for non-smokers, 1-30 for smokers)
df['smoking_years'] = 0
smoker_indices = df[smoker_mask].index
df.loc[smoker_indices, 'smoking_years'] = np.random.randint(1, 25, size=len(smoker_indices))

# 9. Diet quality (1=poor, 2=fair, 3=good, 4=excellent)
df['diet_quality'] = np.random.choice(
    [1, 2, 3, 4], size=n,
    p=[0.15, 0.30, 0.35, 0.20]
)
# Higher BMI tends to correlate with poorer diet
df.loc[df['bmi'] > 30, 'diet_quality'] = df.loc[df['bmi'] > 30, 'diet_quality'].apply(
    lambda x: max(x - 1, 1)
)

# 10. Stress level (1=low, 2=moderate, 3=high, 4=very high)
df['stress_level'] = np.random.choice(
    [1, 2, 3, 4], size=n,
    p=[0.20, 0.35, 0.30, 0.15]
)

# 11. Sleep hours per night (4-10)
df['sleep_hours'] = np.clip(np.random.normal(7, 1.2, n), 4, 10).round(1)
# Smokers tend to sleep less
df.loc[smoker_indices, 'sleep_hours'] = df.loc[smoker_indices, 'sleep_hours'].apply(
    lambda x: max(x - 0.5, 4)
)

# 12. Water intake (liters per day, 0.5-4.0)
df['water_intake'] = np.clip(np.random.normal(2.0, 0.6, n), 0.5, 4.0).round(1)

# 13. Sugar intake (0=low, 1=moderate, 2=high)
df['sugar_intake'] = np.random.choice([0, 1, 2], size=n, p=[0.25, 0.45, 0.30])

# 14. BMI category
def bmi_category(bmi):
    if bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'normal'
    elif bmi < 30:
        return 'overweight'
    else:
        return 'obese'

df['bmi_category'] = df['bmi'].apply(bmi_category)

# 15. Health risk score (composite: 0-10)
df['health_risk_score'] = (
    (df['smoker_encoded'] if 'smoker_encoded' in df.columns else (df['smoker'] == 'yes').astype(int)) * 3 +
    (df['bmi'] > 30).astype(int) * 2 +
    (df['chronic_diseases'] > 1).astype(int) * 2 +
    (df['exercise_freq'] <= 1).astype(int) * 1 +
    (df['age'] > 50).astype(int) * 1 +
    (df['stress_level'] >= 3).astype(int) * 1
).clip(0, 10)

print(f"Enhanced dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# SAVE ENHANCED DATASET
# ============================================================
df.to_csv('data/insurance_enhanced.csv', index=False)
print(f"\nSaved: data/insurance_enhanced.csv")
print(f"New features added: {df.shape[1] - 7} (from 7 original)")

print("\nNew columns:")
new_cols = [c for c in df.columns if c not in ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']]
for i, col in enumerate(new_cols, 1):
    print(f"  {i}. {col}")

print("\n" + "=" * 60)
print("ENHANCED DATASET READY!")
print("=" * 60)
