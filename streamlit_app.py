"""
Medical Insurance Cost Prediction - Streamlit Web App
A Machine Learning project that predicts medical insurance costs.

Author: Krutika Mohanty
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessor
@st.cache_resource
def load_model():
    model = joblib.load('models/insurance_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    return model, scaler, feature_columns

model, scaler, feature_columns = load_model()

# USD to INR conversion rate
USD_TO_INR = 83.0

# Header
st.markdown('<h1 class="main-header">Medical Insurance Cost Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict your estimated health insurance premium using Machine Learning</p>', unsafe_allow_html=True)

# Sidebar for input
st.sidebar.header("Enter Your Details")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=64, value=30, help="Age of the primary beneficiary")
    sex = st.selectbox("Gender", ["male", "female"], help="Gender of the beneficiary")
    bmi = st.number_input("BMI", min_value=15.0, max_value=55.0, value=25.0, step=0.1,
                          help="Body Mass Index (weight in kg / height in m²)")

with col2:
    children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5],
                            help="Number of children covered by health insurance")
    smoker = st.selectbox("Smoker", ["yes", "no"], help="Smoking status")
    region = st.selectbox("Region", ["north", "south", "east", "west"],
                          help="Residential region in India")

# BMI Calculator
with st.expander("Calculate BMI"):
    st.write("Enter your weight and height to calculate BMI:")
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
    
    if weight > 0 and height > 0:
        height_m = height / 100
        calculated_bmi = weight / (height_m ** 2)
        st.info(f"Your BMI: {calculated_bmi:.1f}")
        if st.button("Use This BMI"):
            bmi = calculated_bmi
            st.rerun()

# Prediction button
if st.button("Predict Insurance Cost", type="primary"):
    # Encode categorical variables
    sex_encoded = 1 if sex == 'male' else 0
    smoker_encoded = 1 if smoker == 'yes' else 0
    
    # Region mapping
    region_mapping = {
        'north': {'northeast': 0, 'northwest': 1, 'southeast': 0, 'southwest': 0},
        'south': {'northeast': 0, 'northwest': 0, 'southeast': 1, 'southwest': 0},
        'east': {'northeast': 1, 'northwest': 0, 'southeast': 0, 'southwest': 0},
        'west': {'northeast': 0, 'northwest': 0, 'southeast': 0, 'southwest': 1}
    }
    
    region_encoded = region_mapping.get(region, {
        'northeast': 0, 'northwest': 0, 'southeast': 0, 'southwest': 0
    })
    
    # Create feature array
    features = np.array([[
        age,
        sex_encoded,
        bmi,
        children,
        smoker_encoded,
        region_encoded['northeast'],
        region_encoded['northwest'],
        region_encoded['southeast'],
        region_encoded['southwest']
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction_usd = model.predict(features_scaled)[0]
    prediction_inr = prediction_usd * USD_TO_INR
    
    # Display result
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Estimated Annual Premium (INR)", f"₹{prediction_inr:,.0f}")
    
    with col2:
        st.metric("Estimated Annual Premium (USD)", f"${prediction_usd:,.0f}")
    
    with col3:
        st.metric("Monthly Premium (INR)", f"₹{prediction_inr/12:,.0f}")
    
    # User details
    st.subheader("Your Input Details")
    details_col1, details_col2 = st.columns(2)
    
    with details_col1:
        st.write(f"**Age:** {age} years")
        st.write(f"**Gender:** {sex}")
        st.write(f"**BMI:** {bmi}")
    
    with details_col2:
        st.write(f"**Children:** {children}")
        st.write(f"**Smoker:** {smoker}")
        st.write(f"**Region:** {region}")

# Information section
st.markdown("---")
st.subheader("About This Prediction")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.info("**AI-Powered**\n\nUses Random Forest algorithm trained on medical records to predict insurance costs.")

with info_col2:
    st.info("**Indian Focused**\n\nDesigned for Indian users with INR currency and Indian state regions.")

with info_col3:
    st.info("**Key Factors**\n\nSmoking status, BMI, Age, and Region are the main cost predictors.")

# Model Performance Section
st.markdown("---")
st.subheader("Model Performance")

# Load actual metrics
metrics_data = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest'],
    'R² Score': [0.8069, 0.7992, 0.8843],
    'MAE ($)': [4177.05, 2801.90, 2546.60],
    'RMSE ($)': [5956.34, 6075.06, 4611.70]
}

metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df, use_container_width=True)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# R2 Score comparison
colors = ['#3498DB', '#E74C3C', '#2ECC71']
axes[0].bar(metrics_df['Model'], metrics_df['R² Score'], color=colors, edgecolor='black')
axes[0].set_title('R² Score Comparison', fontweight='bold')
axes[0].set_ylabel('R² Score')
axes[0].set_ylim(0, 1)
for i, v in enumerate(metrics_df['R² Score']):
    axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

# MAE comparison
axes[1].bar(metrics_df['Model'], metrics_df['MAE ($)'], color=colors, edgecolor='black')
axes[1].set_title('Mean Absolute Error', fontweight='bold')
axes[1].set_ylabel('MAE ($)')
for i, v in enumerate(metrics_df['MAE ($)']):
    axes[1].text(i, v + 50, f'${v:,.0f}', ha='center', fontsize=9, fontweight='bold')

# RMSE comparison
axes[2].bar(metrics_df['Model'], metrics_df['RMSE ($)'], color=colors, edgecolor='black')
axes[2].set_title('Root Mean Squared Error', fontweight='bold')
axes[2].set_ylabel('RMSE ($)')
for i, v in enumerate(metrics_df['RMSE ($)']):
    axes[2].text(i, v + 50, f'${v:,.0f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Medical Insurance Cost Prediction | Built with Streamlit & Scikit-learn</p>
    <p>Developer: Krutika Mohanty | B.Tech Computer Science</p>
</div>
""", unsafe_allow_html=True)
