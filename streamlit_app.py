"""
Medical Insurance Cost Prediction - Streamlit Web App (Enhanced)
Features: Medical History, Lifestyle, Deep Learning, User Authentication

Author: Krutika Mohanty
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import hashlib

st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .main-header { font-size: 2.5rem; color: #0d47a1 !important; text-align: center; padding: 1rem; font-weight: 700; }
    .sub-header { font-size: 1.1rem; color: #333 !important; text-align: center; margin-bottom: 2rem; }
    .section-header { 
        color: #0d47a1 !important; 
        border-bottom: 3px solid #0d47a1; 
        padding-bottom: 8px; 
        font-weight: 700; 
        font-size: 1.3rem;
        margin-top: 1.5rem;
    }
    h3 { color: #0d47a1 !important; font-weight: 700 !important; }
    label { font-weight: 600 !important; color: #1a1a1a !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d47a1 0%, #1565c0 100%); }
    [data-testid="stSidebar"] label { color: #ffffff !important; }
    [data-testid="stSidebar"] .stRadio label { color: #ffffff !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #ffffff !important; }
    [data-testid="stSidebar"] .stSuccess { background: rgba(255,255,255,0.2) !important; }
    [data-testid="stSidebar"] .stSuccess p { color: #ffffff !important; }
    .stButton > button { 
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%); 
        color: white; 
        border: none; 
        padding: 0.7rem 2rem; 
        font-weight: 600;
        border-radius: 8px;
        font-size: 1rem;
        width: 100%;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #0a3d91 0%, #0d47a1 100%); }
    div[data-testid="stMetric"] { 
        background: white; 
        padding: 1rem; 
        border-radius: 10px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
    }
    div[data-testid="stMetric"] label { color: #555 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #0d47a1 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# USER AUTHENTICATION
# ============================================================
USERS_FILE = 'data/users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    os.makedirs('data', exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ''

# Sidebar
with st.sidebar:
    st.markdown("## Welcome")
    if st.session_state.logged_in:
        st.success(f"Logged in as: **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ''
            st.rerun()
    else:
        st.markdown("Login to save predictions")
        auth_tab = st.radio("Account", ["Login", "Register"], label_visibility="collapsed")
        if auth_tab == "Login":
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                users = load_users()
                if login_user in users and users[login_user]['password'] == hash_password(login_pass):
                    st.session_state.logged_in = True
                    st.session_state.username = login_user
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        else:
            reg_user = st.text_input("Username", key="reg_user")
            reg_email = st.text_input("Email", key="reg_email")
            reg_pass = st.text_input("Password", type="password", key="reg_pass")
            if st.button("Register"):
                if len(reg_pass) < 6:
                    st.error("Password must be 6+ characters")
                else:
                    users = load_users()
                    if reg_user in users:
                        st.error("Username taken")
                    else:
                        users[reg_user] = {'password': hash_password(reg_pass), 'email': reg_email, 'predictions': []}
                        save_users(users)
                        st.success("Registered! Now login")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("AI-powered insurance cost prediction using **Random Forest** and **Neural Networks**.")

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_or_train_model():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.neural_network import MLPRegressor

    df = pd.read_csv('data/insurance_enhanced.csv')
    df_encoded = df.copy()
    df_encoded['sex_encoded'] = LabelEncoder().fit_transform(df_encoded['sex'])
    df_encoded['smoker_encoded'] = LabelEncoder().fit_transform(df_encoded['smoker'])
    region_dummies = pd.get_dummies(df_encoded['region'], prefix='region', dtype=int)
    df_encoded = pd.concat([df_encoded, region_dummies], axis=1)
    bmi_dummies = pd.get_dummies(df_encoded['bmi_category'], prefix='bmi_cat', dtype=int)
    df_encoded = pd.concat([df_encoded, bmi_dummies], axis=1)

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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)

    mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
                       max_iter=500, random_state=42, early_stopping=True)
    mlp.fit(X_scaled, y)

    return model, mlp, scaler, feature_columns

model, mlp_model, scaler, feature_columns = load_or_train_model()
USD_TO_INR = 83.0

# ============================================================
# MAIN CONTENT - PREDICTION FORM
# ============================================================
st.markdown('<h1 class="main-header">Medical Insurance Cost Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict your estimated health insurance premium using Machine Learning</p>', unsafe_allow_html=True)

# Model selection
model_choice = st.radio("Select Model", ["Random Forest (Recommended)", "Neural Network (MLP)"], horizontal=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<h3 class="section-header">Personal Information</h3>', unsafe_allow_html=True)
    age = st.slider("Age", 18, 64, 30)
    sex = st.selectbox("Gender", ["male", "female"])
    bmi = st.number_input("BMI", 15.0, 55.0, 25.0, 0.1)
    children = st.selectbox("Children", [0, 1, 2, 3, 4, 5])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["north", "south", "east", "west"])

with col2:
    st.markdown('<h3 class="section-header">Medical History</h3>', unsafe_allow_html=True)
    hospitalizations = st.selectbox("Hospitalizations (5 yrs)", [0, 1, 2, 3, 4, 5])
    chronic_diseases = st.selectbox("Chronic Diseases", [0, 1, 2, 3, 4])
    pre_existing = st.selectbox("Pre-existing Condition", [0, 1])
    family_history = st.selectbox("Family History", [0, 1, 2])
    surgeries = st.selectbox("Past Surgeries", [0, 1, 2, 3])

st.markdown('<h3 class="section-header">Lifestyle</h3>', unsafe_allow_html=True)
lcol1, lcol2, lcol3, lcol4 = st.columns(4)

with lcol1:
    exercise_freq = st.selectbox("Exercise", ["Never", "Rarely", "Sometimes", "Regular", "Daily"], index=2)
    exercise_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Regular": 3, "Daily": 4}
    exercise_val = exercise_map[exercise_freq]

with lcol2:
    alcohol = st.selectbox("Alcohol", ["None", "Occasional", "Moderate", "Heavy"], index=1)
    alcohol_map = {"None": 0, "Occasional": 1, "Moderate": 2, "Heavy": 3}
    alcohol_val = alcohol_map[alcohol]

with lcol3:
    smoking_years = st.number_input("Smoking Years", 0, 30, 0)
    diet_quality = st.selectbox("Diet Quality", ["Poor", "Fair", "Good", "Excellent"], index=1)
    diet_map = {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}
    diet_val = diet_map[diet_quality]

with lcol4:
    stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High", "Very High"], index=1)
    stress_map = {"Low": 1, "Moderate": 2, "High": 3, "Very High": 4}
    stress_val = stress_map[stress_level]
    sleep_hours = st.number_input("Sleep Hours", 4.0, 10.0, 7.0, 0.5)
    water_intake = st.number_input("Water Intake (L)", 0.5, 4.0, 2.0, 0.1)
    sugar_intake = st.selectbox("Sugar Intake", ["Low", "Moderate", "High"], index=1)
    sugar_map = {"Low": 0, "Moderate": 1, "High": 2}
    sugar_val = sugar_map[sugar_intake]

if st.button("Predict Insurance Cost", type="primary"):
    sex_encoded = 1 if sex == 'male' else 0
    smoker_encoded = 1 if smoker == 'yes' else 0

    region_mapping = {
        'north': {'northeast': 0, 'northwest': 1, 'southeast': 0, 'southwest': 0},
        'south': {'northeast': 0, 'northwest': 0, 'southeast': 1, 'southwest': 0},
        'east': {'northeast': 1, 'northwest': 0, 'southeast': 0, 'southwest': 0},
        'west': {'northeast': 0, 'northwest': 0, 'southeast': 0, 'southwest': 1}
    }
    r = region_mapping.get(region, {'northeast': 0, 'northwest': 0, 'southeast': 0, 'southwest': 0})

    bmi_cats = {'normal': 0, 'obese': 0, 'overweight': 0, 'underweight': 0}
    if bmi < 18.5: bmi_cats['underweight'] = 1
    elif bmi < 25: bmi_cats['normal'] = 1
    elif bmi < 30: bmi_cats['overweight'] = 1
    else: bmi_cats['obese'] = 1

    health_risk_score = (
        smoker_encoded * 3 + (1 if bmi > 30 else 0) * 2 +
        (1 if chronic_diseases > 1 else 0) * 2 +
        (1 if exercise_val <= 1 else 0) * 1 +
        (1 if age > 50 else 0) * 1 +
        (1 if stress_val >= 3 else 0) * 1
    )

    features = np.array([[
        age, sex_encoded, bmi, children, smoker_encoded,
        r['northeast'], r['northwest'], r['southeast'], r['southwest'],
        hospitalizations, chronic_diseases, pre_existing, family_history, surgeries,
        exercise_val, alcohol_val, smoking_years, diet_val, stress_val,
        sleep_hours, water_intake, sugar_val, health_risk_score,
        bmi_cats['normal'], bmi_cats['obese'], bmi_cats['overweight'], bmi_cats['underweight']
    ]])

    if "Random Forest" in model_choice:
        prediction_usd = model.predict(features)[0]
        model_name = "Random Forest"
    else:
        features_scaled = scaler.transform(features)
        prediction_usd = mlp_model.predict(features_scaled)[0]
        model_name = "Neural Network (MLP)"

    prediction_inr = prediction_usd * USD_TO_INR

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Annual Premium (INR)", f"\u20b9{prediction_inr:,.0f}")
    with c2: st.metric("Annual Premium (USD)", f"${prediction_usd:,.0f}")
    with c3: st.metric("Monthly Premium (INR)", f"\u20b9{prediction_inr/12:,.0f}")

    st.caption(f"Prediction by: {model_name}")

    if st.session_state.logged_in:
        pred_data = {'age': age, 'sex': sex, 'bmi': bmi, 'smoker': smoker,
                     'region': region, 'prediction_inr': f"\u20b9{prediction_inr:,.0f}"}
        users = load_users()
        if 'predictions' not in users[st.session_state.username]:
            users[st.session_state.username]['predictions'] = []
        users[st.session_state.username]['predictions'].append(pred_data)
        save_users(users)

# ============================================================
# MODEL PERFORMANCE
# ============================================================
st.markdown("---")
st.subheader("Model Performance")

metrics_data = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Neural Network (MLP)'],
    'R2 Score': [0.7919, 0.8062, 0.8947, 0.8785, 0.8768],
    'MAE ($)': [4481.63, 2987.77, 2665.70, 2772.35, 2987.30],
    'RMSE ($)': [6184.07, 5968.17, 4398.92, 4724.54, 4757.59]
}
metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df, use_container_width=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = ['#78909c', '#78909c', '#0d47a1', '#78909c', '#78909c']

axes[0].bar(metrics_df['Model'], metrics_df['R2 Score'], color=colors, edgecolor='black')
axes[0].set_title('R2 Score', fontweight='bold')
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=30)
for i, v in enumerate(metrics_df['R2 Score']):
    axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=8, fontweight='bold')

axes[1].bar(metrics_df['Model'], metrics_df['MAE ($)'], color=colors, edgecolor='black')
axes[1].set_title('MAE', fontweight='bold')
axes[1].tick_params(axis='x', rotation=30)

axes[2].bar(metrics_df['Model'], metrics_df['RMSE ($)'], color=colors, edgecolor='black')
axes[2].set_title('RMSE', fontweight='bold')
axes[2].tick_params(axis='x', rotation=30)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #555; padding: 1rem;'>
    <p>Medical Insurance Cost Prediction | Built with Streamlit, Scikit-learn & Neural Networks</p>
    <p>Developer: Krutika Mohanty | B.Tech Computer Science</p>
</div>
""", unsafe_allow_html=True)
