"""
Medical Insurance Cost Prediction - Flask Web App with Authentication
Enhanced with medical history, lifestyle features, and user auth.

Author: Krutika Mohanty
"""

from flask import Flask, render_template, request, session, redirect, url_for, flash
import joblib
import numpy as np
import json
import hashlib
import os
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

USD_TO_INR = 83.0

# ============================================================
# USER AUTHENTICATION SYSTEM
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

def register_user(username, password, email):
    users = load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = {
        'password': hash_password(password),
        'email': email,
        'predictions': []
    }
    save_users(users)
    return True, "Registration successful"

def authenticate_user(username, password):
    users = load_users()
    if username in users and users[username]['password'] == hash_password(password):
        return True
    return False

def save_prediction(username, prediction_data):
    users = load_users()
    if username in users:
        if 'predictions' not in users[username]:
            users[username]['predictions'] = []
        users[username]['predictions'].append(prediction_data)
        save_users(users)

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading enhanced model...")
model = joblib.load('models/insurance_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
print("Model loaded successfully!")

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', logged_in=True, username=session['username'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if authenticate_user(username, password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        success, message = register_user(username, password, email)
        if success:
            flash(message, 'success')
            return redirect(url_for('login'))
        else:
            flash(message, 'error')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))
    users = load_users()
    predictions = users.get(session['username'], {}).get('predictions', [])
    return render_template('history.html', predictions=predictions, logged_in=True, username=session['username'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    try:
        # Original features
        age = float(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # Medical history features
        hospitalizations = int(request.form.get('hospitalizations', 0))
        chronic_diseases = int(request.form.get('chronic_diseases', 0))
        pre_existing = int(request.form.get('pre_existing', 0))
        family_history = int(request.form.get('family_history', 0))
        surgeries = int(request.form.get('surgeries', 0))

        # Lifestyle features
        exercise_freq = int(request.form.get('exercise_freq', 2))
        alcohol = int(request.form.get('alcohol', 1))
        smoking_years = int(request.form.get('smoking_years', 0))
        diet_quality = int(request.form.get('diet_quality', 2))
        stress_level = int(request.form.get('stress_level', 2))
        sleep_hours = float(request.form.get('sleep_hours', 7))
        water_intake = float(request.form.get('water_intake', 2))
        sugar_intake = int(request.form.get('sugar_intake', 1))

        # Encode
        sex_encoded = 1 if sex == 'male' else 0
        smoker_encoded = 1 if smoker == 'yes' else 0

        region_mapping = {
            'north': {'northeast': 0, 'northwest': 1, 'southeast': 0, 'southwest': 0},
            'south': {'northeast': 0, 'northwest': 0, 'southeast': 1, 'southwest': 0},
            'east': {'northeast': 1, 'northwest': 0, 'southeast': 0, 'southwest': 0},
            'west': {'northeast': 0, 'northwest': 0, 'southeast': 0, 'southwest': 1}
        }
        region_enc = region_mapping.get(region, {'northeast': 0, 'northwest': 0, 'southeast': 0, 'southwest': 0})

        # BMI category
        if bmi < 18.5:
            bmi_cats = {'normal': 0, 'obese': 0, 'overweight': 0, 'underweight': 1}
        elif bmi < 25:
            bmi_cats = {'normal': 1, 'obese': 0, 'overweight': 0, 'underweight': 0}
        elif bmi < 30:
            bmi_cats = {'normal': 0, 'obese': 0, 'overweight': 1, 'underweight': 0}
        else:
            bmi_cats = {'normal': 0, 'obese': 1, 'overweight': 0, 'underweight': 0}

        # Health risk score
        health_risk_score = (
            smoker_encoded * 3 +
            (1 if bmi > 30 else 0) * 2 +
            (1 if chronic_diseases > 1 else 0) * 2 +
            (1 if exercise_freq <= 1 else 0) * 1 +
            (1 if age > 50 else 0) * 1 +
            (1 if stress_level >= 3 else 0) * 1
        )

        features = np.array([[
            age, sex_encoded, bmi, children, smoker_encoded,
            region_enc['northeast'], region_enc['northwest'],
            region_enc['southeast'], region_enc['southwest'],
            hospitalizations, chronic_diseases, pre_existing, family_history, surgeries,
            exercise_freq, alcohol, smoking_years, diet_quality, stress_level,
            sleep_hours, water_intake, sugar_intake, health_risk_score,
            bmi_cats['normal'], bmi_cats['obese'], bmi_cats['overweight'], bmi_cats['underweight']
        ]])

        features_scaled = scaler.transform(features)
        prediction_usd = model.predict(features_scaled)[0]
        prediction_inr = prediction_usd * USD_TO_INR

        predicted_cost = f"₹{prediction_inr:,.0f}"
        predicted_cost_usd = f"${prediction_usd:,.0f}"

        # Save prediction
        prediction_data = {
            'age': age, 'sex': sex, 'bmi': bmi, 'children': children,
            'smoker': smoker, 'region': region,
            'prediction_inr': predicted_cost,
            'prediction_usd': predicted_cost_usd
        }
        save_prediction(session['username'], prediction_data)

        return render_template('index.html',
                             prediction=predicted_cost,
                             prediction_usd=predicted_cost_usd,
                             show_result=True,
                             logged_in=True,
                             username=session['username'],
                             age=age, sex=sex, bmi=bmi, children=children,
                             smoker=smoker, region=region)

    except Exception as e:
        return render_template('index.html',
                             error=f"Error: {str(e)}",
                             show_result=False,
                             logged_in=True,
                             username=session['username'])

# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("MEDICAL INSURANCE COST PREDICTION (Enhanced)")
    print("=" * 50)
    print("Starting Flask server...")
    print("Open browser and go to: http://127.0.0.1:8080")
    print("=" * 50)
    app.run(debug=False, host='127.0.0.1', port=8080)
