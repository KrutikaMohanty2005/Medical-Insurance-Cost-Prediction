"""
Medical Insurance Cost Prediction - Flask Web Application (India Version)
This script creates a web interface for users to predict insurance costs in INR.

Author: Krutika Mohanty
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================
from flask import Flask, render_template, request
import joblib
import numpy as np

# ============================================================
# INITIALIZE FLASK APP
# ============================================================
app = Flask(__name__)

# Exchange rate: 1 USD = ~83 INR (approximate)
USD_TO_INR = 83.0

# ============================================================
# LOAD THE TRAINED MODEL AND SCALER
# ============================================================
print("Loading model and scaler...")
model = joblib.load('models/insurance_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
print("Model loaded successfully!")

# ============================================================
# DEFINE ROUTES
# ============================================================

# Home page route
@app.route('/')
def home():
    """Render the home page with the prediction form"""
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    """
    Process form data and predict insurance cost.
    This function:
    1. Gets user input from the form
    2. Converts categorical inputs to numbers
    3. Scales the features
    4. Makes prediction using the trained model
    5. Returns the predicted cost
    """

    try:
        # Get form data
        age = float(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # Convert categorical to numerical
        # Sex: female=0, male=1
        sex_encoded = 1 if sex == 'male' else 0

        # Smoker: no=0, yes=1
        smoker_encoded = 1 if smoker == 'yes' else 0

        # Region: Indian Regions -> Map to model's regions
        # The model was trained on US regions, so we map Indian regions similarly
        # North ~ Northwest, South ~ Southeast, East ~ Northeast, West ~ Southwest
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

        # Convert to INR (Indian Rupees)
        prediction_inr = prediction_usd * USD_TO_INR

        # Format prediction as INR currency
        predicted_cost = f"₹{prediction_inr:,.0f}"
        predicted_cost_usd = f"${prediction_usd:,.0f}"

        # Return result page
        return render_template('index.html',
                             prediction=predicted_cost,
                             prediction_usd=predicted_cost_usd,
                             show_result=True,
                             age=age,
                             sex=sex,
                             bmi=bmi,
                             children=children,
                             smoker=smoker,
                             region=region)

    except Exception as e:
        return render_template('index.html',
                             error=f"Error: {str(e)}",
                             show_result=False)

# ============================================================
# RUN THE APP
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("MEDICAL INSURANCE COST PREDICTION")
    print("=" * 50)
    print("Starting Flask server...")
    print("Open browser and go to: http://127.0.0.1:8080")
    print("=" * 50)
    app.run(debug=False, host='127.0.0.1', port=8080)
