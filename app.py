import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# ------------------------------
# 1️⃣ Setup Flask App
# ------------------------------
app = Flask(__name__)

# ------------------------------
# 2️⃣ Load Model
# ------------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "models")

latest_model = sorted(
    [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")],
    key=os.path.getmtime,
    reverse=True
)[0]

model = joblib.load(latest_model)
print(f"✅ Loaded model: {latest_model}")

# ------------------------------
# 3️⃣ Home Route
# ------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ------------------------------
# 4️⃣ Prediction Route (Form)
# ------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        df = pd.DataFrame([data])

        # Convert numeric fields
        numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'avg_charge_per_month']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        result = "Yes (Churn)" if prediction == 1 else "No (Not Churn)"
        return render_template(
            'index.html',
            prediction_text=f"Customer Churn Prediction: {result}",
            confidence=f"Confidence: {probability * 100:.2f}%"
        )
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

# ------------------------------
# 5️⃣ API Route (JSON)
# ------------------------------
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])

    numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'avg_charge_per_month']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return jsonify({
        "Churn": int(prediction),
        "Prediction": "Yes" if prediction == 1 else "No",
        "Confidence": round(probability * 100, 2)
    })

# ------------------------------
# 6️⃣ Run App
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
