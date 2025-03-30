from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pickle # For model saving and loading

app = Flask(__name__)

# Load your trained model and scaler (replace with your file paths)
with open('best_xgb_model.pkl', 'rb') as f:
    best_xgb = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('x_columns.pkl', 'rb') as f:
    X_columns = pickle.load(f)

def predict_stroke(input_data):
    user_input = pd.DataFrame([input_data])
    missing_cols = set(X_columns) - set(user_input.columns)
    for col in missing_cols:
        user_input[col] = 0
    user_input = user_input[X_columns]
    numeric_features = ["Age", "Cholesterol", "BMI"]
    user_input[numeric_features] = scaler.transform(user_input[numeric_features])
    stroke_prob = best_xgb.predict_proba(user_input)[:, 1][0]
    stroke_prediction = "Yes" if stroke_prob > 0.5 else "No"
    return stroke_prediction, stroke_prob * 100

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    stroke_risk, stroke_probability = predict_stroke(data)
    return jsonify({'stroke_risk': stroke_risk, 'stroke_probability': round(stroke_probability, 2)})

if __name__ == '__main__':
    app.run(debug=True)
