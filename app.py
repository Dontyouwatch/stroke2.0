from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the model from .pkl file
with open('strokemodel.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'Age': float(request.form['age']),
            'Sex': int(request.form['sex']),
            'BMI': float(request.form['bmi']),
            'Smoking': int(request.form['smoking']),
            'Diabetes': int(request.form['diabetes']),
            'Hypertension': float(request.form['hypertension']),
            'Atrial_Fibrillation': int(request.form['afib']),
            'Cholesterol': float(request.form['cholesterol']),
            'Previous_Stroke': int(request.form['previous_stroke'])
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100
        
        # Format result
        result = {
            'prediction': 'High Risk of Stroke' if prediction == 1 else 'Low Risk of Stroke',
            'probability': f"{probability:.2f}%"
        }
        
        return render_template('index.html', result=result, form_data=data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)