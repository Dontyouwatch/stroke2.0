# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained machine learning model
filename = 'best_xgb_model.pkl.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input values from form
        age = int(request.form['age'])
        sex = int(request.form['sex'])  # Assuming binary encoding (0/1)
        bmi = float(request.form['bmi'])
        smoking = int(request.form['smoking'])
        diabetes = int(request.form['diabetes'])
        hypertension = int(request.form['hypertension'])
        atrial_fibrillation = int(request.form['atrial_fibrillation'])
        cholesterol = float(request.form['cholesterol'])
        previous_stroke = int(request.form['previous_stroke'])

        # Create feature array in the correct order
        data = np.array([[age, sex, bmi, smoking, diabetes, hypertension, atrial_fibrillation, cholesterol, previous_stroke]])
        
        # Make prediction
        my_prediction = model.predict(data)

        return render_template('result.html', prediction=my_prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
