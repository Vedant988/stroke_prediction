from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import joblib

model = joblib.load('prediction_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/homee')
def homee():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form['age']),
                    int(request.form['hypertension']),
                    int(request.form['heart_disease']),
                    float(request.form['avg_glucose_level']),
                    float(request.form['bmi']),
                    int(request.form['gender']),
                    int(request.form['marital_status']),
                    int(request.form['work_type']),
                    int(request.form['residence_type']),
                    int(request.form['smoking_status'])]
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)
        result = 'Stroke Risk' if prediction[0] == 1 else 'No Stroke Risk'
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    
    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)
    