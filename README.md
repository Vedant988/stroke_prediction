# 𖡎 Brain-Stroke Prediction Web App

This project is a Flask Based machine learning pipeline built to predict the risk of stroke based on medical and demographic features. It includes rigorous preprocessing, model training, evaluation using ROC curves, and deployment via a Flask web application.

## 📁 Dataset
- Source: `Actual-healthcare-dataset-stroke-data.csv` from kaggle.com

---

## ⚙ Model Training - 

### 1. **Random Forest Classifier**
- Ensemble bagging method for non-linear and high-dimensional data.
- Tuned using GridSearchCV.
- AUC: **0.839**

### 2. **XGBoost Classifier**
- Gradient Boosted Trees with regularization.
- Hyperparameter-tuned with learning rate, depth, estimators.
- AUC: **0.820**

### 3. **Voting Classifier**
- Combined model with soft voting of both RandomForest and XGBoost.
- Improved generalization.
- AUC: **0.833**

---

## 📊 ROC Curve Analysis

![image_alt](https://github.com/Vedant988/stroke_prediction/blob/main/Screenshot%202025-01-03%20140243.png?raw=true)

- **Red** → Random Forest
- **Green** → XGBoost
- **Orange** → Voting Classifier  

---

## 🌐 Flask Web App
A lightweight Flask server wraps the trained model (`prediction_model.pkl`) and provides a simple (as beginner in html at that instance) frontend UI for stroke risk prediction.

### 🔁 Inference Flow:
```python
@app.route('/predict', methods=['POST'])
def predict():
    # Read and preprocess input from HTML form
    # Convert to NumPy array and reshape
    # Call joblib model's .predict method
    # Return prediction to rendered template
```
# 🏁 Getting Started
### Clone and Run:
- git clone https://github.com/Vedant988/stroke_prediction.git
- cd stroke_prediction
- pip install -r requirements.txt
- python app.py
- Open browser at http://127.0.0.1:5000

---
 
# 📌 File Structure
```
├── static/
│   ├── MainVisuals.jpg
│   ├── removed-background.png
│   └── styles.css
│
├── templates/
│   └── index.html
│
├── stroke-prediction.ipynb
├── app.py
├── prediction_model.pkl
├── Actual-healthcare-dataset-stroke-data.csv
├── healthcare-dataset-stroke-data.csv
├── requirements.txt
├── .gitattributes
├── Screenshot 2025-01-03 140215.png
├── Screenshot 2025-01-03 140229.png
├── Screenshot 2025-01-03 140243.png

```
# Thank You for reading it..
