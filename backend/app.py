from flask import Flask, request, jsonify, render_template
import joblib
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('heart_failure_model.pkl')

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    return render_template('predict.html')


@app.route('/results')
def results_page():
    return render_template('results.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    national_id = data['national_id']
    features = pd.DataFrame([data['features']])

    # Predict
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0].tolist()

    # Save to Firestore
    db.collection('patients').document(national_id).set({
        'national_id': national_id,
        'features': data['features'],
        'prediction': int(prediction),
        'prediction_proba': prediction_proba
    })

    return jsonify({
        'prediction': int(prediction),
        'prediction_proba': prediction_proba
    })


@app.route('/patients', methods=['GET'])
def get_patients():
    patients = db.collection('patients').stream()
    patient_list = [patient.to_dict() for patient in patients]
    return jsonify(patient_list)


if __name__ == "__main__":
    app.run(debug=True)
