# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model():
    # Load dataset
    data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

    # Preprocess data
    features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
                'smoking', 'time']
    X = data[features]
    y = data['DEATH_EVENT']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, 'heart_failure_model.pkl')
    print("Model trained and saved!")


if __name__ == "__main__":
    train_model()
