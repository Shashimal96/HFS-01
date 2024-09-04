import pandas as pd
import numpy as np

# Define the number of samples
num_samples = 1000

# Generate synthetic data
data = {
    'age': np.random.randint(40, 95, num_samples),
    'anaemia': np.random.randint(0, 2, num_samples),
    'creatinine_phosphokinase': np.random.randint(23, 7861, num_samples),
    'diabetes': np.random.randint(0, 2, num_samples),
    'ejection_fraction': np.random.randint(14, 80, num_samples),
    'high_blood_pressure': np.random.randint(0, 2, num_samples),
    'platelets': np.random.uniform(25100, 850000, num_samples),
    'serum_creatinine': np.random.uniform(0.5, 9.4, num_samples),
    'serum_sodium': np.random.randint(113, 148, num_samples),
    'sex': np.random.randint(0, 2, num_samples),
    'smoking': np.random.randint(0, 2, num_samples),
    'time': np.random.randint(4, 285, num_samples),
    'DEATH_EVENT': np.random.randint(0, 2, num_samples)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('heart_failure_clinical_records_dataset.csv', index=False)

print("CSV file created successfully!")
