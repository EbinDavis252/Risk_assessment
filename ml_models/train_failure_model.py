# train_loan_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load sample loan data
df = pd.read_csv('data/sample_loan_data.csv')

# Add a simulated target column (1 = high risk, 0 = low risk)
df['risk'] = [0, 1, 0]

X = df[['income', 'farm_size', 'loan_amount', 'prev_defaults']]
y = df['risk']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'ml_models/loan_model.pkl')
print("✅ Loan model trained and saved.")
