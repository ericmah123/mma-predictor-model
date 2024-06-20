import pandas as pd
from mma_predictor.models.train import train_model
from mma_predictor.models.preprocess import load_and_clean_data
import joblib

# Load and clean the data
filepath = 'mma_predictor/data/ufc-fighters-statistics.csv'
data = load_and_clean_data(filepath)

# Train the model
model, X_test, y_test = train_model(data)

# Save the trained model
joblib.dump(model, 'mma_predictor/models/mma_fight_predictor.pkl')

print("Model trained and saved successfully.")
