import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load('mma_predictor/models/mma_fight_predictor.pkl')
scaler, features = joblib.load('mma_predictor/models/scaler.pkl')

def predict_fight(fighter_1_stats, fighter_2_stats):
    # Create the input DataFrame for the prediction
    fighter_1_df = pd.DataFrame([fighter_1_stats], columns=features)
    fighter_2_df = pd.DataFrame([fighter_2_stats], columns=features)

    # Scale the input data
    fighter_1_df = scaler.transform(fighter_1_df)
    fighter_2_df = scaler.transform(fighter_2_df)

    # Predict the outcome for both fighters
    fighter_1_prediction = model.predict_proba(fighter_1_df)[0][1]
    fighter_2_prediction = model.predict_proba(fighter_2_df)[0][1]

    # Calculate the prediction odds
    fighter_1_odds = fighter_1_prediction / (fighter_1_prediction + fighter_2_prediction)
    fighter_2_odds = fighter_2_prediction / (fighter_1_prediction + fighter_2_prediction)

    return fighter_1_odds, fighter_2_odds

if __name__ == "__main__":
    # Example fighter stats
    fighter_1_stats = [71, 135, 72, 29, 7.0, 0.35, 9, 19, 1, 0]  # Ensure these match the order of features
    fighter_2_stats = [68, 135, 70, 31, 4.31, 0.53, 2.3, 32, 0, 1]  # Ensure these match the order of features

    odds = predict_fight(fighter_1_stats, fighter_2_stats)
    print(f"Fighter 1 Win Probability: {odds[0] * 100:.2f}%")
    print(f"Fighter 2 Win Probability: {odds[1] * 100:.2f}%")
