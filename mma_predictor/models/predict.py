import joblib
import pandas as pd

# Load the trained model pipeline
model_pipeline = joblib.load('mma_predictor/models/mma_fight_predictor.pkl')

def predict_fight(fighter_1_stats, fighter_2_stats):
    # Define feature names
    features = [
        'height', 'Weight', 'reach', 'age',
        'significant_strikes_landed_per_minute', 
        'average_takedowns_landed_per_15_minutes', 
        'win_loss_ratio', 'experience',
        'stance_Orthodox', 'stance_Southpaw'
    ]

    # Create the input DataFrame for the prediction
    fighter_1_df = pd.DataFrame([fighter_1_stats], columns=features)
    fighter_2_df = pd.DataFrame([fighter_2_stats], columns=features)

    # Ensure the feature names are retained during transformation
    poly_features = model_pipeline.named_steps['poly'].get_feature_names_out(features) if hasattr(model_pipeline.named_steps['poly'], 'get_feature_names_out') else model_pipeline.named_steps['poly'].get_feature_names(features)
    
    fighter_1_poly = pd.DataFrame(model_pipeline.named_steps['poly'].transform(fighter_1_df), columns=poly_features)
    fighter_2_poly = pd.DataFrame(model_pipeline.named_steps['poly'].transform(fighter_2_df), columns=poly_features)

    # Convert to NumPy array before scaling to avoid feature name warnings
    fighter_1_scaled = model_pipeline.named_steps['scaler'].transform(fighter_1_poly.values)
    fighter_2_scaled = model_pipeline.named_steps['scaler'].transform(fighter_2_poly.values)

    # Predict the outcome for both fighters
    fighter_1_prediction = model_pipeline.named_steps['model'].predict_proba(fighter_1_scaled)[0][1]
    fighter_2_prediction = model_pipeline.named_steps['model'].predict_proba(fighter_2_scaled)[0][1]

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
