import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from mma_predictor.models.preprocess import load_and_clean_data

def train_model(data):
    # Define the features to be used for training
    features = [
        'height', 'Weight', 'reach', 'age',
        'significant_strikes_landed_per_minute', 
        'average_takedowns_landed_per_15_minutes', 
        'win_loss_ratio', 'experience',
        'stance_Orthodox', 'stance_Southpaw'
    ]
    
    # Select the relevant columns
    data = data[features + ['Outcome']]

    # Check for any remaining NaN values
    if data.isnull().any().any():
        raise ValueError("Data contains NaN values. Please ensure all missing values are handled.")
    
    # Split the data into features (X) and target (y)
    X = data[features]
    y = data['Outcome']

    # Ensure the target variable has at least two classes
    if len(y.unique()) < 2:
        raise ValueError("The target variable 'Outcome' must have at least two classes (0 and 1).")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialize and train the model with Grid Search
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_

    # Save the scaler and model with consistent feature names
    joblib.dump((scaler, features), 'mma_predictor/models/scaler.pkl')
    joblib.dump(best_model, 'mma_predictor/models/mma_fight_predictor.pkl')
    
    return best_model, X_test, y_test

if __name__ == "__main__":
    data = load_and_clean_data('mma_predictor/data/ufc-fighters-statistics.csv')
    model, X_test, y_test = train_model(data)
    print("Model trained and saved successfully.")
