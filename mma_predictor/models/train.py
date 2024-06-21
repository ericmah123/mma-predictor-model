import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import time
from mma_predictor.models.preprocess import load_and_clean_data
from scipy.stats import randint

def train_model(data):
    start_time = time.time()

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
    
    # Define a pipeline with feature scaling and model training
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(random_state=42))
    ])
    
    # Define hyperparameters for Randomized Search
    param_dist = {
        'model__n_estimators': randint(50, 200),
        'model__max_depth': [None] + list(range(10, 31, 10)),
        'model__min_samples_split': randint(2, 11),
        'model__min_samples_leaf': randint(1, 5)
    }
    
    # Perform Randomized Search
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=50, cv=kf, scoring='accuracy', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save the entire pipeline
    joblib.dump(best_model, 'mma_predictor/models/mma_fight_predictor.pkl')

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    
    return best_model, X_test, y_test

if __name__ == "__main__":
    data = load_and_clean_data('mma_predictor/data/ufc-fighters-statistics.csv')
    model, X_test, y_test = train_model(data)
    print("Model trained and saved successfully.")
