import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib
import time
from mma_predictor.models.preprocess import load_and_clean_data
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(data):
    start_time = time.time()

    features = [
        'height', 'Weight', 'reach', 'age',
        'significant_strikes_landed_per_minute', 
        'average_takedowns_landed_per_15_minutes', 
        'win_loss_ratio', 'experience',
        'stance_Orthodox', 'stance_Southpaw'
    ]
    
    data = data[features + ['Outcome']]

    if data.isnull().any().any():
        raise ValueError("Data contains NaN values. Please ensure all missing values are handled.")
    
    X = data[features]
    y = data['Outcome']

    if len(y.unique()) < 2:
        raise ValueError("The target variable 'Outcome' must have at least two classes (0 and 1).")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(random_state=42))
    ])
    
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    joblib.dump(best_model, 'mma_predictor/models/mma_fight_predictor.pkl')

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    
    return best_model, X_test, y_test

if __name__ == "__main__":
    data = load_and_clean_data('mma_predictor/data/ufc-fighters-statistics.csv')
    model, X_test, y_test = train_model(data)
    print("Model trained and saved successfully.")
