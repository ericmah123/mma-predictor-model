"""Train the fight predictor on differential features with real outcomes.

Evaluation uses a chronological split (train on older fights, test on the
most recent 15%) — the honest way to measure a sports prediction model.
The final saved model is refit on all data.
"""

import json

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mma_predictor.models.preprocess import DATA_DIR, MODEL_FEATURES

MODEL_PATH = "mma_predictor/models/mma_fight_predictor.pkl"
METHOD_MODEL_PATH = "mma_predictor/models/mma_method_predictor.pkl"
METRICS_PATH = f"{DATA_DIR}/model_metrics.json"


def make_model():
    gbc = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(gbc, method="isotonic", cv=3)),
    ])


def make_method_model():
    """3-class (KO / Sub / Dec) finish-type model — softmax probs for display."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )),
    ])


def train_method_model(df, test_frac=0.15):
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * (1 - test_frac))
    train, test = df.iloc[:split], df.iloc[split:]

    model = make_method_model()
    model.fit(train[MODEL_FEATURES], train["method"])
    acc = accuracy_score(test["method"], model.predict(test[MODEL_FEATURES]))
    base_rate = test["method"].value_counts(normalize=True).max()

    final = make_method_model()
    final.fit(df[MODEL_FEATURES], df["method"])
    return final, {"accuracy": acc, "base_rate": base_rate}


def train_and_evaluate(df, test_frac=0.15):
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * (1 - test_frac))
    train, test = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train[MODEL_FEATURES], train["label"]
    X_test, y_test = test[MODEL_FEATURES], test["label"]

    model = make_model()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "test_fights": len(test),
        "test_from": str(test["date"].min().date()),
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
        "log_loss": log_loss(y_test, proba),
        "brier": brier_score_loss(y_test, proba),
    }

    # Refit on everything for the deployed model.
    final = make_model()
    final.fit(df[MODEL_FEATURES], df["label"])
    return final, metrics


def save_metrics(win_metrics, method_metrics, path=METRICS_PATH):
    """Persist evaluation metrics as the regression baseline for the next
    data refresh to compare against (see mma_predictor/models/refresh.py)."""
    payload = {"win_model": win_metrics, "method_model": method_metrics}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def main():
    df = pd.read_csv("mma_predictor/data/training_data.csv", parse_dates=["date"])
    model, metrics = train_and_evaluate(df)
    joblib.dump(model, MODEL_PATH)
    print(f"Held-out test: {metrics['test_fights']} fights from {metrics['test_from']}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"  Log loss: {metrics['log_loss']:.4f}")
    print(f"  Brier:    {metrics['brier']:.4f}")
    print(f"Saved model -> {MODEL_PATH}")

    method_model, m = train_method_model(df)
    joblib.dump(method_model, METHOD_MODEL_PATH)
    print(f"Method model: accuracy {m['accuracy']:.4f} "
          f"(majority-class base rate {m['base_rate']:.4f})")
    print(f"Saved method model -> {METHOD_MODEL_PATH}")

    save_metrics(metrics, m)
    print(f"Saved metrics -> {METRICS_PATH}")
    return metrics, m


if __name__ == "__main__":
    main()
