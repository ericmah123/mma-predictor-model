"""Head-to-head fight prediction from fighter snapshots."""

import json

import joblib
import pandas as pd

from mma_predictor.models.preprocess import MODEL_FEATURES, SNAPSHOT_COLS, diff_row

MODEL_PATH = "mma_predictor/models/mma_fight_predictor.pkl"
METHOD_MODEL_PATH = "mma_predictor/models/mma_method_predictor.pkl"
FIGHTER_DB_PATH = "mma_predictor/data/fighters.json"

# Snapshot stats surfaced in the UI comparison table.
COMPARE_STATS = [
    ("slpm", "Sig. strikes landed/min", True),
    ("sapm", "Sig. strikes absorbed/min", False),
    ("td_avg", "Takedowns/15 min", True),
    ("sub_avg", "Submission attempts/15 min", True),
    ("ko_rate", "KO win rate", True),
    ("sub_rate", "Submission win rate", True),
    ("win_streak", "Current streak", True),
    ("reach", "Reach (in)", True),
]


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def load_method_model(path=METHOD_MODEL_PATH):
    return joblib.load(path)


def load_fighter_db(path=FIGHTER_DB_PATH):
    with open(path) as f:
        return json.load(f)


def predict_matchup(model, snap_a, snap_b):
    """Symmetric head-to-head probability that fighter A beats fighter B.

    Averages P(A wins | A-B) with 1 - P(B wins | B-A) so the answer doesn't
    depend on input order.
    """
    row_ab = diff_row(snap_a, snap_b)
    row_ba = diff_row(snap_b, snap_a)
    X = pd.DataFrame([row_ab, row_ba])[MODEL_FEATURES]
    p = model.predict_proba(X)[:, 1]
    prob_a = (p[0] + (1 - p[1])) / 2
    return prob_a


def predict_method(method_model, snap_a, snap_b):
    """Fight-level finish-type probabilities, symmetric in input order."""
    X = pd.DataFrame([diff_row(snap_a, snap_b), diff_row(snap_b, snap_a)])[MODEL_FEATURES]
    probs = method_model.predict_proba(X).mean(axis=0)
    return {cls: round(float(p), 4)
            for cls, p in zip(method_model.classes_, probs)}


def comparison(snap_a, snap_b):
    """Per-stat comparison for the UI: which fighter each stat favors."""
    out = []
    for key, label, higher_is_better in COMPARE_STATS:
        va, vb = snap_a.get(key, 0), snap_b.get(key, 0)
        if va == vb:
            favors = None
        elif (va > vb) == higher_is_better:
            favors = "a"
        else:
            favors = "b"
        out.append({"stat": label, "a": round(float(va), 2),
                    "b": round(float(vb), 2), "favors": favors})
    return out


if __name__ == "__main__":
    model = load_model()
    db = load_fighter_db()
    names = list(db)[:2]
    a, b = db[names[0]], db[names[1]]
    p = predict_matchup(model, a, b)
    print(f"{names[0]} vs {names[1]}: {p:.1%} / {1 - p:.1%}")
