# MMA Fight Predictor

Predict UFC fight outcomes head-to-head. Pick two fighters by name and get
calibrated win probabilities from a gradient-boosted model trained on
**thousands of real UFC fight results** (1997–present).

## How it works

1. **Real data** — fight results, per-round stats, and fighter attributes
   scraped from ufcstats.com (via the pre-scraped CSVs published by
   [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats)).
2. **Leakage-free features** — fights are replayed in chronological order;
   each fighter's career stats (striking rates, takedowns, finish rates,
   streaks, physical attributes) are snapshotted *before* every fight, so the
   model never sees information from the fight it's predicting.
3. **Differential modeling** — the model's inputs are the *differences*
   between the two fighters (reach diff, strike-rate diff, streak diff, …),
   which directly encodes the matchup. Predictions are symmetric: swapping
   the fighters gives mirrored probabilities.
4. **Honest evaluation** — the test set is the most recent 15% of fights,
   strictly after all training data.

**Held-out performance** (704 fights, May 2024 – May 2026):

| Metric | Value |
|---|---|
| Accuracy | 62.9% |
| ROC-AUC | 0.680 |
| Log loss | 0.646 |

For context, published UFC prediction models typically land between 62–68%,
with a practical ceiling around 70% given the sport's volatility.

## Stack

- **Model**: scikit-learn Gradient Boosting + isotonic calibration
- **Backend**: Flask JSON API (`/api/fighters` search, `/api/predict`)
- **Frontend**: vanilla JS — fighter autocomplete, async predictions,
  probability bar, and a per-stat comparison table

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the app (a trained model and fighter database ship with the repo):

```bash
python app.py
# open http://127.0.0.1:5000
```

Retrain from scratch (rebuilds the training set, fighter DB, and model):

```bash
python -m mma_predictor.models.main
```

Refresh the underlying data by re-downloading the CSVs in
`mma_predictor/data/` from
[Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats),
then retrain.

## API

```bash
# Search fighters by name
curl 'http://127.0.0.1:5000/api/fighters?q=jones'

# Predict a matchup
curl -X POST http://127.0.0.1:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"fighter_a": "Jon Jones", "fighter_b": "Stipe Miocic"}'
```

## Disclaimer

Predictions are informed estimates from historical statistics — not betting
advice. MMA is one of the most volatile sports to predict.
