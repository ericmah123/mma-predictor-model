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

The feature set includes a chronologically-computed **Elo rating** per fighter
(K=32, K=40 for finishes), updated after every fight in the replay — the
current top of the Elo table is Jon Jones, Islam Makhachev, and GSP, which is
a good sanity check. A second model predicts the **method of victory**
(KO/TKO, Submission, or Decision) shown as "Likely finish" probabilities.

**Held-out performance** (704 fights, May 2024 – May 2026):

| Metric | Value |
|---|---|
| Accuracy | 63.4% |
| ROC-AUC | 0.680 |
| Log loss | 0.644 |

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

Train the models first (model files are gitignored; the data CSVs ship with
the repo — this rebuilds the training set, fighter DB, and both models in a
couple of minutes):

```bash
python -m mma_predictor.models.main
```

Then run the app:

```bash
python app.py
# open http://127.0.0.1:5000
```

Refresh the underlying data by re-downloading the CSVs in
`mma_predictor/data/` from
[Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats),
then retrain.

## API

```bash
# Search fighters by name
curl 'http://127.0.0.1:5000/api/fighters?q=jones'

# Predict a matchup (win probs + likely finish + stat comparison)
curl -X POST http://127.0.0.1:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"fighter_a": "Jon Jones", "fighter_b": "Stipe Miocic"}'

# AI-generated analysis of the pick (requires OPENROUTER_API_KEY)
curl -X POST http://127.0.0.1:5000/api/justify \
  -H 'Content-Type: application/json' \
  -d '{"fighter_a": "Jon Jones", "fighter_b": "Stipe Miocic"}'
```

## AI justification (optional)

The "✨ Explain this pick" button asks an LLM via [OpenRouter](https://openrouter.ai)
for a short analyst-style blurb grounded in the model's own numbers. To enable
it, get an API key from https://openrouter.ai/keys, copy `.env.example` to
`.env`, and set:

```
OPENROUTER_API_KEY=your-key
```

Then run the app as usual (`python app.py`) — it's loaded automatically via
`python-dotenv`. By default it uses the free `meta-llama/llama-3.3-70b-instruct:free`
model; override with `OPENROUTER_MODEL` to use a different OpenRouter model id.

Without the key the app works normally — the button simply hides.

## Disclaimer

Predictions are informed estimates from historical statistics — not betting
advice. MMA is one of the most volatile sports to predict.
