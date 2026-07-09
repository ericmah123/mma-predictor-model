import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

from mma_predictor.models.justify import (
    JustificationFailed,
    JustificationUnavailable,
    generate_justification,
)
from mma_predictor.models.predict import (
    comparison,
    load_fighter_db,
    load_method_model,
    load_model,
    predict_matchup,
    predict_method,
)

app = Flask(__name__, static_folder="static", template_folder="templates")

model = load_model()
method_model = load_method_model()
fighter_db = load_fighter_db()
justification_cache = {}
# Case-insensitive name index for lookup.
name_index = {name.lower(): name for name in fighter_db}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/fighters")
def search_fighters():
    query = request.args.get("q", "").strip().lower()
    if len(query) < 2:
        return jsonify([])
    matches = [
        {"name": fighter_db[real]["name"],
         "record": f"{int(fighter_db[real]['wins'])}-{int(fighter_db[real]['losses'])}",
         "fights": int(fighter_db[real]["n_fights"])}
        for lower, real in name_index.items() if query in lower
    ]
    matches.sort(key=lambda m: -m["fights"])
    return jsonify(matches[:10])


def _resolve_matchup(body):
    """Validate a {fighter_a, fighter_b} payload -> (snap_a, snap_b) or an
    (error response, status) tuple."""
    name_a = str(body.get("fighter_a", "")).strip()
    name_b = str(body.get("fighter_b", "")).strip()

    if not name_a or not name_b:
        return None, (jsonify({"error": "Both fighters are required."}), 400)
    if name_a.lower() == name_b.lower():
        return None, (jsonify({"error": "Pick two different fighters."}), 400)

    snaps = []
    for name in (name_a, name_b):
        real = name_index.get(name.lower())
        if real is None:
            return None, (jsonify({"error": f"Fighter not found: {name}"}), 404)
        snaps.append(fighter_db[real])
    return snaps, None


def _prediction_payload(snap_a, snap_b):
    prob_a = predict_matchup(model, snap_a, snap_b)
    return {
        "fighter_a": {"name": snap_a["name"], "prob": round(float(prob_a), 4),
                      "record": f"{int(snap_a['wins'])}-{int(snap_a['losses'])}"},
        "fighter_b": {"name": snap_b["name"], "prob": round(float(1 - prob_a), 4),
                      "record": f"{int(snap_b['wins'])}-{int(snap_b['losses'])}"},
        "method": predict_method(method_model, snap_a, snap_b),
        "comparison": comparison(snap_a, snap_b),
    }


@app.route("/api/predict", methods=["POST"])
def predict():
    snaps, error = _resolve_matchup(request.get_json(silent=True) or {})
    if error:
        return error
    return jsonify(_prediction_payload(*snaps))


@app.route("/api/justify", methods=["POST"])
def justify():
    snaps, error = _resolve_matchup(request.get_json(silent=True) or {})
    if error:
        return error
    snap_a, snap_b = snaps

    cache_key = tuple(sorted((snap_a["name"], snap_b["name"])))
    if cache_key in justification_cache:
        return jsonify(justification_cache[cache_key])

    try:
        analysis = generate_justification(_prediction_payload(snap_a, snap_b))
    except JustificationUnavailable:
        return jsonify({"error": "AI justification unavailable"}), 503
    except JustificationFailed:
        return jsonify({"error": "AI analysis failed, try again later."}), 502

    justification_cache[cache_key] = analysis
    return jsonify(analysis)


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1",
            host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
