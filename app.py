import os

from flask import Flask, jsonify, render_template, request

from mma_predictor.models.predict import (
    comparison,
    load_fighter_db,
    load_model,
    predict_matchup,
)

app = Flask(__name__, static_folder="static", template_folder="templates")

model = load_model()
fighter_db = load_fighter_db()
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


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    name_a = str(body.get("fighter_a", "")).strip()
    name_b = str(body.get("fighter_b", "")).strip()

    if not name_a or not name_b:
        return jsonify({"error": "Both fighters are required."}), 400
    if name_a.lower() == name_b.lower():
        return jsonify({"error": "Pick two different fighters."}), 400

    snaps = []
    for name in (name_a, name_b):
        real = name_index.get(name.lower())
        if real is None:
            return jsonify({"error": f"Fighter not found: {name}"}), 404
        snaps.append(fighter_db[real])
    snap_a, snap_b = snaps

    prob_a = predict_matchup(model, snap_a, snap_b)
    return jsonify({
        "fighter_a": {"name": snap_a["name"], "prob": round(float(prob_a), 4),
                      "record": f"{int(snap_a['wins'])}-{int(snap_a['losses'])}"},
        "fighter_b": {"name": snap_b["name"], "prob": round(float(1 - prob_a), 4),
                      "record": f"{int(snap_b['wins'])}-{int(snap_b['losses'])}"},
        "comparison": comparison(snap_a, snap_b),
    })


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1",
            host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
