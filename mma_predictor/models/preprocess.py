"""Build a leakage-free training set from real UFC fight data.

Data files (scraped from ufcstats.com, via github.com/Greco1899/scrape_ufc_stats):
  - ufc_event_details.csv   event name -> date
  - ufc_fight_results.csv   one row per bout: outcome, method, round, time
  - ufc_fight_stats.csv     one row per fighter per round: strikes, TDs, control...
  - ufc_fighter_tott.csv    fighter physical attributes: height, reach, stance, DOB

For every fight (in chronological order) we snapshot each fighter's career
stats *before* that fight, so features never contain information from the
fight being predicted. The model trains on differential features (A - B).
"""

import json
import re

import numpy as np
import pandas as pd

DATA_DIR = "mma_predictor/data"

# Features stored per fighter snapshot.
SNAPSHOT_COLS = [
    "height", "reach", "age", "stance_code",
    "wins", "losses", "win_streak", "elo",
    "slpm", "sapm", "td_avg", "sub_avg", "kd_avg", "ctrl_avg",
    "ko_rate", "sub_rate",
]

# Elo settings: standard logistic Elo, larger K for finishes.
ELO_START = 1500.0
ELO_K = 32.0
ELO_K_FINISH = 40.0

# Differential features fed to the model (A minus B for each).
DIFF_FEATURES = [f"{c}_diff" for c in SNAPSHOT_COLS if c != "stance_code"]
MODEL_FEATURES = DIFF_FEATURES + ["southpaw_diff"]

MIN_PRIOR_FIGHTS = 2  # both fighters need this many prior UFC fights


def _parse_of(value):
    """'23 of 38' -> (23, 38)."""
    if isinstance(value, str):
        m = re.match(r"\s*(\d+)\s+of\s+(\d+)", value)
        if m:
            return int(m.group(1)), int(m.group(2))
    return 0, 0


def _parse_ctrl(value):
    """'1:44' -> seconds."""
    if isinstance(value, str) and ":" in value:
        try:
            mins, secs = value.strip().split(":")
            return int(mins) * 60 + int(secs)
        except ValueError:
            return 0
    return 0


def _parse_height(value):
    """`5' 11"` -> inches."""
    if isinstance(value, str):
        m = re.match(r"\s*(\d+)'\s*(\d+)", value)
        if m:
            return int(m.group(1)) * 12 + int(m.group(2))
    return np.nan


def _parse_reach(value):
    """'72"' -> inches."""
    if isinstance(value, str):
        m = re.match(r"\s*(\d+)", value)
        if m:
            return float(m.group(1))
    return np.nan


def _fight_seconds(round_num, time_str):
    """Total fight duration assuming 5-minute rounds."""
    try:
        completed = (int(round_num) - 1) * 300
    except (TypeError, ValueError):
        return 300.0
    mins, secs = 0, 0
    if isinstance(time_str, str) and ":" in time_str:
        try:
            mins, secs = (int(x) for x in time_str.strip().split(":"))
        except ValueError:
            pass
    return float(max(completed + mins * 60 + secs, 60))


def load_raw():
    events = pd.read_csv(f"{DATA_DIR}/ufc_event_details.csv")
    results = pd.read_csv(f"{DATA_DIR}/ufc_fight_results.csv")
    stats = pd.read_csv(f"{DATA_DIR}/ufc_fight_stats.csv")
    fighters = pd.read_csv(f"{DATA_DIR}/ufc_fighter_tott.csv")
    for df, cols in ((events, ["EVENT"]), (results, ["EVENT", "BOUT", "OUTCOME", "METHOD"]),
                     (stats, ["EVENT", "BOUT", "FIGHTER"]), (fighters, ["FIGHTER"])):
        for c in cols:
            df[c] = df[c].astype(str).str.strip()
    return events, results, stats, fighters


def build_fighter_attributes(fighters):
    attrs = {}
    for _, row in fighters.iterrows():
        dob = pd.to_datetime(str(row["DOB"]).strip(), errors="coerce", format="mixed")
        stance = row["STANCE"] if isinstance(row["STANCE"], str) else ""
        attrs[row["FIGHTER"]] = {
            "height": _parse_height(row["HEIGHT"]),
            "reach": _parse_reach(row["REACH"]),
            "stance": stance.strip(),
            "dob": dob,
        }
    return attrs


def aggregate_fight_stats(stats):
    """Sum per-round rows into one row per (event, bout, fighter)."""
    parsed = stats.copy()
    sig = parsed["SIG.STR."].map(_parse_of)
    td = parsed["TD"].map(_parse_of)
    parsed["sig_landed"] = [x[0] for x in sig]
    parsed["td_landed"] = [x[0] for x in td]
    parsed["kd"] = pd.to_numeric(parsed["KD"], errors="coerce").fillna(0)
    parsed["sub_att"] = pd.to_numeric(parsed["SUB.ATT"], errors="coerce").fillna(0)
    parsed["ctrl_sec"] = parsed["CTRL"].map(_parse_ctrl)
    agg = parsed.groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False)[
        ["sig_landed", "td_landed", "kd", "sub_att", "ctrl_sec"]
    ].sum()
    return {(r.EVENT, r.BOUT, r.FIGHTER): r for r in agg.itertuples(index=False)}


def _method_class(method):
    """Collapse the METHOD string into KO / Sub / Dec."""
    if "KO/TKO" in method:
        return "KO"
    if "Submission" in method:
        return "Sub"
    return "Dec"


def _new_career():
    return {
        "n": 0, "wins": 0, "losses": 0, "streak": 0, "elo": ELO_START,
        "sig_landed": 0.0, "sig_absorbed": 0.0, "td": 0.0, "sub_att": 0.0,
        "kd": 0.0, "ctrl": 0.0, "seconds": 0.0, "ko_wins": 0, "sub_wins": 0,
    }


def _snapshot(career, attrs, fight_date):
    """Pre-fight feature snapshot for one fighter."""
    minutes = career["seconds"] / 60.0
    age = np.nan
    dob = attrs.get("dob")
    if dob is not None and pd.notna(dob):
        age = (fight_date - dob).days / 365.25
    wins, losses = career["wins"], career["losses"]
    return {
        "height": attrs.get("height", np.nan),
        "reach": attrs.get("reach", np.nan),
        "age": age,
        "stance_code": 1.0 if attrs.get("stance") == "Southpaw" else 0.0,
        "wins": float(wins),
        "losses": float(losses),
        "win_streak": float(career["streak"]),
        "elo": career["elo"],
        "slpm": career["sig_landed"] / minutes if minutes > 0 else 0.0,
        "sapm": career["sig_absorbed"] / minutes if minutes > 0 else 0.0,
        "td_avg": career["td"] / minutes * 15 if minutes > 0 else 0.0,
        "sub_avg": career["sub_att"] / minutes * 15 if minutes > 0 else 0.0,
        "kd_avg": career["kd"] / minutes * 15 if minutes > 0 else 0.0,
        "ctrl_avg": career["ctrl"] / minutes if minutes > 0 else 0.0,
        "ko_rate": career["ko_wins"] / wins if wins > 0 else 0.0,
        "sub_rate": career["sub_wins"] / wins if wins > 0 else 0.0,
        "n_fights": career["n"],
    }


def diff_row(a, b):
    """Differential feature dict from two snapshots (A minus B)."""
    row = {}
    for c in SNAPSHOT_COLS:
        if c == "stance_code":
            continue
        row[f"{c}_diff"] = a[c] - b[c]
    row["southpaw_diff"] = a["stance_code"] - b["stance_code"]
    return row


def build_dataset(seed=42):
    """Returns (train_df, latest fighter snapshots)."""
    events, results, stats, fighters = load_raw()
    attrs = build_fighter_attributes(fighters)
    stat_lookup = aggregate_fight_stats(stats)

    events["date"] = pd.to_datetime(events["DATE"].astype(str).str.strip(),
                                    errors="coerce", format="mixed")
    event_dates = dict(zip(events["EVENT"], events["date"]))

    results["date"] = results["EVENT"].map(event_dates)
    results = results.dropna(subset=["date"])
    results = results[results["date"] <= pd.Timestamp.now()]
    results = results.sort_values("date").reset_index(drop=True)

    careers = {}
    rows = []
    rng = np.random.RandomState(seed)

    for r in results.itertuples(index=False):
        bout_fighters = [f.strip() for f in r.BOUT.split(" vs. ")]
        if len(bout_fighters) != 2:
            continue
        f1, f2 = bout_fighters
        outcome = r.OUTCOME.strip()
        if outcome not in ("W/L", "L/W"):
            continue  # skip draws / no-contests
        winner = f1 if outcome == "W/L" else f2

        c1 = careers.setdefault(f1, _new_career())
        c2 = careers.setdefault(f2, _new_career())

        if c1["n"] >= MIN_PRIOR_FIGHTS and c2["n"] >= MIN_PRIOR_FIGHTS:
            s1 = _snapshot(c1, attrs.get(f1, {}), r.date)
            s2 = _snapshot(c2, attrs.get(f2, {}), r.date)
            # Randomize corner assignment to avoid label bias.
            if rng.rand() < 0.5:
                a, b, fa, fb = s1, s2, f1, f2
            else:
                a, b, fa, fb = s2, s1, f2, f1
            row = {"fighter_a": fa, "fighter_b": fb, "date": r.date,
                   "label": 1 if winner == fa else 0,
                   "method": _method_class(str(r.METHOD))}
            row.update(diff_row(a, b))
            rows.append(row)

        seconds = _fight_seconds(r.ROUND, r.TIME)
        method = str(r.METHOD)
        for me, opp in ((f1, f2), (f2, f1)):
            career = careers[me]
            mine = stat_lookup.get((r.EVENT, r.BOUT, me))
            theirs = stat_lookup.get((r.EVENT, r.BOUT, opp))
            career["n"] += 1
            career["seconds"] += seconds
            if mine is not None:
                career["sig_landed"] += mine.sig_landed
                career["td"] += mine.td_landed
                career["sub_att"] += mine.sub_att
                career["kd"] += mine.kd
                career["ctrl"] += mine.ctrl_sec
            if theirs is not None:
                career["sig_absorbed"] += theirs.sig_landed
            if me == winner:
                career["wins"] += 1
                career["streak"] = max(career["streak"], 0) + 1
                if "KO/TKO" in method:
                    career["ko_wins"] += 1
                elif "Submission" in method:
                    career["sub_wins"] += 1
            else:
                career["losses"] += 1
                career["streak"] = min(career["streak"], 0) - 1

        # Elo update (winner/loser already decided; draws/NC were skipped).
        loser = f2 if winner == f1 else f1
        k = ELO_K_FINISH if ("KO/TKO" in method or "Submission" in method) else ELO_K
        expected_winner = 1.0 / (1.0 + 10 ** ((careers[loser]["elo"] - careers[winner]["elo"]) / 400.0))
        delta = k * (1.0 - expected_winner)
        careers[winner]["elo"] += delta
        careers[loser]["elo"] -= delta

    df = pd.DataFrame(rows).dropna(subset=MODEL_FEATURES)

    # Latest snapshot per fighter, for the app's lookup database.
    now = pd.Timestamp.now()
    snapshots = {}
    for name, career in careers.items():
        if career["n"] < 1:
            continue
        snap = _snapshot(career, attrs.get(name, {}), now)
        if any(pd.isna(snap[c]) for c in SNAPSHOT_COLS):
            continue
        snap["name"] = name
        snapshots[name] = snap
    return df, snapshots


def save_fighter_db(snapshots, path=f"{DATA_DIR}/fighters.json"):
    clean = {}
    for name, snap in snapshots.items():
        clean[name] = {k: (round(float(v), 3) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in snap.items()}
    with open(path, "w") as f:
        json.dump(clean, f)
    return path


if __name__ == "__main__":
    df, snapshots = build_dataset()
    df.to_csv(f"{DATA_DIR}/training_data.csv", index=False)
    save_fighter_db(snapshots)
    print(f"Training rows: {len(df)}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"Fighter snapshots: {len(snapshots)}")
    print(f"Label balance: {df['label'].mean():.3f}")
