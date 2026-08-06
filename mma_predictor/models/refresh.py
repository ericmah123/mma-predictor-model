"""Weekly data refresh: pull fresh raw CSVs from Greco1899/scrape_ufc_stats
(they scrape ufcstats.com daily; we don't need our own scraper), sanity-check
the pull, rebuild the training set + fighter DB, retrain both models, and
check for a metrics regression before the result is considered safe to land.

Run directly: `python -m mma_predictor.models.refresh`.
Exit code 0 means the refresh passed every check and is safe to commit/merge.
Exit code 1 means a check failed and the result needs a human look — the
rebuilt files (including mma_predictor/data/model_metrics.json) are still
left on disk so a PR/diff can show what was flagged; `git checkout --
mma_predictor/data` discards them if you're testing locally and want a
clean tree afterwards.

See .github/workflows/weekly-data-refresh.yml for how this is scheduled.
"""

import json
from pathlib import Path

import requests

from mma_predictor.models import main as pipeline_main
from mma_predictor.models import train
from mma_predictor.models.preprocess import DATA_DIR

RAW_BASE_URL = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/master"
RAW_FILES = [
    "ufc_event_details.csv",
    "ufc_fight_results.csv",
    "ufc_fight_stats.csv",
    "ufc_fighter_tott.csv",
]

MAX_GROWTH_FRACTION = 0.20  # a single week's data shouldn't grow more than this
MAX_ACCURACY_DROP = 0.02  # 2 percentage points
MAX_LOG_LOSS_INCREASE_FRACTION = 0.05

REQUEST_TIMEOUT_SECONDS = 30


class RefreshCheckFailed(Exception):
    """A sanity or regression check failed; this refresh must not be
    treated as safe to auto-merge."""


def _row_count(source):
    """Count data rows (excluding header) in a CSV file path or raw bytes."""
    text = Path(source).read_text() if isinstance(source, (str, Path)) else source.decode("utf-8")
    return max(text.count("\n") - 1, 0)


def check_row_growth(filename, old_count, new_count):
    if new_count < old_count:
        raise RefreshCheckFailed(
            f"{filename}: row count dropped ({old_count} -> {new_count}); "
            "likely a bad fetch."
        )
    if old_count > 0 and (new_count - old_count) / old_count > MAX_GROWTH_FRACTION:
        raise RefreshCheckFailed(
            f"{filename}: row count jumped implausibly ({old_count} -> "
            f"{new_count}, >{MAX_GROWTH_FRACTION:.0%} growth in a week); "
            "likely the wrong file or an upstream scrape error."
        )


def fetch_fresh_csvs(dest_dir=DATA_DIR):
    """Download the raw CSVs from Greco1899/scrape_ufc_stats, sanity-check
    each against the currently-committed version, and only overwrite once
    every file has passed. Returns {filename: (old_rows, new_rows)}.
    Raises RefreshCheckFailed (leaving files untouched) if any check fails.
    """
    report = {}
    fetched = {}
    for filename in RAW_FILES:
        dest_path = Path(dest_dir) / filename
        old_count = _row_count(dest_path) if dest_path.exists() else 0
        resp = requests.get(f"{RAW_BASE_URL}/{filename}", timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        new_count = _row_count(resp.content)
        check_row_growth(filename, old_count, new_count)
        fetched[filename] = resp.content
        report[filename] = (old_count, new_count)

    # Only write once every file has passed its own check, so a failure
    # partway through never leaves a half-updated set of raw CSVs on disk.
    for filename, content in fetched.items():
        (Path(dest_dir) / filename).write_bytes(content)
    return report


def check_model_regression(new_metrics, baseline=None, baseline_path=None):
    """Compare freshly-trained metrics against the last known-good
    baseline (an in-memory dict, or read from model_metrics.json if not
    given). Raises RefreshCheckFailed if either model regressed beyond the
    configured tolerance. No-ops if no baseline exists yet (first-ever
    run)."""
    if baseline is None:
        baseline_file = Path(baseline_path or train.METRICS_PATH)
        if not baseline_file.exists():
            return
        baseline = json.loads(baseline_file.read_text())

    old_acc = baseline["win_model"]["accuracy"]
    new_acc = new_metrics["win_model"]["accuracy"]
    if old_acc - new_acc > MAX_ACCURACY_DROP:
        raise RefreshCheckFailed(
            f"Win model accuracy dropped {old_acc:.4f} -> {new_acc:.4f} "
            f"(more than {MAX_ACCURACY_DROP:.0%})."
        )

    old_ll = baseline["win_model"]["log_loss"]
    new_ll = new_metrics["win_model"]["log_loss"]
    if old_ll > 0 and (new_ll - old_ll) / old_ll > MAX_LOG_LOSS_INCREASE_FRACTION:
        raise RefreshCheckFailed(
            f"Win model log loss increased {old_ll:.4f} -> {new_ll:.4f} "
            f"(more than {MAX_LOG_LOSS_INCREASE_FRACTION:.0%})."
        )


def main():
    # Snapshot the current baseline *before* rebuilding overwrites the file,
    # so the regression check below compares against last week's
    # known-good numbers rather than the ones we're about to produce.
    baseline_path = Path(train.METRICS_PATH)
    old_baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None

    print("Fetching fresh CSVs from Greco1899/scrape_ufc_stats...")
    try:
        report = fetch_fresh_csvs()
    except RefreshCheckFailed as exc:
        print(f"FAIL: raw data sanity check failed, nothing was changed.\n  {exc}")
        return 1
    for filename, (old, new) in report.items():
        print(f"  {filename}: {old} -> {new} rows")

    print("Rebuilding training data + fighter DB + models...")
    pipeline_main.main()  # also overwrites model_metrics.json with new_metrics

    new_metrics = json.loads(baseline_path.read_text())
    try:
        check_model_regression(new_metrics, baseline=old_baseline)
    except RefreshCheckFailed as exc:
        print(f"FAIL: model regression check failed.\n  {exc}")
        print("Rebuilt files (including the regressed model_metrics.json) "
              "are left on disk for inspection, but this refresh should "
              "NOT be auto-merged.")
        return 1

    print("PASS: raw data and model both look healthy. Safe to merge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
