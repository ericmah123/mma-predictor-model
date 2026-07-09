# CLAUDE.md

Guidance for working in this repo.

## Project

Flask app that predicts UFC fight outcomes head-to-head from a
gradient-boosted model trained on chronologically-replayed fight history
(see README.md for the modeling approach, DESIGN.md for visual/UX direction,
PRODUCT.md for audience and product intent).

## Git workflow

- **Do work on a feature branch, not a worktree.** Worktrees add directory
  indirection that's more confusing to track than it's worth for a
  single-developer repo like this — a plain branch is enough isolation.
- Once a change is implemented and verified to work, **merge it into `main`**
  (a real merge commit is fine — no need to squash or fast-forward-only).
  Don't leave verified work stranded on a branch.
- `main` is the only long-lived branch. Delete feature branches after they're
  merged instead of accumulating them.

## No automated test suite

There are no unit tests or CI in this repo. Verify changes by hand:
- For model/data logic: exercise the function directly (e.g. a short
  `python -c "..."` snippet against real data in `mma_predictor/data/`),
  the way `comparison()`/`build_prompt()`/`parse_justification()` were
  checked when they were added.
- For Flask routes: use `app.test_client()` or run the dev server and hit
  the endpoint.
- For UI changes: actually run the app and click through the flow — don't
  rely on code review alone.

## Model files are gitignored

`mma_predictor/models/*.pkl` are not committed. If they're missing, rebuild
them with:

```bash
python -m mma_predictor.models.main
```

This replays `mma_predictor/data/*.csv` chronologically to rebuild the
training set, the fighter DB (`fighters.json`), and both models
(win-probability + method-of-victory). Do this before trying to run the app
or `app.test_client()` locally if you hit a `FileNotFoundError` for a `.pkl`.

## Modeling integrity

The core design principle is **no data leakage**: every fighter's stats used
as model input are snapshotted strictly *before* the fight being predicted,
via chronological replay in `preprocess.py`. Any change to feature
engineering or the fighter DB must preserve this — never derive a feature
from a fight that hasn't happened yet relative to the snapshot it's used in.

## AI feature (Gemini) quota consciousness

`mma_predictor/models/justify.py` calls the Gemini free tier for the
"Explain this pick" feature. Keep usage frugal:
- One Gemini call per unique fighter matchup, no more — `app.py` already
  dedupes via an in-memory `justification_cache` keyed by the sorted
  fighter-name pair. Don't introduce loops or batch calls that fan out to
  many matchups.
- Prefer enriching the *content* of the single existing call (better prompt
  inputs, structured multi-field responses) over adding new call sites.
- If you add error handling for the upstream call, log the real
  status/body server-side rather than swallowing it silently — the 502 path
  in `app.py` does this so failures are diagnosable.

## Design/UI changes

`DESIGN.md` documents the visual system (palette, radius scale, typography)
and `PRODUCT.md` documents the audience and product intent — check both
before making frontend changes so new UI stays consistent with the existing
system rather than introducing drift.
