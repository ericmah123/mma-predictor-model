"""AI-generated fight analysis via the Google Gemini free tier.

Requires the GEMINI_API_KEY environment variable. The prompt is built
strictly from the model's own prediction payload so the blurb stays
grounded in the data.
"""

import os

import requests

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-flash-latest:generateContent"
)
TIMEOUT_SECONDS = 15


class JustificationUnavailable(Exception):
    """GEMINI_API_KEY is not configured."""


class JustificationFailed(Exception):
    """The upstream Gemini call failed."""


def build_prompt(result):
    a, b = result["fighter_a"], result["fighter_b"]
    favorite = a if a["prob"] >= b["prob"] else b
    method = result.get("method", {})

    stat_lines = []
    for row in result["comparison"]:
        favors = {"a": a["name"], "b": b["name"]}.get(row["favors"], "even")
        stat_lines.append(
            f"- {row['stat']}: {a['name']} {row['a']} vs {b['name']} {row['b']}"
            f" (favors {favors})"
        )

    return (
        "You are an MMA analyst. A statistical model compared two UFC fighters.\n\n"
        f"Matchup: {a['name']} (UFC record {a['record']}) vs "
        f"{b['name']} (UFC record {b['record']})\n"
        f"Model win probability: {a['name']} {a['prob']:.0%}, {b['name']} {b['prob']:.0%}\n"
        f"Likely finish: KO/TKO {method.get('KO', 0):.0%}, "
        f"Submission {method.get('Sub', 0):.0%}, Decision {method.get('Dec', 0):.0%}\n"
        "Stat comparison:\n" + "\n".join(stat_lines) + "\n\n"
        f"In 3-4 sentences, explain why the model favors {favorite['name']}, "
        "referencing only the statistics above. Do not invent facts, fights, or "
        "injuries not present in the data. Write for a fan, not a statistician."
    )


def generate_justification(result):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise JustificationUnavailable()

    try:
        resp = requests.post(
            GEMINI_URL,
            headers={"x-goog-api-key": api_key},
            json={"contents": [{"parts": [{"text": build_prompt(result)}]}]},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        raise JustificationFailed(str(exc)) from exc
