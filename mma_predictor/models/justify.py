"""AI-generated fight analysis via OpenRouter.

Requires the OPENROUTER_API_KEY environment variable. The prompt is built
strictly from the model's own prediction payload so the blurb stays
grounded in the data.
"""

import os

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
TIMEOUT_SECONDS = 15


class JustificationUnavailable(Exception):
    """OPENROUTER_API_KEY is not configured."""


class JustificationFailed(Exception):
    """The upstream OpenRouter call failed."""


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
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise JustificationUnavailable()

    model = os.environ.get("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": build_prompt(result)}],
            },
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        raise JustificationFailed(str(exc)) from exc
