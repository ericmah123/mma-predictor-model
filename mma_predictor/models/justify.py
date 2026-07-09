"""AI-generated fight analysis via the Google Gemini free tier.

Requires the GEMINI_API_KEY environment variable. The prompt is built
strictly from the model's own prediction payload so the blurb stays
grounded in the data.
"""

import os
import re

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
        "Respond with exactly three labeled lines, in this exact order, and "
        "nothing before, between, or after them (no markdown, no headers, no "
        "bullet points, no extra commentary):\n"
        f"SUMMARY: <3-4 sentences explaining why the model favors {favorite['name']}, "
        "referencing only the statistics above>\n"
        f"A_PATH: <one short sentence, 20 words or fewer, on {a['name']}'s clearest "
        "path to victory in this fight, grounded only in the stats above>\n"
        f"B_PATH: <one short sentence, 20 words or fewer, on {b['name']}'s clearest "
        "path to victory in this fight, grounded only in the stats above>\n\n"
        "Each labeled field must be plain text on a single line with no internal "
        "line breaks. Do not invent facts, fights, or injuries not present in "
        "the data. Write for a fan, not a statistician."
    )


_LABEL_PATTERN = re.compile(
    r"^[\s*_#-]*(SUMMARY|A_PATH|B_PATH)[\s*_#-]*:\s*\**\s*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_KEY_MAP = {"summary": "summary", "a_path": "path_a", "b_path": "path_b"}


def parse_justification(text):
    """Parse the SUMMARY/A_PATH/B_PATH labeled Gemini response into a dict
    with keys "summary", "path_a", "path_b". Falls back to treating the whole
    response as the summary if the model didn't follow the label format, so a
    formatting slip never takes down the feature.
    """
    result = {"summary": "", "path_a": "", "path_b": ""}
    matches = list(_LABEL_PATTERN.finditer(text))
    if not matches:
        result["summary"] = text.strip()
        return result

    for i, m in enumerate(matches):
        key = _KEY_MAP.get(m.group(1).lower())
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = " ".join((m.group(2) + text[m.end():end]).split())
        if key and value:
            result[key] = value

    if not result["summary"]:
        result["summary"] = text.strip()
    return result


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
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except requests.HTTPError as exc:
        raise JustificationFailed(f"{exc} - body: {resp.text[:500]}") from exc
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        raise JustificationFailed(f"{exc!r} - raw response: {resp.text[:500] if 'resp' in locals() else 'n/a'}") from exc

    return parse_justification(text)
