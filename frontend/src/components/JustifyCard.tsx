import { useState } from "react";
import { justifyMatchup } from "../api";
import type { ApiError, JustifyResponse } from "../types";

interface JustifyCardProps {
  fighterAName: string;
  fighterBName: string;
}

type Status = "idle" | "loading" | "shown" | "unavailable" | "retry";

export function JustifyCard({ fighterAName, fighterBName }: JustifyCardProps) {
  const [status, setStatus] = useState<Status>("idle");
  const [data, setData] = useState<JustifyResponse | null>(null);

  if (status === "unavailable") return null;

  async function handleClick() {
    setStatus("loading");
    try {
      const result = await justifyMatchup(fighterAName, fighterBName);
      setData(result);
      setStatus("shown");
    } catch (err) {
      setStatus((err as ApiError).status === 503 ? "unavailable" : "retry");
    }
  }

  if (status === "shown" && data) {
    return (
      <div className="justify-section">
        <div className="justify-card">
          <p>{data.summary}</p>
          {(data.path_a || data.path_b) && (
            <ul className="justify-paths">
              <li>
                <strong>{fighterAName}: </strong>
                <span>{data.path_a}</span>
              </li>
              <li>
                <strong>{fighterBName}: </strong>
                <span>{data.path_b}</span>
              </li>
            </ul>
          )}
          <p className="justify-note">
            AI-generated analysis — based only on the model's statistics.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="justify-section">
      <button
        type="button"
        className="justify-btn"
        disabled={status === "loading"}
        onClick={handleClick}
      >
        {status === "loading"
          ? "Analyzing…"
          : status === "retry"
            ? "✨ Explain this pick (retry)"
            : "✨ Explain this pick"}
      </button>
    </div>
  );
}
