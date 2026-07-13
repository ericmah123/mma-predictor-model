import type {
  ApiError,
  FighterSearchResult,
  JustifyResponse,
  PredictResponse,
} from "./types";

async function parseOrThrow<T>(response: Response): Promise<T> {
  const data = await response.json();
  if (!response.ok) {
    const error: ApiError = new Error(data.error || "Request failed.");
    error.status = response.status;
    throw error;
  }
  return data as T;
}

export async function searchFighters(query: string): Promise<FighterSearchResult[]> {
  const response = await fetch(`/api/fighters?q=${encodeURIComponent(query)}`);
  return parseOrThrow<FighterSearchResult[]>(response);
}

export async function predictMatchup(
  fighterA: string,
  fighterB: string,
): Promise<PredictResponse> {
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fighter_a: fighterA, fighter_b: fighterB }),
  });
  return parseOrThrow<PredictResponse>(response);
}

export async function justifyMatchup(
  fighterA: string,
  fighterB: string,
): Promise<JustifyResponse> {
  const response = await fetch("/api/justify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fighter_a: fighterA, fighter_b: fighterB }),
  });
  return parseOrThrow<JustifyResponse>(response);
}
