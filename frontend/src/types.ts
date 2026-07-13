export interface FighterSearchResult {
  name: string;
  record: string;
  fights: number;
}

export interface FighterPrediction {
  name: string;
  prob: number;
  record: string;
}

export interface MethodProbabilities {
  KO?: number;
  Sub?: number;
  Dec?: number;
}

export interface ComparisonRow {
  stat: string;
  a: number;
  b: number;
  favors: "a" | "b" | null;
}

export interface PredictResponse {
  fighter_a: FighterPrediction;
  fighter_b: FighterPrediction;
  method: MethodProbabilities;
  comparison: ComparisonRow[];
}

export interface JustifyResponse {
  summary: string;
  path_a: string;
  path_b: string;
}

export interface ApiError extends Error {
  status?: number;
}
