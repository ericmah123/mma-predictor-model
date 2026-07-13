import { useEffect, useRef, useState } from "react";
import { FighterCard } from "./components/FighterCard";
import { ResultsPanel } from "./components/ResultsPanel";
import { predictMatchup } from "./api";
import type { ApiError, FighterSearchResult, PredictResponse } from "./types";

function App() {
  const [fighterA, setFighterA] = useState<FighterSearchResult | null>(null);
  const [fighterB, setFighterB] = useState<FighterSearchResult | null>(null);
  const [resetKey, setResetKey] = useState(0);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [predictionId, setPredictionId] = useState(0);

  const inputARef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (resetKey > 0) inputARef.current?.focus();
  }, [resetKey]);

  const canPredict =
    !predicting && !!fighterA && !!fighterB && fighterA.name !== fighterB.name;

  async function handlePredict() {
    if (!fighterA || !fighterB) return;
    setError(null);
    setResult(null);
    setPredicting(true);
    try {
      const data = await predictMatchup(fighterA.name, fighterB.name);
      setResult(data);
      setPredictionId((id) => id + 1);
    } catch (err) {
      setError((err as ApiError).message);
    } finally {
      setPredicting(false);
    }
  }

  function handleReset() {
    setFighterA(null);
    setFighterB(null);
    setError(null);
    setResult(null);
    setResetKey((k) => k + 1);
  }

  const favoriteCorner = result
    ? result.fighter_a.prob >= result.fighter_b.prob
      ? "a"
      : "b"
    : null;

  return (
    <main className="page">
      <header className="hero">
        <div>
          <p className="meta-tag">Gradient-boosted model · Real UFC fights since 1997</p>
          <h1>MMA Fight Predictor</h1>
          <p className="subtitle">
            Pick two UFC fighters and get head-to-head win probabilities from a
            gradient-boosted model trained on thousands of real fight outcomes.
          </p>
        </div>
        <div className="hero-card">
          <h2>How it works</h2>
          <p>
            Every fighter's career stats — striking, grappling, finish rates, streaks and
            physical attributes — are compared as differentials. The model learned from real
            results, evaluated on fights it never saw.
          </p>
        </div>
      </header>

      <section className="matchup">
        <FighterCard
          key={`a-${resetKey}`}
          corner="a"
          ref={inputARef}
          isWinner={favoriteCorner === "a"}
          onSelect={setFighterA}
        />
        <div className="vs">VS</div>
        <FighterCard
          key={`b-${resetKey}`}
          corner="b"
          isWinner={favoriteCorner === "b"}
          onSelect={setFighterB}
        />
      </section>

      <div className="actions">
        <button
          type="button"
          className="predict-btn"
          disabled={!canPredict}
          onClick={handlePredict}
        >
          {predicting ? "Predicting…" : "Predict fight"}
        </button>
        <button type="button" className="reset-btn" onClick={handleReset}>
          Reset
        </button>
      </div>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {result && <ResultsPanel key={predictionId} result={result} />}
    </main>
  );
}

export default App;
