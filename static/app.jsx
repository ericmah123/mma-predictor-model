const { useMemo } = React;

const initialData = window.__INITIAL_DATA__ || {};

const formatPercent = (value) => `${(value * 100).toFixed(2)}%`;

function App() {
  const hasResults = useMemo(
    () =>
      typeof initialData.fighter1Odds === "number" &&
      typeof initialData.fighter2Odds === "number",
    []
  );

  return (
    <main className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">MMA Fight Predictor</p>
          <h1>Predict the edge before the bell.</h1>
          <p className="subtitle">
            Enter fighter stats to estimate win probabilities and compare
            strengths at a glance.
          </p>
        </div>
        <div className="hero-card">
          <h2>How it works</h2>
          <ul>
            <li>Enter core stats for each fighter.</li>
            <li>Submit to calculate win probabilities.</li>
            <li>Use results as a quick comparison tool.</li>
          </ul>
        </div>
      </header>

      <section className="form-section">
        <form className="fight-form" method="POST">
          <div className="fighter-card">
            <div className="fighter-header">
              <span className="badge">Corner A</span>
              <h2>Fighter 1</h2>
              <p>Power, precision, and pace.</p>
            </div>
            <label>
              Height (in inches)
              <input type="number" step="any" name="height_1" required />
            </label>
            <label>
              Weight (in pounds)
              <input type="number" step="any" name="weight_1" required />
            </label>
            <label>
              Reach (in inches)
              <input type="number" step="any" name="reach_1" required />
            </label>
            <label>
              Age
              <input type="number" step="any" name="age_1" required />
            </label>
            <label>
              Significant strikes landed / min
              <input type="number" step="any" name="strikes_1" required />
            </label>
            <label>
              Average takedowns / 15 min
              <input type="number" step="any" name="takedowns_1" required />
            </label>
            <label>
              Win/loss ratio
              <input type="number" step="any" name="ratio_1" required />
            </label>
            <label>
              Experience (total fights)
              <input type="number" step="any" name="experience_1" required />
            </label>
            <label>
              Stance
              <select name="stance_1" required>
                <option value="Orthodox">Orthodox</option>
                <option value="Southpaw">Southpaw</option>
              </select>
            </label>
          </div>

          <div className="fighter-card">
            <div className="fighter-header">
              <span className="badge badge-alt">Corner B</span>
              <h2>Fighter 2</h2>
              <p>Strength, timing, and control.</p>
            </div>
            <label>
              Height (in inches)
              <input type="number" step="any" name="height_2" required />
            </label>
            <label>
              Weight (in pounds)
              <input type="number" step="any" name="weight_2" required />
            </label>
            <label>
              Reach (in inches)
              <input type="number" step="any" name="reach_2" required />
            </label>
            <label>
              Age
              <input type="number" step="any" name="age_2" required />
            </label>
            <label>
              Significant strikes landed / min
              <input type="number" step="any" name="strikes_2" required />
            </label>
            <label>
              Average takedowns / 15 min
              <input type="number" step="any" name="takedowns_2" required />
            </label>
            <label>
              Win/loss ratio
              <input type="number" step="any" name="ratio_2" required />
            </label>
            <label>
              Experience (total fights)
              <input type="number" step="any" name="experience_2" required />
            </label>
            <label>
              Stance
              <select name="stance_2" required>
                <option value="Orthodox">Orthodox</option>
                <option value="Southpaw">Southpaw</option>
              </select>
            </label>
          </div>

          <div className="form-actions">
            <button type="submit">Calculate probabilities</button>
            <p>
              Results will appear below once the prediction runs.
            </p>
          </div>
        </form>
      </section>

      {hasResults && (
        <section className="results">
          <div className="results-card">
            <h2>Prediction results</h2>
            <p className="results-subtitle">
              Win probabilities from the trained model pipeline.
            </p>
            <div className="results-grid">
              <div>
                <span>Fighter 1</span>
                <strong>{formatPercent(initialData.fighter1Odds)}</strong>
              </div>
              <div>
                <span>Fighter 2</span>
                <strong>{formatPercent(initialData.fighter2Odds)}</strong>
              </div>
            </div>
          </div>
        </section>
      )}
    </main>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
