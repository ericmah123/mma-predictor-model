import { useEffect, useRef } from "react";
import { motion, useReducedMotion } from "motion/react";
import { JustifyCard } from "./JustifyCard";
import type { PredictResponse } from "../types";

interface ResultsPanelProps {
  result: PredictResponse;
}

const EASE_OUT_QUART = [0.25, 1, 0.5, 1] as const;

const METHOD_LABELS: Record<string, string> = { KO: "KO/TKO", Sub: "Submission", Dec: "Decision" };

export function ResultsPanel({ result }: ResultsPanelProps) {
  const { fighter_a: a, fighter_b: b, method, comparison } = result;
  const pa = Math.round(a.prob * 100);
  const pb = 100 - pa;
  const favorite = a.prob >= b.prob ? a : b;

  const reducedMotion = useReducedMotion();
  const rootRef = useRef<HTMLElement>(null);

  useEffect(() => {
    rootRef.current?.scrollIntoView({
      behavior: reducedMotion ? "auto" : "smooth",
      block: "nearest",
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <motion.section
      className="results"
      aria-live="polite"
      ref={rootRef}
      initial={reducedMotion ? false : { opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: EASE_OUT_QUART }}
    >
      <h2>{favorite.name} is favored to win</h2>
      <div className="prob-bar">
        <motion.div
          className="prob-fill-a"
          style={{ width: `${pa}%` }}
          initial={reducedMotion ? false : { scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 0.5, ease: EASE_OUT_QUART }}
        />
        <div className="prob-fill-b" />
      </div>
      <div className="prob-labels">
        <span>
          {a.name} — {pa}%
        </span>
        <span>
          {pb}% — {b.name}
        </span>
      </div>
      <div className="method-section">
        <h3>Likely finish</h3>
        <div className="method-chips">
          {(["KO", "Sub", "Dec"] as const).map((cls) => (
            <div className="method-chip" key={cls}>
              <span className="method-label">{METHOD_LABELS[cls]}</span>
              <span className="method-value">
                {Math.round((method[cls] ?? 0) * 100)}%
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="compare-scroll">
        <table className="compare">
          <thead>
            <tr>
              <th scope="col">{a.name}</th>
              <th scope="col">Stat</th>
              <th scope="col">{b.name}</th>
            </tr>
          </thead>
          <tbody>
            {comparison.map((row) => (
              <tr key={row.stat}>
                <td className={row.favors === "a" ? "favors" : row.favors === "b" ? "trails" : ""}>
                  {row.a}
                </td>
                <td className="stat-name">{row.stat}</td>
                <td className={row.favors === "b" ? "favors" : row.favors === "a" ? "trails" : ""}>
                  {row.b}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <JustifyCard fighterAName={a.name} fighterBName={b.name} />
      <p className="disclaimer">
        Model accuracy on held-out recent fights is ~63%. MMA is volatile — treat this as an
        informed estimate, not a guarantee.
      </p>
    </motion.section>
  );
}
