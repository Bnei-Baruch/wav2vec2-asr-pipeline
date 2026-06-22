import React from "react";
import { MODELS, STATUS } from "./utils.js";

const ISSUE_LABEL = {
  recognition_error: "שגיאת זיהוי",
  typo: "שגיאת כתיב",
  nonsense: "חסר משמעות",
  punctuation: "פיסוק",
  spacing: "רווחים",
  other: "אחר",
};

// Block under the player: pick a model's transcription and send it to the
// Claude-backed /check endpoint, which reviews it for likely ASR errors.
export default function Check({ cols }) {
  const [selected, setSelected] = React.useState("");
  const [status, setStatus] = React.useState(STATUS.wait);
  const [result, setResult] = React.useState(null);
  const [error, setError] = React.useState(null);

  // Models that finished with usable text can be reviewed.
  const available = MODELS.filter(
    m => cols[m]?.status === STATUS.done && cols[m]?.data?.text
  );

  React.useEffect(() => {
    setSelected(prev => (prev && available.includes(prev) ? prev : (available[0] || "")));
  }, [available.join(",")]);

  const running = status === STATUS.run;

  async function handleCheck() {
    const text = cols[selected]?.data?.text;
    if (!text) return;
    setStatus(STATUS.run);
    setResult(null);
    setError(null);
    try {
      const res = await fetch("/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
      }
      setResult(await res.json());
      setStatus(STATUS.done);
    } catch (err) {
      setError(err.message);
      setStatus(STATUS.error);
    }
  }

  return (
    <div className="check-card">
      <div className="check-controls">
        <select
          value={selected}
          onChange={e => setSelected(e.target.value)}
          disabled={running || !available.length}
        >
          {available.length
            ? available.map(m => <option key={m} value={m}>{m}</option>)
            : <option value="">אין תמלול לבדיקה</option>}
        </select>
        <button
          type="button"
          className="check-btn"
          onClick={handleCheck}
          disabled={running || !selected}
        >
          {running ? "בודק..." : "בדוק עם Claude"}
        </button>
      </div>

      {error && <div className="error-msg">{error}</div>}

      {result && (
        <div className="check-result">
          <div className="check-score">
            <span className="check-score-num">{result.score}</span>
            <span className="check-score-max">/100</span>
          </div>
          {result.summary && <p className="check-summary">{result.summary}</p>}

          {result.issues?.length > 0 ? (
            <ul className="check-issues">
              {result.issues.map((it, i) => (
                <li key={i} className="check-issue">
                  <span className="issue-type">{ISSUE_LABEL[it.type] || it.type}</span>
                  <span className="issue-span">{it.span}</span>
                  {it.suggestion && <span className="issue-arrow">→</span>}
                  {it.suggestion && <span className="issue-fix">{it.suggestion}</span>}
                  {it.explanation && <span className="issue-why">{it.explanation}</span>}
                </li>
              ))}
            </ul>
          ) : (
            <p className="check-clean">לא נמצאו שגיאות בולטות ✓</p>
          )}

          {result.corrected_text && (
            <div className="check-corrected">
              <label>טקסט מתוקן</label>
              <p>{result.corrected_text}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
