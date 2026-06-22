import React from "react";
import { STATUS, BADGE, fmtElapsed, fmtAudio } from "./utils.js";

export default function Column({ alias, state }) {
  const [elapsed, setElapsed] = React.useState(0);
  const isRunning = state.status === STATUS.run;

  React.useEffect(() => {
    if (!isRunning) { setElapsed(0); return; }
    setElapsed(0);
    const id = setInterval(() => setElapsed(s => s + 1), 1000);
    return () => clearInterval(id);
  }, [isRunning]);

  const [cls, label] = BADGE[state.status];
  return (
    <div className="col">
      {isRunning && <div className="progress-bar" />}
      <div className="col-inner">
        <div className="col-header">
          <span className="col-title">{alias}</span>
          <span className={`badge ${cls}`}>{label}</span>
          {isRunning && <span className="elapsed">{fmtElapsed(elapsed)}</span>}
        </div>
        {state.status === STATUS.wait  && <div className="placeholder">—</div>}
        {isRunning                      && <div className="placeholder">מעבד...</div>}
        {state.status === STATUS.error  && <div className="error-msg">{state.error}</div>}
        {state.status === STATUS.done && state.data && (
          <>
            {state.data.chunks?.length > 0 && (
              <div className="chunks">
                {state.data.chunks.map((c, i) => (
                  <div key={i} className="chunk">
                    <span className="ts">{fmtAudio(c.start)} → {fmtAudio(c.end)}</span>
                    <span>{c.text}</span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
