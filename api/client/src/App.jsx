import React from "react";
import { MODELS, STATUS, initCols } from "./utils.js";
import Column from "./Column.jsx";
import Player from "./Player.jsx";
import Check from "./Check.jsx";

export default function App() {
  const [audioUrl, setAudioUrl] = React.useState("");
  const [file, setFile] = React.useState(null);
  const [running, setRunning] = React.useState(false);
  const [cols, setCols] = React.useState(initCols);

  function setCol(alias, patch) {
    setCols(prev => ({ ...prev, [alias]: { ...prev[alias], ...patch } }));
  }

  // Media source for the player: the URL, or an object URL for the uploaded file.
  const mediaSrc = React.useMemo(() => {
    if (audioUrl.trim()) return audioUrl.trim();
    if (file) return URL.createObjectURL(file);
    return "";
  }, [audioUrl, file]);

  React.useEffect(() => {
    if (mediaSrc.startsWith("blob:")) return () => URL.revokeObjectURL(mediaSrc);
  }, [mediaSrc]);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file && !audioUrl.trim()) return;
    setRunning(true);
    setCols(initCols());

    for (const alias of MODELS) {
      setCol(alias, { status: STATUS.run });
      try {
        let res;
        if (audioUrl.trim()) {
          res = await fetch(`/stt?model=${alias}&url=${encodeURIComponent(audioUrl.trim())}`, { method: "POST" });
        } else {
          const fd = new FormData();
          fd.append("file", file);
          res = await fetch(`/stt?model=${alias}`, { method: "POST", body: fd });
        }
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || res.statusText);
        }
        const data = await res.json();
        setCol(alias, { status: STATUS.done, data });
      } catch (err) {
        setCol(alias, { status: STATUS.error, error: err.message });
        break;
      }
    }
    setRunning(false);
  }

  return (
    <>
      <h1>STT — השוואת מודלים</h1>
      <div className="form-card">
        <form onSubmit={handleSubmit}>
          <div className="input-row">
            <div className="input-group">
              <label>קישור MP3</label>
              <input
                type="url"
                placeholder="https://example.com/audio.mp3"
                value={audioUrl}
                onChange={e => { setAudioUrl(e.target.value); if (e.target.value) setFile(null); }}
                disabled={running}
              />
            </div>
            <div className="divider">או</div>
            <div className="input-group" style={{flexGrow:0}}>
              <label>קובץ</label>
              <label className="file-label">
                📎 {file ? file.name : "בחר קובץ"}
                <input type="file" accept="audio/*"
                  onChange={e => { setFile(e.target.files[0]); setAudioUrl(""); }}
                  disabled={running} />
              </label>
            </div>
            <button type="submit" disabled={running || (!file && !audioUrl.trim())}>
              {running ? "מעבד..." : "השווה"}
            </button>
          </div>
        </form>
      </div>

      <div className="layout">
        <div className="layout-left">
          <Player src={mediaSrc} cols={cols} />
          <Check cols={cols} />
        </div>
        <div className="layout-right">
          <div className="columns" style={{ gridTemplateColumns: `repeat(${MODELS.length}, 1fr)` }}>
            {MODELS.map(alias => <Column key={alias} alias={alias} state={cols[alias]} />)}
          </div>
        </div>
      </div>
    </>
  );
}
