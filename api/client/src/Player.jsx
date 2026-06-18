import React from "react";
import { MODELS, STATUS, chunksToVtt } from "./utils.js";

export default function Player({ src, cols }) {
  const [videoUrl, setVideoUrl] = React.useState(src || "");
  const [selected, setSelected] = React.useState("");

  // Follow the form's source (URL typed above, or uploaded file), but allow manual edits.
  React.useEffect(() => { setVideoUrl(src || ""); }, [src]);

  // Models that finished with usable chunks can provide subtitles.
  const available = MODELS.filter(
    m => cols[m]?.status === STATUS.done && cols[m]?.data?.chunks?.length
  );

  // Keep the selection valid: default to the first available model.
  React.useEffect(() => {
    setSelected(prev => (prev && available.includes(prev) ? prev : (available[0] || "")));
  }, [available.join(",")]);

  // Build a WebVTT blob for the selected model's transcript.
  const trackUrl = React.useMemo(() => {
    if (!selected || !cols[selected]?.data?.chunks) return null;
    const vtt = chunksToVtt(cols[selected].data.chunks);
    return URL.createObjectURL(new Blob([vtt], { type: "text/vtt" }));
  }, [selected, cols]);

  React.useEffect(() => {
    return () => { if (trackUrl) URL.revokeObjectURL(trackUrl); };
  }, [trackUrl]);

  // Re-key the <video> on src + selected track so it reloads and applies the track.
  const videoKey = videoUrl + "|" + (selected || "");

  return (
    <div className="player-card">
      <div className="player-controls">
        <input
          type="url"
          className="player-url"
          placeholder="קישור וידאו / אודיו"
          value={videoUrl}
          onChange={e => setVideoUrl(e.target.value)}
        />
        <select value={selected} onChange={e => setSelected(e.target.value)}>
          <option value="">ללא כתוביות</option>
          {available.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
      </div>
      {videoUrl && (
        <video key={videoKey} src={videoUrl} controls crossOrigin="anonymous">
          {trackUrl && (
            <track kind="subtitles" label={selected} srcLang="he" src={trackUrl} default />
          )}
        </video>
      )}
    </div>
  );
}
