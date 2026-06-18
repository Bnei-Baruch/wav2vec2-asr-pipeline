export const MODELS = ["whisper-large-v3-he-lr-1e5", "ivrit-ai", "whisper-v3-audiofolder"];

export function fmtAudio(s) {
  if (s == null) return "--:--";
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1).padStart(4, "0");
  return `${m}:${sec}`;
}

export function fmtElapsed(s) {
  const m = String(Math.floor(s / 60)).padStart(2, "0");
  const sec = String(s % 60).padStart(2, "0");
  return `${m}:${sec}`;
}

export const STATUS = { wait: "wait", run: "run", done: "done", error: "error" };
export const BADGE = {
  wait: ["badge-wait", "ожидание"],
  run: ["badge-run", "обработка"],
  done: ["badge-done", "готово"],
  error: ["badge-error", "ошибка"],
};

export function initCols() {
  return Object.fromEntries(MODELS.map(m => [m, { status: STATUS.wait, data: null, error: null }]));
}

// "HH:MM:SS.mmm" timestamp for WebVTT cues
function vttTime(s) {
  const h = String(Math.floor(s / 3600)).padStart(2, "0");
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const sec = (s % 60).toFixed(3).padStart(6, "0");
  return `${h}:${m}:${sec}`;
}

// Build a WebVTT document from the API's chunks ([{start, end, text}]).
export function chunksToVtt(chunks) {
  const cues = (chunks || [])
    .filter(c => c.start != null && c.end != null)
    .map(c => `${vttTime(c.start)} --> ${vttTime(c.end)}\n${c.text}`);
  return "WEBVTT\n\n" + cues.join("\n\n");
}
