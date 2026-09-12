// Respeak single-page frontend (flow.md B3/B5/B6; decisions D-29…D-34).
// No framework, no build step, no external resources. Loaded as a module, so the DOM is ready.

const POLL_MS = 3000;

// The pipeline's stages, in order (flow.md B4.1…B4.8 plus the finish step).
const STEPS = [
  ["probe", "Inspect the source"],
  ["fetch", "Fetch the video"],
  ["transcribe", "Transcribe the speech"],
  ["translate", "Translate the text"],
  ["speak", "Speak the translation"],
  ["fit", "Fit the audio to the timing"],
  ["subtitles", "Build the subtitles"],
  ["mux", "Mux the final video"],
  ["finish", "Finish up"],
];

// Names for codes that /api/backends does not list in source_languages.
const FALLBACK_NAMES = {
  bn: "Bengali",
  bg: "Bulgarian",
  et: "Estonian",
  fil: "Filipino",
  lt: "Lithuanian",
  lv: "Latvian",
  ms: "Malay",
  no: "Norwegian",
  ro: "Romanian",
  sl: "Slovenian",
  sq: "Albanian",
  sr: "Serbian",
  sw: "Swahili",
  ta: "Tamil",
  th: "Thai",
  ur: "Urdu",
  vi: "Vietnamese",
};

const el = (id) => document.getElementById(id);

const dom = {
  formPanel: el("form-panel"),
  form: el("job-form"),
  sourceType: el("source-type"),
  tabYoutube: el("tab-youtube"),
  tabUpload: el("tab-upload"),
  panelYoutube: el("source-youtube"),
  panelUpload: el("source-upload"),
  url: el("url"),
  file: el("file"),
  toLang: el("to-lang"),
  fromLang: el("from-lang"),
  backendField: el("backend-field"),
  backend: el("backend"),
  burn: el("burn-subtitles"),
  submit: el("submit-button"),
  formError: el("form-error"),
  resultPanel: el("result-panel"),
  jobTitle: el("job-title"),
  jobState: el("job-state"),
  jobMeta: el("job-meta"),
  progress: el("progress"),
  progressFill: el("progress-fill"),
  statusLine: el("status-line"),
  stepList: el("step-list"),
  jobError: el("job-error"),
  output: el("output"),
  video: el("output-video"),
  download: el("download-link"),
};

const state = {
  codeToName: Object.assign({}, FALLBACK_NAMES),
  backends: [], // installed backends only: {name, languages: [...], cloning}
  jobId: "",
  timer: null,
  polling: false,
  submitting: false,
  stepNodes: new Map(),
};

// ---------------------------------------------------------------- helpers

function languageName(code) {
  if (!code) return "";
  return state.codeToName[code] || code.toUpperCase();
}

function titleCase(text) {
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

function show(node, visible) {
  node.hidden = !visible;
}

function setText(node, text) {
  node.textContent = text; // never innerHTML: server text is never trusted markup
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

/** Pull a human message out of an error response body. */
function errorMessage(payload, response) {
  if (payload && typeof payload.error === "string" && payload.error) return payload.error;
  const detail = payload ? payload.detail : null;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail) && detail.length) {
    const parts = detail.map((item) => (item && item.msg ? String(item.msg) : String(item)));
    return parts.join("; ");
  }
  return `The server rejected the request (HTTP ${response.status}).`;
}

function showFormError(message) {
  setText(dom.formError, message);
  show(dom.formError, true);
}

function clearFormError() {
  setText(dom.formError, "");
  show(dom.formError, false);
}

function setSubmitting(busy) {
  state.submitting = busy;
  dom.submit.disabled = busy;
  setText(dom.submit, busy ? "Starting…" : "Start dubbing");
}

// ---------------------------------------------------------------- backends

async function loadBackends() {
  let data;
  try {
    const response = await fetch("/api/backends", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    data = await response.json();
  } catch (err) {
    showFormError(`Could not load the available voices and languages (${err.message}).`);
    dom.submit.disabled = true;
    return;
  }

  const sources = Array.isArray(data.source_languages) ? data.source_languages : [];
  for (const entry of sources) {
    if (entry && entry.code && entry.name) state.codeToName[entry.code] = entry.name;
  }
  fillSourceLanguages(sources);

  show(dom.tabUpload, data.allow_uploads === true);
  if (data.allow_uploads !== true && dom.sourceType.value === "upload") selectSource("youtube");

  const all = Array.isArray(data.backends) ? data.backends : [];
  state.backends = all
    .filter((backend) => backend && backend.installed)
    .map((backend) => ({
      name: String(backend.name),
      languages: Array.isArray(backend.languages) ? backend.languages.map(String) : [],
      cloning: backend.cloning === true,
    }));

  if (!state.backends.length) {
    const reason = all.map((b) => b && b.reason).filter(Boolean)[0];
    showFormError(reason ? `No speech backend is installed: ${reason}` : "No speech backend is installed.");
    dom.submit.disabled = true;
    return;
  }

  fillBackends(typeof data.default === "string" ? data.default : "");
  fillTargetLanguages();
}

function fillSourceLanguages(sources) {
  const options = sources
    .filter((entry) => entry && entry.code && entry.name)
    .slice()
    .sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of options) {
    const option = document.createElement("option");
    option.value = entry.code;
    option.textContent = entry.name;
    dom.fromLang.appendChild(option);
  }
}

function fillBackends(preferred) {
  dom.backend.replaceChildren();
  for (const backend of state.backends) {
    const option = document.createElement("option");
    option.value = backend.name;
    option.textContent = titleCase(backend.name);
    dom.backend.appendChild(option);
  }
  const wanted = state.backends.some((b) => b.name === preferred) ? preferred : state.backends[0].name;
  dom.backend.value = wanted;
  // The selector only earns screen space when there is a real choice (flow.md B6).
  show(dom.backendField, state.backends.length > 1);
}

function selectedBackend() {
  return state.backends.find((b) => b.name === dom.backend.value) || state.backends[0] || null;
}

function fillTargetLanguages() {
  const backend = selectedBackend();
  const codes = backend ? backend.languages.slice() : [];
  const previous = dom.toLang.value;
  const entries = codes
    .map((code) => ({ code, name: languageName(code) }))
    .sort((a, b) => a.name.localeCompare(b.name));

  dom.toLang.replaceChildren();
  if (!entries.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No languages available";
    dom.toLang.appendChild(option);
    dom.toLang.disabled = true;
    return;
  }
  dom.toLang.disabled = false;
  for (const entry of entries) {
    const option = document.createElement("option");
    option.value = entry.code;
    option.textContent = entry.name;
    dom.toLang.appendChild(option);
  }
  const keep = entries.some((e) => e.code === previous) ? previous : "";
  const fallback = entries.some((e) => e.code === "es") ? "es" : entries[0].code;
  dom.toLang.value = keep || fallback;
}

// ---------------------------------------------------------------- source tabs

function selectSource(kind) {
  const upload = kind === "upload";
  dom.sourceType.value = upload ? "upload" : "youtube";

  dom.tabYoutube.classList.toggle("is-active", !upload);
  dom.tabUpload.classList.toggle("is-active", upload);
  dom.tabYoutube.setAttribute("aria-selected", String(!upload));
  dom.tabUpload.setAttribute("aria-selected", String(upload));

  show(dom.panelYoutube, !upload);
  show(dom.panelUpload, upload);
  // Disabled controls stay out of the FormData, so only the active source is ever sent.
  dom.url.disabled = upload;
  dom.file.disabled = !upload;
}

// ---------------------------------------------------------------- submit

async function onSubmit(event) {
  event.preventDefault();
  if (state.submitting) return;
  clearFormError();

  const upload = dom.sourceType.value === "upload";
  if (!upload && !dom.url.value.trim()) {
    showFormError("Paste the URL of the video you want dubbed.");
    dom.url.focus();
    return;
  }
  if (upload && !(dom.file.files && dom.file.files.length)) {
    showFormError("Choose a video file to upload.");
    dom.file.focus();
    return;
  }
  if (!dom.toLang.value) {
    showFormError("Pick the language to dub into.");
    dom.toLang.focus();
    return;
  }

  const body = new FormData(dom.form);
  // An unchecked checkbox is simply absent from a FormData, which would read as "use the default".
  body.set("burn_subtitles", dom.burn.checked ? "true" : "false");
  if (!body.get("from_lang")) body.delete("from_lang"); // "" means auto-detect (D-32)
  if (!upload) body.set("url", dom.url.value.trim());

  setSubmitting(true);
  let response;
  try {
    response = await fetch("/jobs", { method: "POST", body, headers: { Accept: "application/json" } });
  } catch (err) {
    setSubmitting(false);
    showFormError(`Could not reach the server (${err.message}).`);
    return;
  }

  const payload = await readJson(response);
  if (!response.ok || !payload || !payload.id) {
    setSubmitting(false);
    showFormError(response.ok ? "The server did not return a job id." : errorMessage(payload, response));
    return;
  }

  const jobUrl = typeof payload.url === "string" && payload.url ? payload.url : `/jobs/${payload.id}`;
  history.pushState({ jobId: payload.id }, "", jobUrl);
  setSubmitting(false);
  showJobView();
  startPolling(String(payload.id));
}

// ---------------------------------------------------------------- job view

function buildSteps() {
  dom.stepList.replaceChildren();
  state.stepNodes.clear();
  for (const [key, label] of STEPS) {
    const item = document.createElement("li");
    item.className = "step";
    const marker = document.createElement("span");
    marker.className = "step__marker";
    marker.textContent = "·";
    marker.setAttribute("aria-hidden", "true");
    const text = document.createElement("span");
    text.textContent = label;
    item.append(marker, text);
    dom.stepList.appendChild(item);
    state.stepNodes.set(key, { item, marker });
  }
}

function renderSteps(job) {
  const current = STEPS.findIndex(([key]) => key === job.step);
  const finished = job.state === "done";
  STEPS.forEach(([key], index) => {
    const node = state.stepNodes.get(key);
    const done = finished || (current >= 0 && index < current);
    const isCurrent = !finished && current === index;
    const failed = job.state === "failed" && isCurrent;
    node.item.classList.toggle("is-done", done && !failed);
    node.item.classList.toggle("is-current", isCurrent && !failed);
    node.item.classList.toggle("is-failed", failed);
    node.marker.textContent = failed ? "×" : done ? "✓" : isCurrent ? "▸" : "·";
  });
}

function renderMeta(job) {
  const bits = [];
  if (job.id) bits.push(`Job ${job.id}`);
  if (job.detected_language) bits.push(`Detected language: ${languageName(job.detected_language)}`);
  const options = job.options || {};
  if (options.to_lang) bits.push(`Dubbing into ${languageName(options.to_lang)}`);
  if (options.backend) bits.push(`Voice: ${titleCase(options.backend)}`);
  setText(dom.jobMeta, bits.join(" · "));
}

function renderProgress(job) {
  const raw = typeof job.progress === "number" && isFinite(job.progress) ? job.progress : 0;
  const percent = Math.round(Math.min(1, Math.max(0, raw)) * 100);
  dom.progressFill.style.width = `${percent}%`;
  dom.progress.setAttribute("aria-valuenow", String(percent));
  dom.progress.classList.toggle("is-failed", job.state === "failed");

  const stepLabel = (STEPS.find(([key]) => key === job.step) || [null, job.step])[1];
  if (job.state === "failed")
    setText(dom.statusLine, stepLabel ? `Stopped during "${stepLabel}".` : "The job stopped before it started.");
  else if (job.state === "done") setText(dom.statusLine, "Finished. The dubbed video is ready below.");
  else if (job.state === "queued") setText(dom.statusLine, "Queued — waiting for a free worker.");
  else setText(dom.statusLine, `${stepLabel || "Working"}… ${percent}%`);
}

function renderJob(job) {
  setText(dom.jobTitle, job.title || "Dubbing your video");
  setText(dom.jobState, job.state || "queued");
  dom.jobState.dataset.state = job.state || "queued";
  renderMeta(job);
  renderProgress(job);
  renderSteps(job);

  if (job.state === "failed") {
    stopPolling();
    setText(dom.jobError, job.error || "The job failed without a message.");
    show(dom.jobError, true);
    return;
  }
  show(dom.jobError, false);

  if (job.state === "done") {
    stopPolling();
    const url = job.download_url || `/api/jobs/${encodeURIComponent(job.id || state.jobId)}/download`;
    if (dom.video.getAttribute("src") !== url) dom.video.setAttribute("src", url);
    dom.download.setAttribute("href", url);
    show(dom.output, true);
  }
}

function showJobError(message) {
  setText(dom.jobError, message);
  show(dom.jobError, true);
}

function showJobView() {
  show(dom.formPanel, false);
  show(dom.resultPanel, true);
}

// ---------------------------------------------------------------- polling

function stopPolling() {
  if (state.timer !== null) {
    clearInterval(state.timer);
    state.timer = null;
  }
}

function startPolling(jobId) {
  state.jobId = jobId;
  stopPolling();
  poll();
  state.timer = setInterval(poll, POLL_MS);
}

async function poll() {
  if (state.polling || !state.jobId) return;
  state.polling = true;
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(state.jobId)}`, {
      headers: { Accept: "application/json" },
    });
    if (response.status === 404) {
      stopPolling();
      showJobError("That job no longer exists. Finished jobs are deleted after a while.");
      return;
    }
    if (!response.ok) return; // transient server hiccup: keep polling
    const job = await readJson(response);
    if (job) renderJob(job);
  } catch {
    // network blip; the next tick tries again
  } finally {
    state.polling = false;
  }
}

// ---------------------------------------------------------------- wiring

buildSteps();
dom.tabYoutube.addEventListener("click", () => selectSource("youtube"));
dom.tabUpload.addEventListener("click", () => selectSource("upload"));
dom.backend.addEventListener("change", fillTargetLanguages);
dom.form.addEventListener("submit", onSubmit);
window.addEventListener("popstate", () => window.location.reload());

selectSource("youtube");

const initialJobId = document.body.dataset.jobId || "";
if (initialJobId) {
  // The page was reloaded on /jobs/<id> or the link was shared: go straight to the job view.
  showJobView();
  startPolling(initialJobId);
}

loadBackends();
