// Respeak single-page frontend (docs/flow.md B3/B5/B6).
// No framework, no build step, no external resources. Loaded as a module, so the DOM is ready.

const POLL_MS = 3000;

// The pipeline's stages, in order (docs/flow.md B4.1…B4.8 plus the finish step).
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

const STATE_LABELS = {
  queued: "Queued",
  running: "Running",
  done: "Done",
  failed: "Failed",
};

const el = (id) => document.getElementById(id);

const dom = {
  formPanel: el("form-panel"),
  form: el("job-form"),
  sourceType: el("source-type"),
  tabUrl: el("tab-url"),
  tabUpload: el("tab-upload"),
  panelUrl: el("source-url"),
  panelUpload: el("source-upload"),
  url: el("url"),
  file: el("file"),
  toLang: el("to-lang"),
  fromLang: el("from-lang"),
  backendField: el("backend-field"),
  backend: el("backend"),
  voiceField: el("voice-field"),
  voice: el("voice"),
  preview: el("voice-preview"),
  sample: el("voice-sample"),
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
  jobDetail: el("job-detail"),
  runCard: el("run-card"),
  stepList: el("step-list"),
  jobWarnings: el("job-warnings"),
  warningList: el("job-warning-list"),
  jobError: el("job-error"),
  output: el("output"),
  video: el("output-video"),
  download: el("download-link"),
  donePair: el("done-pair"),
  doneFile: el("done-file"),
};

const state = {
  codeToName: Object.assign({}, FALLBACK_NAMES),
  backends: [], // installed backends only: {name, languages, cloning, voices}
  jobId: "",
  lastJob: null, // so the job view can be redrawn once the backend list arrives
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
  if (data.allow_uploads !== true && dom.sourceType.value === "upload") selectSource("url");

  const all = Array.isArray(data.backends) ? data.backends : [];
  state.backends = all
    .filter((backend) => backend && backend.installed)
    .map((backend) => ({
      name: String(backend.name),
      languages: Array.isArray(backend.languages) ? backend.languages.map(String) : [],
      cloning: backend.cloning === true,
      // {lang: [{id, name}]} — absent on a server that predates the voice picker, which simply
      // means "no voice choice here" and leaves the select hidden.
      voices: backend.voices && typeof backend.voices === "object" ? backend.voices : {},
    }));

  if (!state.backends.length) {
    const reason = all.map((b) => b && b.reason).filter(Boolean)[0];
    showFormError(reason ? `No speech backend is installed: ${reason}` : "No speech backend is installed.");
    dom.submit.disabled = true;
    return;
  }

  fillBackends(typeof data.default === "string" ? data.default : "");
  fillTargetLanguages();
  // A job page loaded straight from /jobs/<id> may already have drawn itself without language or
  // voice names to hand; now that they are here, draw it again.
  if (state.lastJob) renderJob(state.lastJob);
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
  // The selector only earns screen space when there is a real choice (docs/flow.md B6).
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
    fillVoices();
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
  fillVoices();
}

/** The voices the chosen backend offers for the chosen language (docs/flow.md B6). */
function voicesFor(backend, lang) {
  if (!backend || !lang) return [];
  const list = backend.voices ? backend.voices[lang] : null;
  if (!Array.isArray(list)) return [];
  return list
    .filter((voice) => voice && voice.id)
    .map((voice) => ({ id: String(voice.id), name: voice.name ? String(voice.name) : String(voice.id) }));
}

function fillVoices() {
  const entries = voicesFor(selectedBackend(), dom.toLang.value);
  const previous = dom.voice.value;
  stopPreview(); // the list under the player is about to change

  dom.voice.replaceChildren();
  if (!entries.length) {
    // A disabled control stays out of the FormData, so no `voice` is posted when there is no choice.
    dom.voice.disabled = true;
    dom.preview.disabled = true;
    show(dom.voiceField, false);
    return;
  }
  for (const entry of entries) {
    const option = document.createElement("option");
    option.value = entry.id;
    option.textContent = entry.name;
    dom.voice.appendChild(option);
  }
  dom.voice.value = entries.some((e) => e.id === previous) ? previous : entries[0].id;
  dom.voice.disabled = false;
  dom.preview.disabled = false;
  show(dom.voiceField, true);
}

function voiceName(backendName, lang, id) {
  const backend = state.backends.find((b) => b.name === backendName);
  const hit = voicesFor(backend, lang).find((voice) => voice.id === String(id));
  return hit ? hit.name : String(id);
}

// ---------------------------------------------------------------- voice previews

const preview = {
  // "backend/lang/voice" -> object URL. Kept for the life of the page, so listening to the same
  // voice twice costs one request; the server caches the WAV on disk for every later visit.
  urls: new Map(),
  key: "", // the voice that is loading or playing right now, "" when nothing is
  run: 0, // bumped by every stop, so a fetch the user has moved on from can never start playing
};

const PREVIEW_LABELS = {
  idle: "Preview this voice",
  loading: "Loading the voice preview",
  playing: "Stop the preview",
};

/** The voice the button would play, or null when there is no choice to play. */
function pickedVoice() {
  const backend = selectedBackend();
  if (!backend || dom.voice.disabled || !dom.toLang.value || !dom.voice.value) return null;
  const [name, lang, voice] = [backend.name, dom.toLang.value, dom.voice.value];
  return { name, lang, voice, key: `${name}/${lang}/${voice}` };
}

function setPreviewState(mode) {
  dom.preview.dataset.state = mode;
  dom.preview.setAttribute("aria-busy", mode === "loading" ? "true" : "false");
  dom.preview.setAttribute("aria-label", PREVIEW_LABELS[mode] || PREVIEW_LABELS.idle);
}

/** Silence the player and forget whatever is in flight. Safe to call at any time. */
function stopPreview() {
  preview.run += 1;
  preview.key = "";
  dom.sample.pause();
  if (dom.sample.currentTime) dom.sample.currentTime = 0;
  setPreviewState("idle");
}

/** The object URL for one voice, fetching and caching it the first time. */
async function previewUrl(pick) {
  const cached = preview.urls.get(pick.key);
  if (cached) return cached;
  const path = [pick.name, pick.lang, pick.voice].map(encodeURIComponent).join("/");
  const response = await fetch(`/api/voices/${path}`);
  if (!response.ok) throw new Error(errorMessage(await readJson(response), response));
  const objectUrl = URL.createObjectURL(await response.blob());
  preview.urls.set(pick.key, objectUrl);
  return objectUrl;
}

async function onPreviewClick() {
  const pick = pickedVoice();
  if (!pick) return;
  // Clicking the button while it is busy with this same voice means "stop", loading or playing.
  const busy = dom.preview.dataset.state !== "idle" && preview.key === pick.key;
  stopPreview();
  if (busy) return;

  const run = preview.run; // this attempt's ticket; any stop after here invalidates it
  preview.key = pick.key;
  setPreviewState("loading");
  try {
    const objectUrl = await previewUrl(pick);
    if (run !== preview.run) return;
    dom.sample.src = objectUrl;
    await dom.sample.play(); // resolves once it is actually audible, which is what "busy" meant
    if (run !== preview.run) {
      dom.sample.pause();
      return;
    }
    setPreviewState("playing");
  } catch (err) {
    if (run !== preview.run) return; // the user moved on; their stop already tidied up
    stopPreview();
    showFormError(`Could not play a sample of this voice (${err.message}).`);
  }
}

// ---------------------------------------------------------------- source tabs

function selectSource(kind) {
  const upload = kind === "upload";
  dom.sourceType.value = upload ? "upload" : "url";

  dom.tabUrl.classList.toggle("is-active", !upload);
  dom.tabUpload.classList.toggle("is-active", upload);
  dom.tabUrl.setAttribute("aria-selected", String(!upload));
  dom.tabUpload.setAttribute("aria-selected", String(upload));
  // Roving tabindex: one stop for the whole tablist, arrow keys move between the tabs.
  dom.tabUrl.tabIndex = upload ? -1 : 0;
  dom.tabUpload.tabIndex = upload ? 0 : -1;

  show(dom.panelUrl, !upload);
  show(dom.panelUpload, upload);
  // Disabled controls stay out of the FormData, so only the active source is ever sent.
  dom.url.disabled = upload;
  dom.file.disabled = !upload;
}

function onTabKey(event) {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
  if (dom.tabUpload.hidden) return; // only one tab to stand on
  event.preventDefault();
  const next = dom.sourceType.value === "upload" ? "url" : "upload";
  selectSource(next);
  (next === "upload" ? dom.tabUpload : dom.tabUrl).focus();
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
  if (!body.get("from_lang")) body.delete("from_lang"); // "" means auto-detect
  if (!body.get("voice")) body.delete("voice"); // "" means the backend's own default
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
    item.className = "rail__step";

    const node = document.createElement("span");
    node.className = "rail__node";
    node.setAttribute("aria-hidden", "true");

    const text = document.createElement("span");
    text.className = "rail__label";
    text.textContent = label;

    // The rail is colour and shape; this says the same thing out loud.
    const spoken = document.createElement("span");
    spoken.className = "rail__sr";

    item.append(node, text, spoken);
    dom.stepList.appendChild(item);
    state.stepNodes.set(key, { item, spoken });
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
    node.spoken.textContent = failed
      ? " — stopped here"
      : done
        ? " — done"
        : isCurrent
          ? " — in progress"
          : " — waiting";
  });
}

function fact(key, value, modifier) {
  const wrap = document.createElement("div");
  wrap.className = "fact";
  const term = document.createElement("dt");
  term.className = "fact__k";
  term.textContent = key;
  const description = document.createElement("dd");
  description.className = modifier ? `fact__v ${modifier}` : "fact__v";
  description.textContent = value;
  wrap.append(term, description);
  return wrap;
}

/** The language pair, labelled honestly: the source is not known until transcribe has run. */
function languagePair(job) {
  const options = job.options || {};
  const from = job.detected_language || options.from_lang;
  const into = languageName(options.to_lang) || "the target language";
  if (from) return { label: "Languages", value: `${languageName(from)} → ${into}` };
  if (job.state === "queued" || job.state === "running") {
    return { label: "Languages", value: `Listening → ${into}` };
  }
  return { label: "Dubbing into", value: into };
}

function renderMeta(job) {
  const options = job.options || {};
  const pair = languagePair(job);
  const facts = [fact(pair.label, pair.value)];

  // A backend's voice name may already carry its own parenthesis ("Dora (female)"), so the engine
  // gets a column of its own rather than a second pair of brackets around it.
  if (options.voice) facts.push(fact("Voice", voiceName(options.backend, options.to_lang, options.voice)));
  if (options.backend) facts.push(fact("Engine", titleCase(String(options.backend))));
  if (typeof options.burn_subtitles === "boolean") {
    facts.push(fact("Subtitles", options.burn_subtitles ? "Burned in" : "Separate track"));
  }
  if (job.id) facts.push(fact("Job", String(job.id), "fact__v--id"));
  dom.jobMeta.replaceChildren(...facts);
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
  else if (job.state === "done") setText(dom.statusLine, "Finished. Your video is below.");
  else if (job.state === "queued") setText(dom.statusLine, "Queued — waiting for a free worker.");
  else setText(dom.statusLine, `${stepLabel || "Working"} — ${percent}%`);

  // `detail` is the stage's own running commentary: "segment 4 of 12" (docs/flow.md B5).
  const detail = typeof job.detail === "string" ? job.detail.trim() : "";
  setText(dom.jobDetail, detail);
  show(dom.jobDetail, detail !== "" && job.state !== "done");
}

/** Warnings are things the user should know about a job that still worked (e.g. a cut-off tail). */
function renderWarnings(job) {
  const messages = Array.isArray(job.warnings)
    ? job.warnings.filter((item) => typeof item === "string" && item.trim())
    : [];
  dom.warningList.replaceChildren(
    ...messages.map((message) => {
      const item = document.createElement("li");
      item.textContent = message.trim();
      return item;
    })
  );
  dom.warningList.classList.toggle("is-single", messages.length === 1);
  show(dom.jobWarnings, messages.length > 0);
}

/** Mirrors `safe_name` in respeak/api.py so the page shows the name the file will be saved under. */
function safeName(raw) {
  if (!raw) return "";
  const cleaned = String(raw)
    .replace(/[^\p{L}\p{N}_.\- ]+/gu, "_")
    .replace(/^[ ._]+/, "")
    .replace(/[ ._]+$/, "");
  return cleaned.slice(0, 120);
}

function outputFileName(job) {
  const source = job.source || {};
  const title = safeName(job.title) || safeName(source.filename) || String(job.id || state.jobId);
  const target = (job.options || {}).to_lang || "dub";
  return `${title}-${target}.mp4`;
}

function renderDone(job) {
  const url = job.download_url || `/api/jobs/${encodeURIComponent(job.id || state.jobId)}/download`;
  if (dom.video.getAttribute("src") !== url) dom.video.setAttribute("src", url);
  const name = outputFileName(job);
  dom.download.setAttribute("href", url);
  dom.download.setAttribute("download", name);
  setText(dom.donePair, languagePair(job).value);
  setText(dom.doneFile, name);
  show(dom.output, true);
}

function renderJob(job) {
  state.lastJob = job;
  const jobState = job.state || "queued";

  setText(dom.jobTitle, job.title || "Dubbing your video");
  setText(dom.jobState, STATE_LABELS[jobState] || jobState);
  dom.jobState.dataset.state = jobState;
  document.title = job.title ? `${job.title} — Respeak` : "Respeak";
  renderMeta(job);
  renderProgress(job);
  renderSteps(job);
  renderWarnings(job);
  // Once it is finished the machinery has nothing left to say, so the page gets out of the way and
  // the video comes straight after the facts. A failed job keeps the rail: it shows where it stopped.
  show(dom.runCard, jobState !== "done");

  if (jobState === "failed") {
    stopPolling();
    setText(dom.jobError, job.error || "The job failed without a message.");
    show(dom.jobError, true);
    return;
  }
  show(dom.jobError, false);

  if (jobState === "done") {
    stopPolling();
    renderDone(job);
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
dom.tabUrl.addEventListener("click", () => selectSource("url"));
dom.tabUpload.addEventListener("click", () => selectSource("upload"));
dom.tabUrl.addEventListener("keydown", onTabKey);
dom.tabUpload.addEventListener("keydown", onTabKey);
dom.backend.addEventListener("change", fillTargetLanguages);
dom.toLang.addEventListener("change", fillVoices);
dom.voice.addEventListener("change", stopPreview); // a different voice is not the one playing
dom.preview.addEventListener("click", onPreviewClick);
dom.sample.addEventListener("ended", stopPreview);
dom.form.addEventListener("submit", onSubmit);
window.addEventListener("popstate", () => window.location.reload());

selectSource("url");

const initialJobId = document.body.dataset.jobId || "";
if (initialJobId) {
  // The page was reloaded on /jobs/<id> or the link was shared: go straight to the job view.
  showJobView();
  startPolling(initialJobId);
}

loadBackends();
