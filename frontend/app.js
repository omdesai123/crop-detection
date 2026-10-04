/* Fieldscan frontend - talks to the FastAPI backend at /api */
(() => {
  "use strict";

  const API = ""; // same origin. If you host the frontend elsewhere, set e.g. "http://localhost:8000"
  const $ = (id) => document.getElementById(id);

  const el = {
    status: $("status"), statusText: $("statusText"), maxMb: $("maxMb"),
    alert: $("alert"), alertText: $("alertText"), alertClose: $("alertClose"),
    finder: $("finder"), finderEmpty: $("finderEmpty"), preview: $("preview"), scan: $("scan"),
    fileInput: $("fileInput"), chooseBtn: $("chooseBtn"), resetBtn: $("resetBtn"),
    verdict: document.querySelector(".verdict"),
    idle: $("verdictIdle"), busy: $("verdictBusy"), busyText: $("busyText"), done: $("verdictDone"),
    resFile: $("resFile"), resLabel: $("resLabel"), resConf: $("resConf"),
    strip: $("strip"), legend: $("legend"), resNote: $("resNote"), resMeta: $("resMeta"),
    historySection: $("historySection"), historyList: $("historyList"), clearHistory: $("clearHistory"),
    modelInfo: $("modelInfo"),
  };

  const state = { ready: false, maxMb: 10, classes: [], busy: false, history: [], currentId: null, nextId: 1 };

  /* Colors: crop and weed have fixed colors, any other class gets one from the list */
  const FIXED = { crop: "#2b7a4b", weed: "#b8741a" };
  const EXTRA = ["#3a6ea5", "#7a4f9a", "#a3302a", "#5d6e64"];
  const colorFor = (name) => {
    const key = name.toLowerCase();
    if (FIXED[key]) return FIXED[key];
    const i = Math.max(0, state.classes.indexOf(name)) % EXTRA.length;
    return EXTRA[i];
  };
  const pct = (x) => (x * 100).toFixed(1) + "%";

  /* ---------- small UI helpers ---------- */
  function showAlert(msg) { el.alertText.textContent = msg; el.alert.hidden = false; }
  function hideAlert() { el.alert.hidden = true; }

  function setView(view) {
    el.idle.hidden = view !== "idle";
    el.busy.hidden = view !== "busy";
    el.done.hidden = view !== "done";
  }

  function setStatus(kind, text) {
    el.status.classList.remove("status--ok", "status--bad");
    if (kind) el.status.classList.add("status--" + kind);
    el.statusText.textContent = text;
  }

  /* ---------- server health ---------- */
  async function checkHealth() {
    try {
      const res = await fetch(API + "/api/health");
      const h = await res.json();
      state.maxMb = h.max_upload_mb || 10;
      el.maxMb.textContent = state.maxMb;
      if (h.model_loaded) {
        state.ready = true;
        state.classes = h.classes;
        setStatus("ok", "Model ready");
        el.modelInfo.textContent =
          `Model: custom CNN, ${h.img_size} x ${h.img_size} px input. Classes: ${h.classes.join(", ")}. ` +
          `Results below ${Math.round(h.low_confidence_threshold * 100)}% confidence are flagged.`;
      } else {
        state.ready = false;
        setStatus("bad", "Model not loaded");
        el.modelInfo.textContent = h.detail || "The server is running but no model is loaded.";
        showAlert(h.detail || "The server is running but no model is loaded.");
      }
    } catch {
      state.ready = false;
      setStatus("bad", "Server offline");
      el.modelInfo.textContent = "Cannot reach the server. Start it with: uvicorn backend.main:app";
      showAlert("Cannot reach the server. Start it with: uvicorn backend.main:app --reload");
    }
  }

  /* ---------- file handling ---------- */
  function validate(file) {
    if (!["image/jpeg", "image/png"].includes(file.type)) return `${file.name} is not a JPG or PNG image.`;
    if (file.size > state.maxMb * 1024 * 1024) return `${file.name} is larger than ${state.maxMb} MB. Choose a smaller photo.`;
    return null;
  }

  async function predict(file) {
    const body = new FormData();
    body.append("file", file);
    const res = await fetch(API + "/api/predict", { method: "POST", body });
    let data = {};
    try { data = await res.json(); } catch { /* non-JSON error */ }
    if (!res.ok) throw new Error(data.detail || `The server returned an error (${res.status}).`);
    return data;
  }

  async function handleFiles(fileList) {
    if (state.busy) return;
    hideAlert();
    if (!state.ready) { await checkHealth(); if (!state.ready) return; }

    const files = Array.from(fileList);
    if (!files.length) return;

    state.busy = true;
    el.chooseBtn.disabled = true;
    let firstShownId = null;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const problem = validate(file);
      if (problem) { showAlert(problem); continue; }

      const url = URL.createObjectURL(file);
      if (firstShownId === null) {            // only the first photo takes over the main view
        showPreview(url);
        setView("busy");
        el.busyText.textContent = files.length > 1 ? `Photo 1 of ${files.length}` : "This takes a moment.";
        el.scan.hidden = false;
      } else {
        el.busyText.textContent = `Photo ${i + 1} of ${files.length}`;
      }

      try {
        const result = await predict(file);
        const item = { id: state.nextId++, name: file.name, url, result };
        state.history.unshift(item);
        if (firstShownId === null) { firstShownId = item.id; showResult(item); }
      } catch (err) {
        showAlert(`${file.name}: ${err.message}`);
        if (firstShownId === null) { clearPreview(); setView("idle"); }
      }
    }

    el.scan.hidden = true;
    state.busy = false;
    el.chooseBtn.disabled = false;
    renderHistory();
    el.fileInput.value = "";
  }

  /* ---------- rendering ---------- */
  function showPreview(url) {
    el.preview.src = url;
    el.preview.hidden = false;
    el.finderEmpty.hidden = true;
    el.finder.classList.add("has-image");
    el.finder.setAttribute("tabindex", "-1");
    el.finder.removeAttribute("role");
    el.resetBtn.hidden = false;
  }

  function clearPreview() {
    el.preview.hidden = true;
    el.preview.removeAttribute("src");
    el.finderEmpty.hidden = false;
    el.finder.classList.remove("has-image");
    el.finder.setAttribute("tabindex", "0");
    el.finder.setAttribute("role", "button");
    el.resetBtn.hidden = true;
    el.verdict.classList.remove("is-crop", "is-weed");
    state.currentId = null;
    renderHistory();
  }

  function showResult(item) {
    const r = item.result;
    state.currentId = item.id;
    showPreview(item.url);
    el.scan.hidden = true;

    el.resFile.textContent = item.name;
    el.resLabel.textContent = r.label;
    el.resConf.textContent = pct(r.confidence);

    el.verdict.classList.remove("is-crop", "is-weed");
    const key = r.label.toLowerCase();
    if (key === "crop" || key === "weed") el.verdict.classList.add("is-" + key);

    // split strip + legend, biggest class first
    const entries = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]);
    el.strip.setAttribute("aria-label", entries.map(([n, p]) => `${n} ${pct(p)}`).join(", "));
    el.strip.replaceChildren();
    el.legend.replaceChildren();
    for (const [name, p] of entries) {
      const seg = document.createElement("div");
      seg.className = "strip__seg";
      seg.style.background = colorFor(name);
      seg.style.flexBasis = "0%";
      el.strip.appendChild(seg);
      requestAnimationFrame(() => requestAnimationFrame(() => { seg.style.flexBasis = (p * 100).toFixed(2) + "%"; }));

      const li = document.createElement("li");
      li.innerHTML = '<span class="legend__swatch"></span><span class="legend__name"></span><span class="legend__pct"></span>';
      li.querySelector(".legend__swatch").style.background = colorFor(name);
      li.querySelector(".legend__name").textContent = name;
      li.querySelector(".legend__pct").textContent = pct(p);
      el.legend.appendChild(li);
    }

    el.resNote.className = "note";
    if (r.low_confidence) {
      el.resNote.classList.add("note--warn");
      el.resNote.textContent = "The model is not sure about this one. Retake the photo closer to the plant, in daylight, with one plant in the frame.";
    } else {
      el.resNote.textContent = `The model is confident this is a ${r.label}.`;
    }
    el.resMeta.textContent = `Analyzed in ${r.inference_ms} ms`;

    setView("done");
    renderHistory();
  }

  function renderHistory() {
    el.historySection.hidden = state.history.length === 0;
    el.historyList.replaceChildren();
    for (const item of state.history) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "hitem";
      if (item.id === state.currentId) btn.setAttribute("aria-current", "true");
      btn.setAttribute("aria-label", `${item.name}: ${item.result.label}, ${pct(item.result.confidence)}. Show this result.`);

      const img = document.createElement("img");
      img.src = item.url; img.alt = "";
      const body = document.createElement("div");
      body.className = "hitem__body";
      body.innerHTML = '<div class="hitem__label"><span class="hitem__dot"></span><span class="t"></span></div><div class="hitem__sub"></div>';
      body.querySelector(".hitem__dot").style.background = colorFor(item.result.label);
      body.querySelector(".t").textContent = item.result.label;
      body.querySelector(".hitem__sub").textContent = pct(item.result.confidence) + " confident";

      btn.append(img, body);
      btn.addEventListener("click", () => { if (!state.busy) showResult(item); });
      li.appendChild(btn);
      el.historyList.appendChild(li);
    }
  }

  /* ---------- events ---------- */
  const openPicker = () => { if (!state.busy) el.fileInput.click(); };

  el.chooseBtn.addEventListener("click", openPicker);
  el.finder.addEventListener("click", () => { if (!el.finder.classList.contains("has-image")) openPicker(); });
  el.finder.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !el.finder.classList.contains("has-image")) { e.preventDefault(); openPicker(); }
  });
  el.fileInput.addEventListener("change", () => handleFiles(el.fileInput.files));
  el.resetBtn.addEventListener("click", () => { if (!state.busy) { hideAlert(); clearPreview(); setView("idle"); } });
  el.alertClose.addEventListener("click", hideAlert);
  el.clearHistory.addEventListener("click", () => {
    state.history.forEach((h) => URL.revokeObjectURL(h.url));
    state.history = []; clearPreview(); setView("idle"); renderHistory();
  });

  ["dragenter", "dragover"].forEach((t) => el.finder.addEventListener(t, (e) => { e.preventDefault(); el.finder.classList.add("is-drag"); }));
  ["dragleave", "drop"].forEach((t) => el.finder.addEventListener(t, (e) => { e.preventDefault(); el.finder.classList.remove("is-drag"); }));
  el.finder.addEventListener("drop", (e) => handleFiles(e.dataTransfer.files));

  // stop the browser from opening a photo dropped outside the drop area
  ["dragover", "drop"].forEach((t) => window.addEventListener(t, (e) => e.preventDefault()));

  document.addEventListener("paste", (e) => {
    const files = Array.from(e.clipboardData?.files || []).filter((f) => f.type.startsWith("image/"));
    if (files.length) handleFiles(files);
  });

  checkHealth();
})();
