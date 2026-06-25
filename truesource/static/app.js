const $ = (s) => document.querySelector(s);
const BASE = window.location.pathname.replace(/\/$/, "");
const api = (p) => `${BASE}${p}`;

let currentBlob = null; // File or Blob to send
let currentSample = null; // sample filename
let stream = null;

/* ---------- Toast ---------- */
function toast(msg) {
  let t = $(".toast");
  if (!t) {
    t = document.createElement("div");
    t.className = "toast";
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), 3200);
}

/* ---------- Model status polling ---------- */
let modelReady = false;
async function pollStatus() {
  try {
    const r = await fetch(api("/healthz"));
    const data = await r.json();
    const pill = $("#modelPill");
    const txt = $("#modelStatusText");
    if (data.ready) {
      modelReady = true;
      pill.classList.add("ready");
      pill.classList.remove("error");
      txt.textContent = "Model ready";
      return;
    }
    if (data.error) {
      pill.classList.add("error");
      txt.textContent = "Model failed to load";
      return;
    }
    txt.textContent = "Loading model…";
    setTimeout(pollStatus, 1500);
  } catch (e) {
    setTimeout(pollStatus, 2500);
  }
}

/* ---------- Mode switching ---------- */
document.querySelectorAll(".seg-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const mode = btn.dataset.mode;
    $("#uploadMode").classList.toggle("active", mode === "upload");
    $("#cameraMode").classList.toggle("active", mode === "camera");
    if (mode !== "camera") stopCamera();
  });
});

/* ---------- Upload / drag & drop ---------- */
const dropzone = $("#dropzone");
const fileInput = $("#fileInput");
fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) setImageFromFile(e.target.files[0]);
});
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) setImageFromFile(f);
});

function setImageFromFile(file) {
  currentBlob = file;
  currentSample = null;
  showPreview(URL.createObjectURL(file), "Selected from your device");
}

/* ---------- Camera ---------- */
$("#startCam").addEventListener("click", startCamera);
$("#snap").addEventListener("click", capturePhoto);

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    const v = $("#video");
    v.srcObject = stream;
    v.classList.add("live");
    $("#cameraEmpty").style.display = "none";
    $("#snap").disabled = false;
    $("#startCam").textContent = "Restart camera";
  } catch (e) {
    toast("Couldn't access the camera. Check permissions.");
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  const v = $("#video");
  v.classList.remove("live");
  v.srcObject = null;
  $("#cameraEmpty").style.display = "grid";
  $("#snap").disabled = true;
  $("#startCam").textContent = "Start camera";
}

function capturePhoto() {
  const v = $("#video");
  const canvas = $("#snapCanvas");
  canvas.width = v.videoWidth;
  canvas.height = v.videoHeight;
  canvas.getContext("2d").drawImage(v, 0, 0);
  canvas.toBlob((blob) => {
    currentBlob = blob;
    currentSample = null;
    showPreview(URL.createObjectURL(blob), "Captured from your camera");
    stopCamera();
  }, "image/png");
}

/* ---------- Preview ---------- */
function showPreview(src, label) {
  $("#previewImg").src = src;
  $("#previewLabel").textContent = label;
  $("#previewRow").hidden = false;
  resetResult();
}

$("#clearBtn").addEventListener("click", () => {
  currentBlob = null;
  currentSample = null;
  $("#previewRow").hidden = true;
  fileInput.value = "";
  resetResult();
});

$("#analyzeBtn").addEventListener("click", analyze);

/* ---------- Result rendering ---------- */
function resetResult() {
  $("#resultEmpty").hidden = false;
  $("#resultLoading").hidden = true;
  $("#resultCard").hidden = true;
}

async function analyze() {
  if (!currentBlob && !currentSample) {
    toast("Pick an image first.");
    return;
  }
  if (!modelReady) {
    toast("The model is still warming up — one moment.");
  }
  $("#resultEmpty").hidden = true;
  $("#resultCard").hidden = true;
  $("#resultLoading").hidden = false;

  // Keep the scanning animation visible for at least 2s after clicking, then drop it.
  const started = Date.now();
  const settle = async () => {
    const elapsed = Date.now() - started;
    if (elapsed < 2000) await new Promise((r) => setTimeout(r, 2000 - elapsed));
  };

  try {
    let res;
    if (currentSample) {
      res = await fetch(api("/predict"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample: currentSample }),
      });
    } else {
      const fd = new FormData();
      fd.append("file", currentBlob, "upload.png");
      res = await fetch(api("/predict"), { method: "POST", body: fd });
    }
    const data = await res.json();
    await settle();
    if (!res.ok) {
      if (data.error === "model_loading") {
        toast("Model still loading — try again in a few seconds.");
      } else if (data.error === "model_failed") {
        modelReady = false;
        const pill = $("#modelPill");
        const txt = $("#modelStatusText");
        if (pill) { pill.classList.add("error"); pill.classList.remove("ready"); }
        if (txt) txt.textContent = "Model failed to load";
        toast(data.message || "The detection model failed to load.");
      } else {
        toast(data.message || "Something went wrong.");
      }
      resetResult();
      return;
    }
    renderResult(data);
  } catch (e) {
    await settle();
    toast("Network error — please try again.");
    resetResult();
  }
}

function renderResult(data) {
  $("#resultLoading").hidden = true;
  $("#resultCard").hidden = false;

  const isAi = data.label === "AI-Generated";
  const pctReal = Math.round(data.prob_real * 100);
  const pctAi = Math.round(data.prob_ai * 100);
  const conf = Math.round(data.confidence * 100);

  const badge = $("#verdictBadge");
  badge.classList.remove("real", "ai");
  badge.classList.add(isAi ? "ai" : "real");
  $("#verdictLabel").textContent = isAi ? "AI-Generated" : "Real";
  $("#verdictConf").textContent = `${conf}% confident`;

  // Only the winning side is emphasized; bars animate from 0.
  const fillReal = $("#fillReal");
  const fillAi = $("#fillAi");
  fillReal.style.width = "0%";
  fillAi.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      fillReal.style.width = `${pctReal}%`;
      fillAi.style.width = `${pctAi}%`;
    });
  });
  $("#valReal").textContent = `${pctReal}%`;
  $("#valAi").textContent = `${pctAi}%`;

  let note;
  if (isAi) {
    note =
      conf >= 85
        ? "Strong signs of synthetic generation — pixel patterns line up with AI-created imagery."
        : "Leans AI-generated, but it's a closer call. Inspect the details below before you trust it.";
  } else {
    note =
      conf >= 85
        ? "Looks like a genuine photograph — the texture and noise read as authentic."
        : "Leans real, but the model isn't fully certain. Cross-check with the tips below.";
  }
  $("#verdictNote").textContent = note;
}

/* ---------- Samples ---------- */
async function loadSamples() {
  try {
    const r = await fetch(api("/samples-list"));
    const items = await r.json();
    const wrap = $("#samples");
    wrap.innerHTML = "";
    items.forEach((it) => {
      const el = document.createElement("div");
      el.className = "sample";
      const isAi = it.truth === "AI-Generated";
      el.innerHTML = `
        <span class="tag ${isAi ? "ai" : "real"}">${isAi ? "AI" : "Real"}</span>
        <img src="${api("/" + it.url)}" alt="${it.name}" loading="lazy" />
        <div class="hover-cta">Analyze this</div>`;
      el.addEventListener("click", () => {
        currentSample = it.name;
        currentBlob = null;
        showPreview(api("/" + it.url), `Sample · labeled ${it.truth}`);
        document.querySelector(".workspace").scrollIntoView({ behavior: "smooth", block: "start" });
      });
      wrap.appendChild(el);
    });
  } catch (e) {
    /* silent */
  }
}

/* ---------- Tips ---------- */
const TIPS = [
  {
    color: "linear-gradient(135deg,#8a6cff,#a9d8ff)",
    title: "Check the hands & teeth",
    body: "AI still fumbles fine detail — count fingers, look for warped teeth, mangled jewelry or melted accessories.",
    icon: '<path d="M18 11V6a2 2 0 0 0-2-2 2 2 0 0 0-2 2"/><path d="M14 10V4a2 2 0 0 0-2-2 2 2 0 0 0-2 2v2"/><path d="M10 10.5V6a2 2 0 0 0-2-2 2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/>',
  },
  {
    color: "linear-gradient(135deg,#ff5f86,#ffc4dd)",
    title: "Read the background",
    body: "Look for nonsense text, repeating patterns, fused objects, or edges that bend where they shouldn't.",
    icon: '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="m9 13 2 2 4-4"/>',
  },
  {
    color: "linear-gradient(135deg,#2bbd8a,#9ff0d6)",
    title: "Watch the light",
    body: "Shadows pointing different ways, missing reflections, or skin that looks too smooth and waxy are classic tells.",
    icon: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4"/>',
  },
  {
    color: "linear-gradient(135deg,#a9d8ff,#8a6cff)",
    title: "Zoom into textures",
    body: "Hair, fabric and foliage can dissolve into mush under magnification when an image is synthetic.",
    icon: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/><path d="M11 8v6M8 11h6"/>',
  },
];

function loadTips() {
  const wrap = $("#tips");
  wrap.innerHTML = "";
  TIPS.forEach((t) => {
    const el = document.createElement("div");
    el.className = "tip";
    el.innerHTML = `
      <div class="tip-ico" style="background:${t.color}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${t.icon}</svg>
      </div>
      <h3>${t.title}</h3>
      <p>${t.body}</p>`;
    wrap.appendChild(el);
  });
}

/* ---------- Init ---------- */
pollStatus();
loadSamples();
loadTips();
