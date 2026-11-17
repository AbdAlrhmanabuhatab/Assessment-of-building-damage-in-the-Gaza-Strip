// Gaza Building Damage Detection AI - Frontend Controller

let selectedFile = null;
let currentAssessmentResult = null;

document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  initApp();
});

async function initApp() {
  fetchHealth();
  fetchSamples();
  setupEventListeners();
}

async function fetchHealth() {
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    if (res.ok) {
      const data = await res.json();
      const badge = document.getElementById("model-name-badge");
      if (badge) {
        badge.textContent = data.model || "MobileNetV2 V2";
      }
    }
  } catch (err) {
    console.warn("Could not connect to health endpoint:", err);
  }
}

async function fetchSamples() {
  const container = document.getElementById("samples-container");
  try {
    const res = await fetch(`/api/samples?_t=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) throw new Error("Failed to fetch samples");
    const samples = await res.json();

    if (!samples.length) {
      container.innerHTML = `<p class="text-xs text-slate-500 col-span-3">No preset samples found.</p>`;
      return;
    }

    container.innerHTML = "";
    samples.forEach(sample => {
      const card = document.createElement("div");
      card.className = "group relative rounded-xl overflow-hidden border border-slate-800 hover:border-sky-500/80 cursor-pointer transition-all duration-200 bg-slate-950/60 hover:shadow-lg hover:shadow-sky-500/10";
      
      let badgeColor = "bg-rose-500/80";
      if (sample.category === "Repair") badgeColor = "bg-amber-500/80";
      if (sample.category === "Safe") badgeColor = "bg-emerald-500/80";

      // Append cache buster to img src to force fresh image render
      const imgSrc = sample.url.includes("?") ? `${sample.url}&_c=${Date.now()}` : `${sample.url}?_c=${Date.now()}`;

      card.innerHTML = `
        <div class="aspect-square w-full overflow-hidden bg-slate-900">
          <img src="${imgSrc}" alt="${sample.title}" class="w-full h-full object-cover group-hover:scale-105 transition duration-300">
        </div>
        <div class="p-2 text-left">
          <span class="inline-block text-[9px] font-mono font-bold px-1.5 py-0.5 rounded text-white ${badgeColor} mb-1">${sample.category}</span>
          <p class="text-[11px] font-semibold text-slate-200 truncate">${sample.title}</p>
        </div>
      `;

      card.addEventListener("click", () => loadSampleFromUrl(sample.url, sample.id));
      container.appendChild(card);
    });

  } catch (err) {
    console.error("Error loading samples:", err);
    container.innerHTML = `<p class="text-xs text-slate-500 col-span-3">Samples unavailable.</p>`;
  }
}

async function loadSampleFromUrl(url, filename) {
  try {
    const fetchUrl = url.includes("?") ? `${url}&_bust=${Date.now()}` : `${url}?_bust=${Date.now()}`;
    const res = await fetch(fetchUrl, { cache: "no-store" });
    const blob = await res.blob();
    selectedFile = new File([blob], filename, { type: blob.type || "image/jpeg" });
    
    displayPreview(selectedFile);
    runAssessment();
  } catch (err) {
    console.error("Failed to load sample image:", err);
  }
}

function setupEventListeners() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const clearBtn = document.getElementById("clear-image-btn");
  const analyzeBtn = document.getElementById("analyze-btn");
  const downloadJsonBtn = document.getElementById("download-json-btn");

  dropZone.addEventListener("click", (e) => {
    if (e.target.closest("#clear-image-btn")) return;
    fileInput.click();
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      selectedFile = e.target.files[0];
      displayPreview(selectedFile);
    }
  });

  ["dragenter", "dragover"].forEach(event => {
    dropZone.addEventListener(event, (e) => {
      e.preventDefault();
      dropZone.classList.add("border-sky-500", "bg-sky-500/10");
    });
  });

  ["dragleave", "drop"].forEach(event => {
    dropZone.addEventListener(event, (e) => {
      e.preventDefault();
      dropZone.classList.remove("border-sky-500", "bg-sky-500/10");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      selectedFile = e.dataTransfer.files[0];
      displayPreview(selectedFile);
    }
  });

  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
  });

  analyzeBtn.addEventListener("click", () => {
    if (selectedFile) {
      runAssessment();
    }
  });

  document.getElementById("view-tab-overlay").addEventListener("click", () => switchGradCamView("overlay"));
  document.getElementById("view-tab-split").addEventListener("click", () => switchGradCamView("split"));
  document.getElementById("view-tab-heatmap").addEventListener("click", () => switchGradCamView("heatmap"));

  if (downloadJsonBtn) {
    downloadJsonBtn.addEventListener("click", () => {
      if (!currentAssessmentResult) return;
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentAssessmentResult, null, 2));
      const downloadAnchor = document.createElement("a");
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `structural_assessment_${Date.now()}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    });
  }
}

function displayPreview(file) {
  const uploadPrompt = document.getElementById("upload-prompt");
  const previewContainer = document.getElementById("preview-container");
  const previewImg = document.getElementById("image-preview");
  const previewFilename = document.getElementById("preview-filename");
  const clearBtn = document.getElementById("clear-image-btn");
  const analyzeBtn = document.getElementById("analyze-btn");

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewFilename.textContent = file.name;
    uploadPrompt.classList.add("hidden");
    previewContainer.classList.remove("hidden");
    clearBtn.classList.remove("hidden");

    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove("bg-slate-800", "text-slate-500", "cursor-not-allowed");
    analyzeBtn.classList.add("bg-sky-600", "hover:bg-sky-500", "text-white", "shadow-lg", "shadow-sky-500/20");
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  selectedFile = null;
  currentAssessmentResult = null;
  document.getElementById("file-input").value = "";
  document.getElementById("upload-prompt").classList.remove("hidden");
  document.getElementById("preview-container").classList.add("hidden");
  document.getElementById("clear-image-btn").classList.add("hidden");
  
  const analyzeBtn = document.getElementById("analyze-btn");
  analyzeBtn.disabled = true;
  analyzeBtn.classList.add("bg-slate-800", "text-slate-500", "cursor-not-allowed");
  analyzeBtn.classList.remove("bg-sky-600", "hover:bg-sky-500", "text-white", "shadow-lg", "shadow-sky-500/20");

  document.getElementById("results-state").classList.add("hidden");
  document.getElementById("loading-state").classList.add("hidden");
  document.getElementById("idle-state").classList.remove("hidden");
}

async function runAssessment() {
  if (!selectedFile) return;

  const idleState = document.getElementById("idle-state");
  const loadingState = document.getElementById("loading-state");
  const resultsState = document.getElementById("results-state");

  idleState.classList.add("hidden");
  resultsState.classList.add("hidden");
  loadingState.classList.remove("hidden");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
      cache: "no-store"
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned error status ${response.status}`);
    }

    const data = await response.json();
    currentAssessmentResult = data;
    renderAssessmentResults(data);

  } catch (err) {
    console.error("Assessment failed:", err);
    alert(`Assessment Error: ${err.message}`);
    loadingState.classList.add("hidden");
    idleState.classList.remove("hidden");
  }
}

function renderAssessmentResults(data) {
  document.getElementById("loading-state").classList.add("hidden");
  const resultsState = document.getElementById("results-state");
  resultsState.classList.remove("hidden");

  const verdictCard = document.getElementById("verdict-card");
  const verdictTitle = document.getElementById("verdict-title");
  const verdictPill = document.getElementById("verdict-pill");
  const verdictSummary = document.getElementById("verdict-summary");
  const confidenceVal = document.getElementById("confidence-value");

  verdictCard.className = "rounded-2xl p-6 border transition-all duration-300 relative overflow-hidden";

  const pred = data.prediction;
  const conf = data.confidence;
  const adv = data.advisory || {};

  confidenceVal.textContent = `${conf.toFixed(1)}%`;
  verdictTitle.textContent = pred.toUpperCase();
  verdictSummary.textContent = adv.action_summary || "Structural integrity assessment complete.";

  if (pred === "Demolish") {
    verdictCard.classList.add("verdict-demolish");
    verdictPill.className = "text-xs font-mono font-bold px-3 py-1 rounded-full uppercase bg-rose-500/20 text-rose-300 border border-rose-500/40";
    verdictPill.textContent = "CRITICAL COLLAPSE";
  } else if (pred === "Repair") {
    verdictCard.classList.add("verdict-repair");
    verdictPill.className = "text-xs font-mono font-bold px-3 py-1 rounded-full uppercase bg-amber-500/20 text-amber-300 border border-amber-500/40";
    verdictPill.textContent = "MODERATE DAMAGE";
  } else {
    verdictCard.classList.add("verdict-safe");
    verdictPill.className = "text-xs font-mono font-bold px-3 py-1 rounded-full uppercase bg-emerald-500/20 text-emerald-300 border border-emerald-500/40";
    verdictPill.textContent = "LOW HAZARD";
  }

  const probs = data.probabilities || {};
  const probDemolish = probs.Demolish || 0;
  const probRepair = probs.Repair || 0;
  const probSafe = probs.Safe || 0;

  document.getElementById("prob-demolish").textContent = `${probDemolish.toFixed(1)}%`;
  document.getElementById("bar-demolish").style.width = `${probDemolish}%`;

  document.getElementById("prob-repair").textContent = `${probRepair.toFixed(1)}%`;
  document.getElementById("bar-repair").style.width = `${probRepair}%`;

  document.getElementById("prob-safe").textContent = `${probSafe.toFixed(1)}%`;
  document.getElementById("bar-safe").style.width = `${probSafe}%`;

  if (data.gradcam_overlay_base64) {
    document.getElementById("gradcam-overlay-img").src = data.gradcam_overlay_base64;
    document.getElementById("split-orig-img").src = data.original_image_base64 || "";
    document.getElementById("split-cam-img").src = data.gradcam_overlay_base64;
    document.getElementById("gradcam-thermal-img").src = data.gradcam_heatmap_base64 || "";
  }

  const protocolsList = document.getElementById("protocols-list");
  protocolsList.innerHTML = "";
  const protocols = adv.protocols || [];
  protocols.forEach(protocol => {
    const li = document.createElement("li");
    li.className = "flex items-start gap-2 text-slate-300";
    li.innerHTML = `
      <i data-lucide="check-circle-2" class="w-4 h-4 text-sky-400 mt-0.5 shrink-0"></i>
      <span>${protocol}</span>
    `;
    protocolsList.appendChild(li);
  });

  if (window.lucide) {
    lucide.createIcons();
  }
}

function switchGradCamView(viewMode) {
  const tabs = {
    overlay: document.getElementById("view-tab-overlay"),
    split: document.getElementById("view-tab-split"),
    heatmap: document.getElementById("view-tab-heatmap")
  };
  const displays = {
    overlay: document.getElementById("display-overlay"),
    split: document.getElementById("display-split"),
    heatmap: document.getElementById("display-heatmap")
  };

  Object.keys(tabs).forEach(k => {
    tabs[k].className = "px-3 py-1 rounded-lg text-slate-400 hover:text-white transition";
    displays[k].classList.add("hidden");
  });

  tabs[viewMode].className = "px-3 py-1 rounded-lg bg-sky-600 text-white font-medium transition";
  displays[viewMode].classList.remove("hidden");
}
