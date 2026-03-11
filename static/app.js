const SYNTHETIC_SIGNALS = {
    sensor_movimiento:     { label: "Sensor Movimiento",       type: "binary" },
    camara_activa:         { label: "Cámara Activa",           type: "binary" },
    hora_nocturna:         { label: "Hora Nocturna",           type: "binary" },
    zona_riesgo:           { label: "Zona de Riesgo",          type: "binary" },
    sensor_puerta:         { label: "Sensor Puerta",           type: "binary" },
    sensor_ventana:        { label: "Sensor Ventana",          type: "binary" },
    nivel_ruido:           { label: "Nivel de Ruido",          type: "range", min: 0, max: 1, step: 0.01, default: 0.5 },
    historico_incidentes:  { label: "Histórico Incidentes",    type: "range", min: 0, max: 1, step: 0.01, default: 0.5 },
};

const UCF_SIGNALS = {
    avg_personas:           { label: "Prom. Personas",         type: "range", min: 0, max: 10,  step: 0.1,  default: 1 },
    max_personas:           { label: "Máx. Personas",          type: "range", min: 0, max: 20,  step: 1,    default: 2 },
    avg_confianza_persona:  { label: "Confianza Persona",      type: "range", min: 0, max: 1,   step: 0.01, default: 0.5 },
    area_persona_max:       { label: "Área Persona Máx.",      type: "range", min: 0, max: 1,   step: 0.01, default: 0.1 },
    intensidad_movimiento:  { label: "Intensidad Movimiento",  type: "range", min: 0, max: 1,   step: 0.01, default: 0.3 },
    clases_unicas:          { label: "Clases Únicas",          type: "range", min: 0, max: 20,  step: 1,    default: 3 },
    detecciones_promedio:   { label: "Detecciones Prom.",      type: "range", min: 0, max: 30,  step: 0.5,  default: 3 },
    velocidad_persona:      { label: "Velocidad Persona",      type: "range", min: 0, max: 1,   step: 0.01, default: 0.1 },
};

const LABEL_DISPLAY = {
    intrusion_probable:             "Intrusión Probable",
    requiere_verificacion_visual:   "Verificación Visual",
    notificar_propietario:          "Notificar Propietario",
    despachar_movil:                "Despachar Móvil",
};

const SEVERITY_CLASS = {
    NORMAL:   "sev-normal",
    BAJO:     "sev-bajo",
    MEDIO:    "sev-medio",
    ALTO:     "sev-alto",
    CRÍTICO:  "sev-critico",
};

let datasetReady = false;
let currentSource = "synthetic";
let extractionPollId = null;

// ── Init ──────────────────────────────────────────────────────────────────────

function init() {
    buildSignalControls();
    checkStatus();
    loadIncidents();
    startClock();
}

function startClock() {
    const el = document.getElementById("header-clock");
    const tick = () => { el.textContent = new Date().toLocaleTimeString("es-ES"); };
    tick();
    setInterval(tick, 1000);
}

// ── Status ────────────────────────────────────────────────────────────────────

async function checkStatus() {
    try {
        const res = await fetch("/api/info");
        const data = await res.json();

        const trained = data.sklearn_trained || data.pytorch_trained;
        setDot("dataset", data.n_samples > 0);
        setDot("model", trained);
        document.getElementById("status-dataset").textContent =
            data.n_samples > 0 ? `${data.n_samples} muestras (${data.data_source})` : "Sin datos";
        document.getElementById("status-model").textContent =
            trained ? "Entrenado" : "Sin entrenar";

        if (data.n_samples > 0) {
            datasetReady = true;
            if (data.data_source === "ucf-crime") {
                currentSource = "ucf";
                switchTab("ucf");
            }
        }
    } catch {
        // servidor no listo aún
    }
}

function setDot(name, active) {
    document.getElementById(`dot-${name}`).classList.toggle("active", active);
}

// ── Signal Controls ───────────────────────────────────────────────────────────

function getActiveSignals() {
    return currentSource === "ucf" ? UCF_SIGNALS : SYNTHETIC_SIGNALS;
}

function buildSignalControls() {
    const list = document.getElementById("signals-list");
    list.innerHTML = "";
    const signals = getActiveSignals();

    for (const [key, info] of Object.entries(signals)) {
        const row = document.createElement("div");
        row.className = "signal-row";

        const min   = info.min  ?? 0;
        const max   = info.max  ?? 1;
        const step  = info.step ?? (info.type === "binary" ? 1 : 0.01);
        const def   = info.type === "binary" ? 0 : (info.default ?? (max - min) / 2);

        row.innerHTML = `
            <span class="signal-name">${info.label}</span>
            <input type="range" id="signal-${key}"
                   min="${min}" max="${max}" step="${step}" value="${def}"
                   oninput="updateSignalVal(this, '${key}', ${step < 1 ? 2 : 0})">
            <span class="signal-val" id="val-${key}">${def}</span>
        `;
        list.appendChild(row);
    }
}

function updateSignalVal(input, key, decimals) {
    document.getElementById(`val-${key}`).textContent =
        parseFloat(input.value).toFixed(decimals);
}

function switchTab(tab) {
    currentSource = tab;
    document.getElementById("tab-synthetic").classList.toggle("hidden", tab !== "synthetic");
    document.getElementById("tab-ucf").classList.toggle("hidden", tab !== "ucf");
    document.querySelectorAll(".tab").forEach((t, i) => {
        t.classList.toggle("active", (tab === "synthetic" && i === 0) || (tab === "ucf" && i === 1));
    });
    buildSignalControls();
}

// ── Loading ───────────────────────────────────────────────────────────────────

function setLoading(on, text = "Procesando...") {
    document.getElementById("loading").classList.toggle("hidden", !on);
    document.getElementById("loading-text").textContent = text;
}

// ── API helper ────────────────────────────────────────────────────────────────

async function apiCall(endpoint, body = {}, method = "POST") {
    setLoading(true);
    try {
        const opts = { method, headers: { "Content-Type": "application/json" } };
        if (method !== "GET") opts.body = JSON.stringify(body);
        const res = await fetch(endpoint, opts);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Error del servidor");
        return data;
    } catch (err) {
        alert(err.message);
        return null;
    } finally {
        setLoading(false);
    }
}

// ── Dataset ───────────────────────────────────────────────────────────────────

async function generateDataset() {
    const n = parseInt(document.getElementById("n-samples").value);
    const data = await apiCall("/api/generate", { n_samples: n });
    if (!data) return;

    datasetReady = true;
    currentSource = "synthetic";
    setDot("dataset", true);
    document.getElementById("status-dataset").textContent = `${data.n_samples} muestras (sintético)`;

    let html = `Dataset generado: <strong>${data.n_samples} muestras</strong>`;
    showResult("setup-result", html);
    buildSignalControls();
}

// ── UCF-Crime ─────────────────────────────────────────────────────────────────

async function ucfSetup() {
    const data = await apiCall("/api/ucf/setup");
    if (!data) return;
    showResult("setup-result", `${data.mensaje} — <code>${data.path}</code>`);
}

async function ucfCheckStatus() {
    const data = await apiCall("/api/ucf/status", {}, "GET");
    if (!data) return;
    const lines = Object.entries(data.categorias)
        .map(([cat, n]) => `${n > 0 ? "✅" : "❌"} ${cat}: ${n}`)
        .join(" &nbsp;|&nbsp; ");
    showResult("setup-result", `Total: <strong>${data.total_videos}</strong> videos &nbsp;|&nbsp; ${lines}`);
}

async function ucfExtract() {
    const frameInterval = parseInt(document.getElementById("frame-interval").value);
    const maxFrames = parseInt(document.getElementById("max-frames").value);
    setLoading(true);
    try {
        const res = await fetch("/api/ucf/extract", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame_interval: frameInterval, max_frames: maxFrames }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        setLoading(false);
        showResult("setup-result", data.mensaje);
        document.getElementById("extraction-progress").classList.remove("hidden");
        pollExtraction();
    } catch (err) {
        setLoading(false);
        alert(err.message);
    }
}

function pollExtraction() {
    if (extractionPollId) clearInterval(extractionPollId);
    extractionPollId = setInterval(async () => {
        try {
            const s = await (await fetch("/api/ucf/extract/status")).json();
            const pct = s.total > 0 ? (s.progress / s.total) * 100 : 0;
            document.getElementById("progress-fill").style.width = `${pct}%`;
            document.getElementById("progress-text").textContent = `${s.progress} / ${s.total}`;
            document.getElementById("progress-video").textContent = s.current_video;

            if (s.done) {
                clearInterval(extractionPollId);
                extractionPollId = null;
                if (s.error) {
                    showResult("setup-result", `<span style="color:#e74c3c">Error: ${s.error}</span>`);
                } else {
                    datasetReady = true;
                    currentSource = "ucf";
                    setDot("dataset", true);
                    document.getElementById("status-dataset").textContent =
                        `${s.n_samples} muestras (UCF-Crime)`;
                    showResult("setup-result", `Extracción completada: <strong>${s.n_samples} muestras</strong>`);
                    buildSignalControls();
                }
            }
        } catch { /* ignore poll errors */ }
    }, 2000);
}

// ── Train ─────────────────────────────────────────────────────────────────────

async function trainModel() {
    if (!datasetReady) { alert("Primero genera o extrae un dataset"); return; }

    const modelType = document.getElementById("model-select").value;
    const body = { model: modelType };
    if (modelType === "pytorch") {
        body.epochs = parseInt(document.getElementById("epochs").value);
    }

    setLoading(true, "Entrenando modelo...");
    const data = await apiCall("/api/train", body);
    if (!data) return;

    setDot("model", true);
    document.getElementById("status-model").textContent = `${modelType} entrenado`;

    const m = data.metrics;
    let html = `<strong>${modelType.toUpperCase()}</strong>`;
    html += ` &nbsp;|&nbsp; F1 Micro: <strong>${m.f1_micro}</strong>`;
    html += ` &nbsp;|&nbsp; Hamming: <strong>${m.hamming_loss}</strong>`;
    html += ` &nbsp;|&nbsp; ${m.samples_train} train / ${m.samples_test} test`;
    if (m.epochs) html += ` &nbsp;|&nbsp; ${m.epochs} epochs, loss: ${m.final_loss}`;
    showResult("train-result", html);
}

// ── Analyze (predict + LLM report) ───────────────────────────────────────────

async function analyze() {
    const modelType = document.getElementById("model-select").value;
    const signals = [];
    for (const key of Object.keys(getActiveSignals())) {
        signals.push(parseFloat(document.getElementById(`signal-${key}`).value));
    }

    setLoading(true, "Analizando evento y generando reporte con IA...");
    const data = await apiCall("/api/report", { model: modelType, signals });
    setLoading(false);
    if (!data) return;

    renderLabels(data.prediction);
    renderReport(data.report, modelType);
    loadIncidents();
}

// ── Detection Labels ──────────────────────────────────────────────────────────

function renderLabels(prediction) {
    const grid = document.getElementById("labels-display");
    grid.innerHTML = "";

    for (const [label, info] of Object.entries(prediction)) {
        const card = document.createElement("div");
        card.className = `label-card ${info.activo ? "lc-active" : "lc-inactive"}`;
        card.innerHTML = `
            <div class="lc-name">${LABEL_DISPLAY[label] || label}</div>
            <div class="lc-prob">${(info.probabilidad * 100).toFixed(1)}%</div>
            <div class="lc-status">${info.activo ? "ACTIVADO" : "inactivo"}</div>
        `;
        grid.appendChild(card);
    }
}

// ── Incident Report ───────────────────────────────────────────────────────────

function renderReport(report, modelType) {
    const sevClass = SEVERITY_CLASS[report.severidad] || "sev-normal";
    const ts = new Date(report.timestamp).toLocaleTimeString("es-ES");

    const actionsHtml = report.acciones?.length
        ? `<ol class="report-actions">${report.acciones.map(a => `<li>${a}</li>`).join("")}</ol>`
        : "";

    document.getElementById("report-content").innerHTML = `
        <div class="report-meta">
            <span class="severity-badge ${sevClass}">${report.severidad}</span>
            <span class="report-ts">${ts}</span>
            <span class="report-model-tag">${modelType}</span>
        </div>
        <h3 class="report-title">${report.titulo}</h3>
        ${report.resumen  ? `<div class="report-block"><span class="report-label">Resumen</span><p>${report.resumen}</p></div>`   : ""}
        ${report.analisis ? `<div class="report-block"><span class="report-label">Análisis Técnico</span><p>${report.analisis}</p></div>` : ""}
        ${actionsHtml     ? `<div class="report-block"><span class="report-label">Acciones Recomendadas</span>${actionsHtml}</div>` : ""}
        ${report.riesgo   ? `<div class="report-block"><span class="report-label">Evaluación de Riesgo</span><p>${report.riesgo}</p></div>`   : ""}
        ${report.error    ? `<div class="report-block error-block"><span class="report-label">Error</span><p>${report.error}</p></div>` : ""}
    `;
}

// ── Incident Log ──────────────────────────────────────────────────────────────

async function loadIncidents() {
    try {
        const data = await (await fetch("/api/incidents")).json();
        renderIncidentLog(data.incidents);
    } catch { /* ignore */ }
}

function renderIncidentLog(incidents) {
    const container = document.getElementById("incidents-container");
    if (!incidents.length) {
        container.innerHTML = '<div class="placeholder-msg">No hay incidentes registrados</div>';
        return;
    }

    const rows = [...incidents].reverse().map(inc => {
        const r = inc.report;
        const sc = SEVERITY_CLASS[r.severidad] || "sev-normal";
        const ts = new Date(r.timestamp).toLocaleTimeString("es-ES");
        const active = Object.entries(inc.prediction)
            .filter(([, v]) => v.activo)
            .map(([k]) => LABEL_DISPLAY[k] || k)
            .join(", ") || "ninguno";
        return `
            <tr>
                <td><span class="severity-badge ${sc}">${r.severidad}</span></td>
                <td class="col-time">${ts}</td>
                <td class="col-title">${r.titulo}</td>
                <td class="col-labels">${active}</td>
                <td class="col-model">${inc.model}</td>
            </tr>`;
    }).join("");

    container.innerHTML = `
        <table class="incident-table">
            <thead>
                <tr>
                    <th>Severidad</th>
                    <th>Hora</th>
                    <th>Título</th>
                    <th>Labels Activos</th>
                    <th>Modelo</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>`;
}

// ── Utils ─────────────────────────────────────────────────────────────────────

function showResult(id, html) {
    const el = document.getElementById(id);
    el.classList.remove("hidden");
    el.innerHTML = html;
}

document.addEventListener("DOMContentLoaded", init);
