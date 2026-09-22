(() => {
  const state = { runs: [], sort: { key: "executionAccuracy", direction: "desc" }, query: "" };
  const body = document.querySelector("#runs-body");
  const empty = document.querySelector("#empty-state");
  const summary = document.querySelector("#summary");
  const number = (value, digits = 2) => Number.isFinite(value) ? value.toLocaleString("fr-FR", { maximumFractionDigits: digits }) : "—";
  const percent = value => Number.isFinite(value) ? `${(value * 100).toLocaleString("fr-FR", { maximumFractionDigits: 2 })} %` : "—";
  const valueAt = (object, path) => path.split(".").reduce((item, key) => item?.[key], object);

  function makeRun(manifest, metrics, source = "") {
    const gen = manifest?.generation || {};
    const evaluation = metrics?.evaluation || {};
    const accuracy = evaluation.accuracy || {};
    const modelPath = manifest?.model?.path || metrics?.source?.generation?.model_path || "Modèle inconnu";
    return {
      label: source || manifest?.artifact?.created_at_utc || metrics?.artifact?.created_at_utc || "Run importé",
      createdAt: manifest?.artifact?.created_at_utc || metrics?.artifact?.created_at_utc,
      model: modelPath.split(/[\\/]/).filter(Boolean).pop(),
      mode: manifest?.configuration?.mode || metrics?.configuration?.mode || "—",
      environment: manifest?.environment?.hostname || metrics?.environment?.hostname || "—",
      groupedBatches: manifest?.configuration?.group_batches_by_length,
      paddingRate: gen.tokens?.padding?.rate,
      executionAccuracy: accuracy.execution?.rate,
      exactMatch: accuracy.exact_match?.rate,
      syntaxValid: accuracy.syntax?.valid_rate,
      generationEps: gen.throughput?.examples_per_second,
      generationDuration: gen.duration_seconds ? gen.duration_seconds / 60 : undefined,
      evaluationDuration: evaluation.duration_seconds,
      gpuPeak: manifest?.gpu?.vram_mib?.peak,
      energy: manifest?.gpu?.energy_wh?.total,
      truncationRate: gen.reliability?.truncation_rate,
    };
  }

  function render() {
    const filtered = state.runs.filter(run => Object.values(run).join(" ").toLowerCase().includes(state.query));
    const { key, direction } = state.sort;
    filtered.sort((a, b) => {
      const av = a[key], bv = b[key];
      if (av == null) return 1; if (bv == null) return -1;
      return (typeof av === "number" ? av - bv : String(av).localeCompare(String(bv))) * (direction === "asc" ? 1 : -1);
    });
    const bestAccuracy = Math.max(...filtered.map(run => run.executionAccuracy ?? -Infinity));
    body.innerHTML = filtered.map(run => `<tr>
      <td><span class="run-name">${escapeHtml(run.label)}</span><span class="subtext">${escapeHtml(run.createdAt || "date inconnue")}</span></td>
      <td>${escapeHtml(run.model || "—")}</td><td>${escapeHtml(run.mode)}</td>
      <td>${run.groupedBatches == null ? "<span class=\"na\">—</span>" : run.groupedBatches ? "Oui" : "Non"}</td>
      <td class="metric">${percent(run.paddingRate)}</td>
      <td class="metric ${run.executionAccuracy === bestAccuracy ? "best" : ""}">${percent(run.executionAccuracy)}</td>
      <td class="metric">${percent(run.exactMatch)}</td><td class="metric">${percent(run.syntaxValid)}</td>
      <td class="metric">${number(run.generationEps)}</td><td class="metric">${number(run.generationDuration)}</td>
      <td class="metric">${number(run.evaluationDuration)}</td><td class="metric">${number(run.gpuPeak, 0)}</td>
      <td class="metric">${number(run.energy)}</td><td class="metric">${percent(run.truncationRate)}</td>
    </tr>`).join("");
    empty.hidden = filtered.length > 0;
    const accuracies = filtered.map(run => run.executionAccuracy).filter(Number.isFinite);
    summary.innerHTML = filtered.length ? [
      ["Runs chargés", filtered.length],
      ["Meilleure execution accuracy", accuracies.length ? percent(Math.max(...accuracies)) : "—"],
      ["Modèles", new Set(filtered.map(run => run.model)).size]
    ].map(([label, value]) => `<div class="stat"><span>${label}</span><strong>${value}</strong></div>`).join("") : "";
  }
  function escapeHtml(value) { const element = document.createElement("span"); element.textContent = value ?? "—"; return element.innerHTML; }
  function importFiles(files) {
    const pairs = new Map();
    const importId = `import-${Date.now()}`;
    [...files].filter(file => file.name === "generation_manifest.json" || file.name === "metrics.json").forEach(file => {
      const key = file.webkitRelativePath ? file.webkitRelativePath.replace(/[/\\][^/\\]+$/, "") : importId;
      const entry = pairs.get(key) || { label: key };
      const reader = new FileReader();
      reader.onload = () => { try { entry[file.name === "metrics.json" ? "metrics" : "manifest"] = JSON.parse(reader.result); pairs.set(key, entry); refreshImports(pairs); } catch { alert(`JSON invalide : ${file.name}`); } };
      reader.readAsText(file);
    });
  }
  function refreshImports(pairs) { state.runs = [...pairs.values()].filter(item => item.manifest || item.metrics).map(item => makeRun(item.manifest, item.metrics, item.label)); render(); }
  document.querySelector("#file-picker").addEventListener("change", event => importFiles(event.target.files));
  document.querySelector("#folder-picker").addEventListener("change", event => importFiles(event.target.files));
  document.querySelector("#clear-button").addEventListener("click", () => { state.runs = []; render(); });
  document.querySelector("#search").addEventListener("input", event => { state.query = event.target.value.toLowerCase(); render(); });
  document.querySelectorAll("th[data-sort]").forEach(header => header.addEventListener("click", () => { const key = header.dataset.sort; state.sort.direction = state.sort.key === key && state.sort.direction === "desc" ? "asc" : "desc"; state.sort.key = key; render(); }));
  const dropZone = document.querySelector("#drop-zone");
  ["dragenter", "dragover"].forEach(type => dropZone.addEventListener(type, event => { event.preventDefault(); dropZone.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach(type => dropZone.addEventListener(type, event => { event.preventDefault(); dropZone.classList.remove("is-over"); }));
  dropZone.addEventListener("drop", event => importFiles(event.dataTransfer.files));
  state.runs = [...(window.initialRuns || [])];
  render();
})();
