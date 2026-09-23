(() => {
  const state = { runs: [] };
  const head = document.querySelector("#runs-head");
  const body = document.querySelector("#runs-body");
  const empty = document.querySelector("#empty-state");
  const summary = document.querySelector("#summary");
  const totalExamples = 2147;
  const percent = value => Number.isFinite(value) ? `${(value * 100).toLocaleString("fr-FR", { maximumFractionDigits: 2 })} %` : "—";
  const count = value => Number.isFinite(value) ? value.toLocaleString("fr-FR") : "—";
  const modelSize = model => Number.parseFloat((model || "").match(/Qwen3\.5-([\d.]+)B/)?.[1]) || Number.POSITIVE_INFINITY;
  const runOrder = run => run.mode === "zero-shot" ? 0 : Number.isFinite(run.fewShotK) ? run.fewShotK : Number.POSITIVE_INFINITY;
  const compareRuns = (left, right) => modelSize(left.model) - modelSize(right.model)
    || runOrder(left) - runOrder(right)
    || left.model.localeCompare(right.model, "fr");

  function makeRun(manifest, metrics, source = "") {
    const evaluation = metrics?.evaluation || {};
    const accuracy = evaluation.accuracy || {};
    const modelPath = manifest?.model?.path || metrics?.source?.generation?.model_path || "Modèle inconnu";
    return {
      label: source || manifest?.artifact?.created_at_utc || metrics?.artifact?.created_at_utc || "Run importé",
      model: modelPath.split(/[\\/]/).filter(Boolean).pop(),
      mode: manifest?.configuration?.mode || metrics?.configuration?.mode || "—",
      fewShotK: manifest?.configuration?.few_shot_k ?? metrics?.configuration?.few_shot_k,
      examples: evaluation.examples,
      executionAccuracy: accuracy.execution?.rate,
      correctCount: accuracy.execution?.correct_count,
      executionSuccess: accuracy.execution?.success_rate,
      successCount: accuracy.execution?.success_count,
      exactMatch: accuracy.exact_match?.rate,
      exactCount: accuracy.exact_match?.count,
      syntaxValid: accuracy.syntax?.valid_rate,
      validCount: accuracy.syntax?.valid_count,
      invalidCount: accuracy.syntax?.invalid_count,
      executionErrors: evaluation.errors?.execution_error_count,
      timeouts: evaluation.errors?.timeout_count,
      emptyPredictions: evaluation.errors?.empty_prediction_count,
    };
  }

  const metricRows = [
    { label: "Execution accuracy", help: "Part des requêtes dont le résultat SQLite est identique à celui du SQL de référence.", value: run => percent(run.executionAccuracy), detail: run => `${count(run.correctCount)} / ${count(run.examples || totalExamples)}`, best: "max", key: "executionAccuracy" },
    { label: "Exact match", help: "Part des requêtes dont le SQL prédit correspond exactement au SQL de référence après normalisation des espaces et de la casse.", value: run => percent(run.exactMatch), detail: run => `${count(run.exactCount)} / ${count(run.examples || totalExamples)}`, best: "max", key: "exactMatch" },
    { label: "SQL exécutable", help: "Part des prédictions qui s’exécutent sans erreur dans SQLite, quel que soit leur résultat.", value: run => percent(run.executionSuccess), detail: run => `${count(run.successCount)} / ${count(run.examples || totalExamples)}`, best: "max", key: "executionSuccess" },
    { label: "SQL syntaxiquement valide", help: "Part des requêtes acceptées par le parseur SQLite ; une requête peut être valide mais échouer à l’exécution si elle référence un élément inexistant.", value: run => percent(run.syntaxValid), detail: run => `${count(run.validCount)} / ${count(run.examples || totalExamples)}`, best: "max", key: "syntaxValid" },
    { label: "Erreurs d’exécution", help: "Nombre de prédictions que SQLite n’a pas pu exécuter : erreurs de schéma, de type, de requête ou timeout inclus.", value: run => count(run.executionErrors), best: "min", key: "executionErrors" },
    { label: "SQL invalides", help: "Nombre de prédictions rejetées pour une erreur de syntaxe SQL.", value: run => count(run.invalidCount), best: "min", key: "invalidCount" },
    { label: "Timeouts", help: "Nombre de requêtes interrompues après avoir dépassé la limite d’exécution de 5 secondes.", value: run => count(run.timeouts), best: "min", key: "timeouts" },
    { label: "Prédictions vides", help: "Nombre de sorties sans requête SQL exploitable après nettoyage de la réponse du modèle.", value: run => count(run.emptyPredictions), best: "min", key: "emptyPredictions" },
  ];

  function render() {
    const runs = [...state.runs].sort(compareRuns);
    const models = [...new Set(runs.map(run => run.model))];
    head.innerHTML = `<tr><th>Métrique SQL</th>${runs.map(run => {
      const shots = Number.isFinite(run.fewShotK) ? run.fewShotK : run.mode === "zero-shot" ? 0 : "—";
      const tone = models.indexOf(run.model) % 4;
      return `<th class="model-head model-tone-${tone}"><strong>${escapeHtml(run.model)}</strong><div class="run-meta"><span class="run-card"><small>Mode</small>${escapeHtml(run.mode)}</span><span class="run-card"><small>Shots</small>${shots}</span></div></th>`;
    }).join("")}</tr>`;
    body.innerHTML = metricRows.map(metric => {
      const values = runs.map(run => run[metric.key]).filter(Number.isFinite);
      const best = values.length ? (metric.best === "min" ? Math.min(...values) : Math.max(...values)) : undefined;
      return `<tr><td><span class="metric-name">${metric.label}<button class="metric-help" type="button" aria-label="Explication : ${escapeHtml(metric.help)}" data-tooltip="${escapeHtml(metric.help)}">?</button></span></td>${runs.map(run => `<td class="${run[metric.key] === best ? "best" : ""}">${metric.value(run)}${metric.detail ? `<span class="cell-detail">${metric.detail(run)}</span>` : ""}</td>`).join("")}</tr>`;
    }).join("");
    empty.hidden = runs.length > 0;
    const accuracies = runs.map(run => run.executionAccuracy).filter(Number.isFinite);
    summary.innerHTML = runs.length ? [
      ["Runs comparés", runs.length],
      ["Meilleure execution accuracy", accuracies.length ? percent(Math.max(...accuracies)) : "—"],
      ["Jeu de test", `${count(runs[0]?.examples || totalExamples)} requêtes`]
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
      reader.onload = () => { try { entry[file.name === "metrics.json" ? "metrics" : "manifest"] = JSON.parse(reader.result); pairs.set(key, entry); state.runs = [...pairs.values()].map(item => makeRun(item.manifest, item.metrics, item.label)); render(); } catch { alert(`JSON invalide : ${file.name}`); } };
      reader.readAsText(file);
    });
  }
  document.querySelector("#file-picker").addEventListener("change", event => importFiles(event.target.files));
  document.querySelector("#folder-picker").addEventListener("change", event => importFiles(event.target.files));
  document.querySelector("#clear-button").addEventListener("click", () => { state.runs = []; render(); });
  const dropZone = document.querySelector("#drop-zone");
  ["dragenter", "dragover"].forEach(type => dropZone.addEventListener(type, event => { event.preventDefault(); dropZone.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach(type => dropZone.addEventListener(type, event => { event.preventDefault(); dropZone.classList.remove("is-over"); }));
  dropZone.addEventListener("drop", event => importFiles(event.dataTransfer.files));
  state.runs = [...(window.initialRuns || [])];
  render();
})();
