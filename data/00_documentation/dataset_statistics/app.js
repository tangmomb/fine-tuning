const stats = window.DATASET_STATS;
const nf = new Intl.NumberFormat("fr-FR");
const display = value => value === 0 ? "—" : nf.format(value);
const size = bytes => `${(bytes / 1024 / 1024).toLocaleString("fr-FR", { maximumFractionDigits: 2 })} Mo`;
const all = stats.files;
const training = all.filter(file => file.group === "Apprentissage");
const evaluation = all.filter(file => file.group === "Évaluation");

document.querySelector("#method").textContent = `Mesures calculées le ${new Date(stats.generatedAt).toLocaleString("fr-FR", { dateStyle: "long", timeStyle: "short" })} avec ${stats.tokenizer}. Le prompt inclut le template de conversation Qwen (system + user + jeton de génération).`;
document.querySelector("#summary").innerHTML = [
  ["Fichiers analysés", all.length],
  ["Cas d’apprentissage", training.reduce((sum, file) => sum + file.records, 0)],
  ["Cas d’évaluation", evaluation.find(file => file.split === "test_inputs").records],
  ["Tokens SQL d’apprentissage", nf.format(training.reduce((sum, file) => sum + file.response.total, 0))],
].map(([label, value]) => `<article><strong>${value}</strong><span>${label}</span></article>`).join("");

const maxPrompt = Math.max(...all.map(file => file.prompt.max), 1);
const maxResponse = Math.max(...all.map(file => file.response.max), 1);
document.querySelector("#cards").innerHTML = all.map(file => {
  const promptWidth = file.prompt.max / maxPrompt * 100;
  const responseWidth = file.response.max / maxResponse * 100;
  return `<article class="card"><div class="card-title"><div><span class="group ${file.group === "Évaluation" ? "eval" : ""}">${file.group}</span><h3>${file.split}</h3></div><strong>${nf.format(file.records)} <small>cas</small></strong></div><p class="path">${file.path}</p><div class="metric"><span>Prompt</span><b>${display(file.prompt.mean)} moy. · ${display(file.prompt.max)} max.</b><i><em class="prompt" style="width:${promptWidth}%"></em></i></div><div class="metric"><span>SQL cible</span><b>${display(file.response.mean)} moy. · ${display(file.response.max)} max.</b><i><em class="response" style="width:${responseWidth}%"></em></i></div></article>`;
}).join("");
document.querySelector("#rows").innerHTML = all.map(file => `<tr><td><b>${file.split}</b><small>${file.group}</small></td><td>${nf.format(file.records)}</td><td>${display(file.prompt.mean)}</td><td>${display(file.prompt.max)}</td><td>${display(file.question.mean)}</td><td>${display(file.schema.mean)}</td><td>${display(file.response.mean)}</td><td>${display(file.response.max)}</td><td>${size(file.bytes)}</td></tr>`).join("");
