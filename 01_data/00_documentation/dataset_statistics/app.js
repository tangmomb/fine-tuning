const stats = window.DATASET_STATS;
const nf = new Intl.NumberFormat("fr-FR");
const display = value => value === 0 ? "—" : nf.format(value);
const size = bytes => `${(bytes / 1024 / 1024).toLocaleString("fr-FR", { maximumFractionDigits: 2 })} Mo`;
const all = stats.files;
const training = all.filter(file => file.group === "Apprentissage");
const evaluation = all.filter(file => file.group === "Évaluation");
const train = training.find(file => file.split === "train");
const validation = training.find(file => file.split === "validation");

document.querySelector("#summary").innerHTML = [
  ["Fichiers analysés", all.length],
  ["Cas d’apprentissage", train.records],
  ["Cas de validation", validation.records],
  ["Cas d’évaluation", evaluation.find(file => file.split === "test_inputs").records],
].map(([label, value]) => `<article><strong>${value}</strong><span>${label}</span></article>`).join("");

document.querySelector("#rows").innerHTML = all.map(file => `<tr><td><b>${file.split}</b><small>${file.group}</small></td><td>${nf.format(file.records)}</td><td>${display(file.prompt.mean)}</td><td>${display(file.prompt.max)}</td><td>${display(file.question.mean)}</td><td>${display(file.schema.mean)}</td><td>${display(file.response.mean)}</td><td>${display(file.response.max)}</td><td>${size(file.bytes)}</td></tr>`).join("");
