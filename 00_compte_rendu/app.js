const datasets = [
  ["Apprentissage", "train", 8651, 20607463, [637.2, 1432, 1952, 2769, 13.85, 2.91, 0.95]],
  ["Apprentissage", "validation", 1034, 1839329, [482.8, 1070, 1135, 1159, 7.54, 0, 0]],
  ["Évaluation", "test", 2147, 3914858, [456.4, 1248, 1306, 1409, 6.89, 0, 0]],
];

const runs = [
  ["Qwen 0.8B", "Zero-shot", "11,78 %", "1,12 %", "68,70 %", "95,53 %", 672, 96, 2],
  ["Qwen 0.8B", "Few-shot · 2", "7,27 %", "2,42 %", "59,48 %", "96,23 %", 870, 81, 0],
  ["Qwen 0.8B", "Few-shot · 4", "8,29 %", "3,17 %", "55,89 %", "94,97 %", 947, 108, 0],
  ["Qwen 2B", "Zero-shot", "29,48 %", "7,55 %", "85,05 %", "97,72 %", 321, 49, 0],
  ["Qwen 2B", "Few-shot · 2", "26,36 %", "6,85 %", "80,11 %", "97,95 %", 427, 44, 1],
  ["Qwen 2B", "Few-shot · 4", "25,24 %", "6,52 %", "81,28 %", "98,04 %", 402, 42, 1],
  ["Qwen 4B", "Zero-shot", "54,26 %", "12,90 %", "94,41 %", "99,67 %", 120, 7, 0],
  ["Qwen 4B", "Few-shot · 2", "54,08 %", "17,93 %", "93,67 %", "99,49 %", 136, 11, 0],
  ["Qwen 4B", "Few-shot · 4", "57,76 %", "21,66 %", "94,64 %", "99,67 %", 115, 7, 0],
  ["Qwen 9B", "Zero-shot", "63,06 %", "16,21 %", "97,21 %", "99,49 %", 60, 11, 0],
  ["Qwen 9B", "Few-shot · 2", "63,81 %", "20,77 %", "95,90 %", "99,35 %", 88, 14, 0],
  ["Qwen 9B", "Few-shot · 4", "63,11 %", "23,06 %", "96,13 %", "99,39 %", 83, 13, 0],
  ["GPT-5.6 Luna", "Zero-shot", "67,77 %", "16,07 %", "99,12 %", "99,21 %", 19, 17, 0],
  ["GPT-5.6 Luna", "Few-shot · 2", "70,00 %", "25,52 %", "99,16 %", "99,30 %", 18, 15, 0],
  ["GPT-5.6 Luna", "Few-shot · 4", "69,82 %", "27,01 %", "99,21 %", "99,30 %", 17, 15, 0],
];

// Premier enregistrement réellement présent dans chacun des fichiers affichés.
const rawExamples = {
  Train: { messages: [
    { role: "system", content: "Tu génères une requête SQL SQLite à partir d'une question en français et du schéma fourni.\nRetourne uniquement la requête SQL valide, sans explication ni balise Markdown." },
    { role: "user", content: "{\"question\": \"Combien de chefs de département ont plus de 56 ans ?\", \"schema\": \"CREATE TABLE \\\"department\\\" (\\n\\\"Department_ID\\\" int,\\n\\\"Name\\\" text,\\n\\\"Creation\\\" text,\\n\\\"Ranking\\\" int,\\n\\\"Budget_in_Billions\\\" real,\\n\\\"Num_Employees\\\" real,\\nPRIMARY KEY (\\\"Department_ID\\\")\\n);\\n\\nCREATE TABLE \\\"head\\\" (\\n\\\"head_ID\\\" int,\\n\\\"name\\\" text,\\n\\\"born_state\\\" text,\\n\\\"age\\\" real,\\nPRIMARY KEY (\\\"head_ID\\\")\\n);\\n\\nCREATE TABLE \\\"management\\\" (\\n\\\"department_ID\\\" int,\\n\\\"head_ID\\\" int,\\n\\\"temporary_acting\\\" text,\\nPRIMARY KEY (\\\"Department_ID\\\",\\\"head_ID\\\"),\\nFOREIGN KEY (\\\"Department_ID\\\") REFERENCES `department`(\\\"Department_ID\\\"),\\nFOREIGN KEY (\\\"head_ID\\\") REFERENCES `head`(\\\"head_ID\\\")\\n);\"}" },
    { role: "assistant", content: "SELECT count(*) FROM head WHERE age > 56" },
  ] },
  Validation: { messages: [
    { role: "system", content: "Tu génères une requête SQL SQLite à partir d'une question en français et du schéma fourni.\nRetourne uniquement la requête SQL valide, sans explication ni balise Markdown." },
    { role: "user", content: "{\"question\": \"Combien de chanteurs avons-nous ?\", \"schema\": \"CREATE TABLE \\\"concert\\\" (\\n\\\"concert_ID\\\" int,\\n\\\"concert_Name\\\" text,\\n\\\"Theme\\\" text,\\n\\\"Stadium_ID\\\" text,\\n\\\"Year\\\" text,\\nPRIMARY KEY (\\\"concert_ID\\\"),\\nFOREIGN KEY (\\\"Stadium_ID\\\") REFERENCES \\\"stadium\\\"(\\\"Stadium_ID\\\")\\n);\\n\\nCREATE TABLE \\\"singer\\\" (\\n\\\"Singer_ID\\\" int,\\n\\\"Name\\\" text,\\n\\\"Country\\\" text,\\n\\\"Song_Name\\\" text,\\n\\\"Song_release_year\\\" text,\\n\\\"Age\\\" int,\\n\\\"Is_male\\\" bool,\\nPRIMARY KEY (\\\"Singer_ID\\\")\\n);\\n\\nCREATE TABLE \\\"singer_in_concert\\\" (\\n\\\"concert_ID\\\" int,\\n\\\"Singer_ID\\\" text,\\nPRIMARY KEY (\\\"concert_ID\\\",\\\"Singer_ID\\\"),\\nFOREIGN KEY (\\\"concert_ID\\\") REFERENCES \\\"concert\\\"(\\\"concert_ID\\\"),\\nFOREIGN KEY (\\\"Singer_ID\\\") REFERENCES \\\"singer\\\"(\\\"Singer_ID\\\")\\n);\\n\\nCREATE TABLE \\\"stadium\\\" (\\n\\\"Stadium_ID\\\" int,\\n\\\"Location\\\" text,\\n\\\"Name\\\" text,\\n\\\"Capacity\\\" int,\\n\\\"Highest\\\" int,\\n\\\"Lowest\\\" int,\\n\\\"Average\\\" int,\\nPRIMARY KEY (\\\"Stadium_ID\\\")\\n);\"}" },
    { role: "assistant", content: "SELECT count(*) FROM singer" },
  ] },
  Test: { id: "test:0", db_id: "soccer_3", messages: [
    { role: "system", content: "Tu génères une requête SQL SQLite à partir d'une question en français et du schéma fourni.\nRetourne uniquement la requête SQL valide, sans explication ni balise Markdown." },
    { role: "user", content: "{\"question\": \"Combien y a-t-il de clubs ?\", \"schema\": \"CREATE TABLE \\\"club\\\" (\\n\\\"Club_ID\\\" int,\\n\\\"Name\\\" text,\\n\\\"Manager\\\" text,\\n\\\"Captain\\\" text,\\n\\\"Manufacturer\\\" text,\\n\\\"Sponsor\\\" text,\\nPRIMARY KEY (\\\"Club_ID\\\")\\n);\\n\\nCREATE TABLE \\\"player\\\" (\\n\\\"Player_ID\\\" real,\\n\\\"Name\\\" text,\\n\\\"Country\\\" text,\\n\\\"Earnings\\\" real,\\n\\\"Events_number\\\" int,\\n\\\"Wins_count\\\" int,\\n\\\"Club_ID\\\" int,\\nPRIMARY KEY (\\\"Player_ID\\\"),\\nFOREIGN KEY (\\\"Club_ID\\\") REFERENCES \\\"club\\\"(\\\"Club_ID\\\")\\n);\"}" },
  ] },
};

// Exemple train le plus court, tokenisé avec le tokenizer officiel Qwen local.
const smallestTokenizedExample = [[248045,"<|im_start|>"],[8678,"system"],[198,"↵"],[51193,"Tu"],[74676,"␠gén"],[29448,"ères"],[5992,"␠une"],[5710,"␠requ"],[34800,"ête"],[7622,"␠SQL"],[27413,"␠SQLite"],[3656,"␠à"],[29536,"␠partir"],[293,"␠d"],[29027,"'une"],[3296,"␠question"],[644,"␠en"],[52543,"␠français"],[1778,"␠et"],[3717,"␠du"],[5521,"␠sch"],[161201,"éma"],[209773,"␠fourni"],[13,"."],[198,"↵"],[11678,"Ret"],[405,"our"],[785,"ne"],[181162,"␠uniquement"],[1147,"␠la"],[5710,"␠requ"],[34800,"ête"],[7622,"␠SQL"],[95340,"␠valide"],[11,","],[15055,"␠sans"],[1295,"␠exp"],[1633,"lication"],[12421,"␠ni"],[9354,"␠bal"],[1029,"ise"],[70703,"␠Markdown"],[13,"."],[248046,"<|im_end|>"],[198,"↵"],[248045,"<|im_start|>"],[846,"user"],[198,"↵"],[4754,"{\""],[7593,"question"],[763,"\":"],[328,"␠\""],[2114,"Qu"],[300,"el"],[1725,"␠est"],[324,"␠l"],[6,"'"],[183584,"âge"],[86919,"␠moyen"],[401,"␠de"],[38332,"␠toutes"],[3424,"␠les"],[45894,"␠personnes"],[40907,"␠?\","],[328,"␠\""],[16838,"schema"],[763,"\":"],[328,"␠\""],[21898,"CREATE"],[13951,"␠TABLE"],[7126,"␠Person"],[19185,"␠(\\"],[77,"n"],[220,"␠"],[803,"␠name"],[31865,"␠varchar"],[7,"("],[17,"2"],[15,"0"],[8,")"],[36227,"␠PRIMARY"],[11671,"␠KEY"],[25427,",\\"],[77,"n"],[220,"␠"],[4092,"␠age"],[29392,"␠INTEGER"],[25427,",\\"],[77,"n"],[220,"␠"],[3177,"␠city"],[15299,"␠TEXT"],[25427,",\\"],[77,"n"],[220,"␠"],[9533,"␠gender"],[15299,"␠TEXT"],[25427,",\\"],[77,"n"],[220,"␠"],[2531,"␠job"],[15299,"␠TEXT"],[1639,"\\n"],[42913,");\\"],[77,"n"],[1639,"\\n"],[21898,"CREATE"],[13951,"␠TABLE"],[7126,"␠Person"],[40261,"Friend"],[19185,"␠(\\"],[77,"n"],[220,"␠"],[803,"␠name"],[31865,"␠varchar"],[7,"("],[17,"2"],[15,"0"],[681,"),"],[59,"\\"],[77,"n"],[220,"␠"],[4099,"␠friend"],[31865,"␠varchar"],[7,"("],[17,"2"],[15,"0"],[681,"),"],[59,"\\"],[77,"n"],[220,"␠"],[1007,"␠year"],[29392,"␠INTEGER"],[25427,",\\"],[77,"n"],[220,"␠"],[78552,"␠FOREIGN"],[11671,"␠KEY"],[318,"␠("],[591,"name"],[8,")"],[75965,"␠REFERENCES"],[7126,"␠Person"],[3052,"(name"],[681,"),"],[59,"\\"],[77,"n"],[220,"␠"],[78552,"␠FOREIGN"],[11671,"␠KEY"],[318,"␠("],[10385,"friend"],[8,")"],[75965,"␠REFERENCES"],[7126,"␠Person"],[3052,"(name"],[10383,")\\"],[77,"n"],[1174,");"],[8934,"\"}"],[248046,"<|im_end|>"],[198,"↵"],[248045,"<|im_start|>"],[74455,"assistant"],[198,"↵"],[248068,"<think>"],[271,"↵↵"],[248069,"</think>"],[271,"↵↵"],[4703,"SELECT"],[19110,"␠avg"],[64344,"(age"],[8,")"],[4154,"␠FROM"],[7126,"␠Person"],[248046,"<|im_end|>"]];

const smallestExample = {
  system: "Tu génères une requête SQL SQLite à partir d'une question en français et du schéma fourni.\nRetourne uniquement la requête SQL valide, sans explication ni balise Markdown.",
  user: "{\"question\": \"Quel est l'âge moyen de toutes les personnes ?\", \"schema\": \"CREATE TABLE Person (\\n  name varchar(20) PRIMARY KEY,\\n  age INTEGER,\\n  city TEXT,\\n  gender TEXT,\\n  job TEXT\\n);\\n\\nCREATE TABLE PersonFriend (\\n  name varchar(20),\\n  friend varchar(20),\\n  year INTEGER,\\n  FOREIGN KEY (name) REFERENCES Person(name),\\n  FOREIGN KEY (friend) REFERENCES Person(name)\\n);\"}",
  sql: "SELECT avg(age) FROM Person",
};

const nf = new Intl.NumberFormat("fr-FR");
const display = value => value === 0 ? "—" : nf.format(value);
const size = bytes => `${(bytes / 1024 / 1024).toLocaleString("fr-FR", { maximumFractionDigits: 2 })} Mo`;
const colorJson = value => JSON.stringify(value, null, 2).replace(/("(?:\\.|[^"\\])*")(?=\s*:)|("(?:\\.|[^"\\])*")|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?/g, (match, key) => {
  const kind = key ? "json-key" : match[0] === '"' ? "json-string" : "json-value";
  return `<span class="${kind}">${match.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</span>`;
});
const chatPrompt = value => `${value.messages.filter(message => message.role !== "assistant").map(message => `<|im_start|>${message.role}\n${message.content}<|im_end|>`).join("\n")}\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
const colorChat = value => chatPrompt(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/(&lt;\|im_(?:start|end)\|&gt;)/g, '<span class="chat-token">$1</span>').replace(/\b(system|user|assistant)\n/g, '<span class="chat-role">$1</span>\n');
const predictionTarget = value => {
  const answer = value.messages.find(message => message.role === "assistant")?.content;
  return answer ? `${answer}\n<|im_end|>` : "Le modèle doit prédire la requête SQL (puis <|im_end|>), qui est ensuite normalisée et testée directement sur les bases SQLite.";
};
const escapeHtml = value => value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const structureTokens = new Set(["<|im_start|>", "<|im_end|>", "<think>", "</think>", "system", "user", "assistant"]);
const predictionStart = smallestTokenizedExample.findIndex(([id]) => id === 4703);
const sequencePart = (value, kind) => `<span class="sequence-part ${kind}">${escapeHtml(value)}</span>`;
document.querySelector("#complete-sequence").innerHTML = [
  sequencePart("<|im_start|>", "token-structural"), sequencePart("system", "token-structural"), "\n", sequencePart(smallestExample.system, "token-content"), "\n", sequencePart("<|im_end|>", "token-structural"), "\n",
  sequencePart("<|im_start|>", "token-structural"), sequencePart("user", "token-structural"), "\n", sequencePart(smallestExample.user, "token-content"), "\n", sequencePart("<|im_end|>", "token-structural"), "\n",
  sequencePart("<|im_start|>", "token-structural"), sequencePart("assistant", "token-structural"), "\n", sequencePart("<think>\n\n</think>", "token-structural"), "\n", sequencePart(smallestExample.sql, "token-prediction"), "\n", sequencePart("<|im_end|>", "token-prediction"),
].join("");
document.querySelector("#tokenized-example").innerHTML = smallestTokenizedExample.map(([id, token], tokenIndex) => {
  const structural = structureTokens.has(token);
  const breakAfter = structural ? "<br>" : "";
  const kind = tokenIndex >= predictionStart ? "token-prediction" : structural ? "token-structural" : "token-content";
  return `<span class="token-chip ${kind}">${id} · ${escapeHtml(token)}</span>${breakAfter}`;
}).join("");

document.querySelector("#dataset-summary").innerHTML = [
  ["Train", "8 651", "146", "train_spider + train_others", "01_data/06_training_dataset/03_production/train.jsonl", "book_2", "Combien y a-t-il de livres ?", "book(Book_ID, Title, Issues, Writer)", "SELECT count(*) FROM book"],
  ["Validation", "1 034", "20", "split dev", "01_data/06_training_dataset/03_production/validation.jsonl", "course_teach", "Combien y a-t-il de professeurs ?", "teacher(Teacher_ID, Name, Age, Hometown)", "SELECT count(*) FROM teacher"],
  ["Test", "2 147", "40", "évaluation finale", "01_data/07_evaluation_dataset/03_production/test_inputs.jsonl", "soccer_3", "Combien y a-t-il de clubs ?", "club(Club_ID, Name, Manager, Captain, …)", "SELECT count(*) FROM club"],
].map(([split, examples, databases, detail, path]) => `<article><span>${split}</span><div class="card-metrics"><div class="metric-line metric-examples"><strong>${examples}</strong><small>exemples</small></div><div class="metric-line metric-databases"><strong>${databases}</strong><small>bases</small></div></div><small>${detail}</small><code class="file-path">${path}</code><div class="dataset-example"><button type="button" aria-label="Afficher le premier exemple JSONL de ${split}" aria-expanded="false">Exemple JSONL</button><div class="example-tooltip" role="dialog" aria-label="Premier exemple JSONL ${split}"><button type="button" class="close-example" aria-label="Fermer l’exemple">×</button><div class="popup-grid"><section><h3>JSONL brut</h3><pre class="json-code">${colorJson(rawExamples[split])}</pre></section><section><h3>Après le chat template Qwen · mode non-thinking</h3><pre class="json-code chat-code">${colorChat(rawExamples[split])}</pre></section><section class="prediction-target${split === "Test" ? " test-prediction-target" : ""}"><h3>${split === "Test" ? "Inférence et évaluation" : "Suite à prédire"}</h3><pre class="json-code">${predictionTarget(rawExamples[split])}</pre></section></div></div></div></article>`).join("");
document.querySelectorAll(".dataset-example button").forEach(button => button.addEventListener("click", () => {
  if (button.classList.contains("close-example")) return;
  const example = button.parentElement;
  const isOpen = example.classList.toggle("is-open");
  button.setAttribute("aria-expanded", String(isOpen));
}));
document.querySelectorAll(".close-example").forEach(button => button.addEventListener("click", () => {
  const example = button.closest(".dataset-example");
  example.classList.remove("is-open");
  example.querySelector("button:not(.close-example)").setAttribute("aria-expanded", "false");
}));
document.addEventListener("keydown", event => {
  if (event.key !== "Escape") return;
  document.querySelectorAll(".dataset-example.is-open").forEach(example => {
    example.classList.remove("is-open");
    example.querySelector("button:not(.close-example)").setAttribute("aria-expanded", "false");
  });
});
document.querySelector("#dataset-rows").innerHTML = datasets.map(([group, split, records, bytes, sequence]) => `<tr><td><b>${split}</b><small>${group}</small></td><td>${nf.format(records)}</td><td>${nf.format(sequence[0])}</td><td>${nf.format(sequence[1])}</td><td>${nf.format(sequence[2])}</td><td>${nf.format(sequence[3])}</td><td>${nf.format(sequence[4])}&nbsp;%</td><td>${nf.format(sequence[5])}&nbsp;%</td><td>${nf.format(sequence[6])}&nbsp;%</td><td>${size(bytes)}</td></tr>`).join("");

document.querySelector("#evaluation-head").innerHTML = `<tr><th>Modèle</th><th>Configuration</th><th>Execution accuracy</th><th>Exact match</th><th>SQL exécutable</th><th>SQL valide</th><th>Erreurs d’exécution</th><th>SQL invalides</th><th>Timeouts</th></tr>`;
const qwenRuns = runs.filter(([model]) => model !== "GPT-5.6 Luna");
const metricDirections = [1, 1, 1, 1, -1, -1, -1];
const metricValue = value => Number(String(value).replace("%", "").replace(",", ".").trim());
const bestQwenMetrics = metricDirections.map((direction, metricIndex) => Math[direction === 1 ? "max" : "min"](...qwenRuns.map(run => metricValue(run[metricIndex + 2]))));
const worstQwenMetrics = metricDirections.map((direction, metricIndex) => Math[direction === 1 ? "min" : "max"](...qwenRuns.map(run => metricValue(run[metricIndex + 2]))));
document.querySelector("#evaluation-rows").innerHTML = runs.map((run, index) => {
  const [model, config, execution, exact, executable, valid, errors, invalid, timeouts] = run;
  const luna = index >= 12 ? " luna" : "";
  const modelClass = ` model-${model.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
  const metrics = [execution, exact, executable, valid, errors, invalid, timeouts];
  const cells = metrics.map((value, metricIndex) => {
    const metric = metricValue(value);
    const className = model === "GPT-5.6 Luna" ? "" : metric === bestQwenMetrics[metricIndex] ? "best" : metric === worstQwenMetrics[metricIndex] ? "worst" : "";
    return `<td class="${className}">${typeof value === "number" ? nf.format(value) : value}</td>`;
  }).join("");
  return `<tr class="${luna}${modelClass}"><td>${model}</td><td>${config}</td>${cells}</tr>`;
}).join("");
