"""Prépare des requêtes de génération text-to-SQL diversifiées, guidées par des exemples Spider.

Différence avec 03_03_more_data : aucun gabarit SQL imposé par difficulté. Chaque prompt
montre des exemples réels du train Spider (questions françaises, autres bases), choisis
pour couvrir des structures différentes, et demande des analogues structurels sur la base
cible. La distribution des structures suit donc celle de Spider (jointures, ORDER BY/LIMIT,
opérations ensemblistes, sous-requêtes, HAVING...). Des lignes réelles de chaque table sont
fournies pour que les valeurs littérales existent dans la base.

Le fichier produit garde le format OpenAI Batch (/v1/responses). `--run-sync` exécute les
requêtes directement (pilote) et écrit les réponses au même format que la sortie Batch.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import error, request

HERE = Path(__file__).resolve().parent
FINE_TUNING = HERE.parent
sys.path.insert(0, str(FINE_TUNING / "03_03_more_data" / "03_03_02_batch"))
from create_batch import SCHEMA, key, largest_remainder, post, upload  # noqa: E402

PREPARE = FINE_TUNING / "03_03_more_data" / "03_03_01_prepare"
SPIDER_TRAIN = FINE_TUNING / "03_02_train_data_scaling" / "train_with_hardness.jsonl"
DEFAULT_MODEL = "gpt-5.6-luna"
MAX_OUTPUT_TOKENS = 64000
SAMPLE_ROWS = 3
MAX_VALUE_CHARS = 40

SYSTEM = "Tu produis des données text-to-SQL françaises fiables et variées, dans le style du jeu Spider."
SQL_RULES = """Contraintes SQL (le SQL sera parsé par l'évaluateur officiel Spider, une requête non reconnue sera rejetée) :
- Autorisé : SELECT, DISTINCT, FROM, JOIN ... ON, WHERE, GROUP BY, HAVING, ORDER BY ... ASC/DESC, LIMIT, COUNT/SUM/AVG/MIN/MAX, =, !=, <, >, <=, >=, BETWEEN, LIKE, IN / NOT IN (sous-requête), sous-requête comparée (col > (SELECT avg(col) FROM t)), UNION, INTERSECT, EXCEPT.
- Alias uniquement pour les tables, au format Spider : FROM a AS T1 JOIN b AS T2 ON T1.x = T2.y. Jamais d'alias de colonne ou d'agrégat.
- Interdit : WITH, LEFT/RIGHT/OUTER JOIN, CASE, CAST, IS NULL, EXISTS, fonctions de fenêtre, IN avec une liste de valeurs, guillemets ou accents graves autour des identifiants, commentaires.
- Chaînes entre apostrophes simples, « différent de » écrit !=."""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


KEYWORDS = {"select", "from", "where", "group", "by", "having", "order", "limit", "join", "on", "and", "or", "not", "in",
            "count", "avg", "sum", "min", "max", "distinct", "desc", "asc", "union", "intersect", "except", "like", "between"}


def skeleton(sql: str) -> str:
    """Structure SQL sans noms ni valeurs, pour mesurer la diversité."""
    s = re.sub(r"'[^']*'|\"[^\"]*\"|\b\d+(?:\.\d+)?\b", "V", sql.lower().rstrip("; "))
    s = re.sub(r"\bas t\d\b|\bt\d\.", "", s)
    return " ".join(w if w in KEYWORDS or not re.match(r"[a-z_]", w) else "x"
                    for w in re.findall(r"[a-z_][a-z_0-9]*|[^\sa-z_]", s))


def structure_class(sql: str) -> str:
    """Classe grossière de structure, pour varier les exemples d'un même prompt."""
    s = sql.lower()
    if re.search(r"\b(union|intersect|except)\b", s):
        return "set_op"
    joins = len(re.findall(r"\bjoin\b", s))
    nested = bool(re.search(r"\(\s*select\b", s))
    parts = ["join2+" if joins >= 2 else "join1" if joins == 1 else "single"]
    if nested:
        parts.append("nested")
    if " having " in s:
        parts.append("having")
    elif " group by " in s:
        parts.append("group")
    if re.search(r"\border by\b", s):
        parts.append("order")
    return "+".join(parts)


def load_exemplars() -> list[dict]:
    exemplars = []
    for row in read_jsonl(SPIDER_TRAIN):
        messages = row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"])
        question = json.loads(messages[1]["content"])["question"]
        sql = messages[2]["content"]
        exemplars.append({"db_id": row["db_id"], "hardness": row["hardness"], "question": question,
                          "sql": sql, "structure": structure_class(sql)})
    return exemplars


def pick_exemplars(pool: list[dict], target_db: str, count: int, rng: random.Random) -> list[dict]:
    """Tire des exemples d'autres bases, proportionnellement à Spider, sans répéter un squelette SQL."""
    candidates = [row for row in pool if row["db_id"] != target_db]
    rng.shuffle(candidates)
    chosen, skeletons = [], set()
    for row in candidates:
        if skeleton(row["sql"]) in skeletons:
            continue
        chosen.append(row)
        skeletons.add(skeleton(row["sql"]))
        if len(chosen) == count:
            break
    return chosen


def sample_rows(database_path: str) -> str:
    """Quelques lignes réelles par table, pour ancrer les valeurs littérales."""
    connection = sqlite3.connect(f"file:{Path(database_path).as_posix()}?mode=ro", uri=True)
    connection.text_factory = lambda value: value.decode("utf-8", errors="replace")
    blocks = []
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for table in tables:
            cursor = connection.execute(f'SELECT * FROM "{table}" LIMIT {SAMPLE_ROWS}')
            columns = [description[0] for description in cursor.description]
            lines = [" | ".join(columns)]
            for values in cursor.fetchall():
                lines.append(" | ".join(str(value)[:MAX_VALUE_CHARS] for value in values))
            blocks.append(f"{table} :\n" + "\n".join(lines))
    finally:
        connection.close()
    return "\n\n".join(blocks)


def build_prompt(seed: dict, exemplars: list[dict], per_exemplar: int) -> str:
    examples = "\n\n".join(
        f"Exemple {index} (base {row['db_id']}, difficulté Spider {row['hardness']}) :\nQuestion : {row['question']}\nSQL : {row['sql']}"
        for index, row in enumerate(exemplars, start=1))
    total = len(exemplars) * per_exemplar
    return f"""Voici {len(exemplars)} exemples tirés du jeu Spider, sur d'autres bases que la base cible.

{examples}

Tâche : pour chaque exemple, écris {per_exemplar} nouvelle(s) paire(s) question/SQL sur la base cible ({seed['db_id']}) qui reprennent la même structure SQL : mêmes clauses, même nombre de tables jointes, mêmes sous-requêtes ou opérations ensemblistes, même usage de GROUP BY / HAVING / ORDER BY / LIMIT. Change les tables, colonnes, valeurs et le sujet de la question. Soit {total} paires au total, dans l'ordre des exemples.

Règles de qualité :
- Si la base cible ne permet pas exactement une structure (pas de clé étrangère adaptée, par exemple), prends la structure la plus proche possible. N'écris jamais de jointure, sous-requête ou condition inutile ou toujours vraie (par exemple WHERE id IN (SELECT id FROM la_même_table)).
- Les jointures suivent les clés étrangères du schéma.
- Les valeurs littérales doivent exister dans la base : appuie-toi sur les lignes d'exemple fournies et reprends exactement leur forme (casse, codes comme 'T' ou 'F', format des dates).
- Chaque requête doit renvoyer au moins une ligne.
- Le SELECT ne contient que ce que la question demande, dans l'ordre où elle le demande.
- La question est formulée comme un vrai utilisateur la poserait, en français naturel : elle ne décrit pas le SQL et n'emploie pas les noms techniques des colonnes, mais elle désigne sans ambiguïté le résultat attendu.
- Toutes les questions sont différentes entre elles.

{SQL_RULES}

SCHÉMA ({seed['db_id']}) :
{seed['schema']}

LIGNES D'EXEMPLE ({SAMPLE_ROWS} premières lignes par table) :
{sample_rows(seed['database_path'])}"""


def build_requests(args: argparse.Namespace) -> list[dict]:
    rng = random.Random(args.seed)
    pool = [row for row in load_exemplars() if not args.hardness or row["hardness"] in args.hardness]
    existing = read_jsonl(PREPARE / "existing_sql_bases.jsonl")
    new = read_jsonl(PREPARE / "new_sql_bases.jsonl")
    targets = []
    if args.pilot:
        targets += [("existing", seed) for seed in rng.sample(existing, args.pilot_existing)]
        targets += [("new", seed) for seed in rng.sample(new, args.pilot_new)]
    else:
        source = existing if args.bases == "existing" else new
        targets = [(args.bases, seed) for seed in source]
    if args.total_prompts and not args.pilot:
        # Même pondération que l'ancien A : au prorata des exemples Spider de chaque base, au moins un prompt.
        weights = {seed["db_id"]: seed.get("source_examples", 1) for _, seed in targets}
        prompts_per_base = {db_id: max(1, count) for db_id, count in largest_remainder(weights, args.total_prompts).items()}
    else:
        prompts_per_base = {seed["db_id"]: args.prompts_per_base for _, seed in targets}
    requests = []
    for kind, seed in targets:
        for chunk in range(1, prompts_per_base[seed["db_id"]] + 1):
            exemplars = pick_exemplars(pool, seed["db_id"], args.exemplars, rng)
            body = {"model": args.model, "input": [{"role": "system", "content": SYSTEM},
                                                   {"role": "user", "content": build_prompt(seed, exemplars, args.per_exemplar)}],
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "text": {"format": {"type": "json_schema", "name": "synthetic_examples", "strict": True, "schema": SCHEMA}}}
            requests.append({"custom_id": f"v2:{kind}:{seed['db_id']}:{chunk}", "method": "POST", "url": "/v1/responses",
                             "body": body, "_exemplars": [{"structure": row["structure"], "hardness": row["hardness"], "sql": row["sql"]} for row in exemplars],
                             "_database_path": seed["database_path"]})
    return requests


def call(body: dict, token: str, retries: int = 3) -> dict:
    payload = json.dumps(body).encode()
    for attempt in range(retries):
        req = request.Request("https://api.openai.com/v1/responses", data=payload, method="POST",
                              headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        try:
            with request.urlopen(req, timeout=900) as response:
                return {"status_code": 200, "body": json.load(response)}
        except error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                return {"status_code": exc.code, "body": json.loads(exc.read() or b"{}")}
        except (error.URLError, TimeoutError):
            pass
        time.sleep(10 * (attempt + 1))
    return {"status_code": 599, "body": {}}


def submit(output: Path, model: str) -> None:
    token = key()
    uploaded = upload(output / "requests.jsonl", token)
    batch = post("https://api.openai.com/v1/batches", {"input_file_id": uploaded["id"], "endpoint": "/v1/responses", "completion_window": "24h",
                                                        "metadata": {"pipeline": "diverse-data-v2", "output": output.name, "model": model}}, token)
    state = {"batch_id": batch["id"], "input_file_id": uploaded["id"], "model": model, "status": batch.get("status"),
             "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (output / "batch_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Batch soumis : {batch['id']} ({batch.get('status')})")


def download(output: Path) -> None:
    token = key()
    state_path = output / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))

    def get(url: str) -> bytes:
        with request.urlopen(request.Request(url, headers={"Authorization": f"Bearer {token}"})) as response:
            return response.read()

    batch = json.loads(get(f"https://api.openai.com/v1/batches/{state['batch_id']}"))
    counts = batch.get("request_counts", {})
    state["status"] = batch.get("status")
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Statut : {batch.get('status')} · {counts.get('completed', 0)}/{counts.get('total', 0)} terminées · {counts.get('failed', 0)} échouées")
    if batch.get("output_file_id"):
        (output / "responses.jsonl").write_bytes(get(f"https://api.openai.com/v1/files/{batch['output_file_id']}/content"))
        print(f"Réponses téléchargées : {output / 'responses.jsonl'}")
    if batch.get("error_file_id"):
        (output / "errors.jsonl").write_bytes(get(f"https://api.openai.com/v1/files/{batch['error_file_id']}/content"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Petit lot de test sur quelques bases existantes et nouvelles.")
    parser.add_argument("--pilot-existing", type=int, default=6)
    parser.add_argument("--pilot-new", type=int, default=6)
    parser.add_argument("--bases", choices=("existing", "new"), default="existing")
    parser.add_argument("--prompts-per-base", type=int, default=1)
    parser.add_argument("--total-prompts", type=int, help="Nombre total de prompts, répartis au prorata des exemples Spider par base.")
    parser.add_argument("--exemplars", type=int, default=16, help="Exemples Spider de structures distinctes par prompt.")
    parser.add_argument("--per-exemplar", type=int, default=1, help="Paires générées par exemple.")
    parser.add_argument("--hardness", nargs="+", choices=("easy", "medium", "hard", "extra"),
                        help="Ne tire que des exemples Spider de ces difficultés (par défaut : proportions de Spider).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--output", type=Path, default=HERE / "pilot")
    parser.add_argument("--run-sync", action="store_true", help="Exécute les requêtes directement (pilote) au lieu d'un Batch.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--submit", action="store_true", help="Envoie les requêtes en Batch OpenAI.")
    parser.add_argument("--download", action="store_true", help="Vérifie le Batch soumis et télécharge ses réponses.")
    args = parser.parse_args()
    if args.download:
        download(args.output)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    requests = build_requests(args)
    batch_lines = [{key_: value for key_, value in row.items() if not key_.startswith("_")} for row in requests]
    (args.output / "requests.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in batch_lines) + "\n", encoding="utf-8")
    (args.output / "request_meta.jsonl").write_text("\n".join(json.dumps(
        {"custom_id": row["custom_id"], "database_path": row["_database_path"], "exemplars": row["_exemplars"]}, ensure_ascii=False)
        for row in requests) + "\n", encoding="utf-8")
    print(f"{len(requests)} prompts / {len(requests) * args.exemplars * args.per_exemplar} paires demandées -> {args.output / 'requests.jsonl'}")
    if args.submit:
        submit(args.output, args.model)
        return
    if not args.run_sync:
        return

    token = key()
    outputs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call, row["body"], token): row["custom_id"] for row in batch_lines}
        for done, future in enumerate(as_completed(futures), start=1):
            response = future.result()
            outputs.append({"custom_id": futures[future], "response": response})
            print(f"  {done}/{len(futures)} {futures[future]} -> HTTP {response['status_code']}", flush=True)
    (args.output / "responses.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in outputs) + "\n", encoding="utf-8")
    usage = defaultdict(int)
    for row in outputs:
        for name, value in (row["response"]["body"].get("usage") or {}).items():
            if isinstance(value, int):
                usage[name] += value
    print("Usage :", dict(usage))


if __name__ == "__main__":
    main()
