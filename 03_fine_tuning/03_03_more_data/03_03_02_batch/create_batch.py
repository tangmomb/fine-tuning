"""Prépare un Batch OpenAI de génération text-to-SQL par base et difficulté."""
from __future__ import annotations
import argparse, json, math, mimetypes, os, uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

HERE = Path(__file__).resolve().parents[1]; ROOT = HERE.parents[1]; DEFAULT_MODEL = "gpt-5.6-luna"; MAX_OUTPUT_TOKENS = 128000
PREPARE = HERE / "03_03_01_prepare"
BATCH = HERE / "03_03_02_batch"
QUALITY = HERE / "03_03_03_quality"
PAIR = {"type": "object", "additionalProperties": False, "properties": {"question": {"type": "string"}, "sql": {"type": "string"}}, "required": ["question", "sql"]}
SCHEMA = {"type": "object", "additionalProperties": False, "properties": {"examples": {"type": "array", "items": PAIR}}, "required": ["examples"]}
SPIDER_SQL_RULES = """Le SQL doit impérativement appartenir au sous-ensemble pris en charge par l'évaluateur officiel Spider : il sera parsé pour calculer sa difficulté, donc une requête SQLite valide mais non reconnue sera rejetée. Utilise uniquement SELECT, DISTINCT, FROM, JOIN ... ON, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, agrégats (COUNT, SUM, AVG, MIN, MAX), comparaisons et sous-requêtes SELECT avec IN/NOT IN. AS est permis uniquement pour les alias de tables (par exemple FROM table AS t) ; n'aliase jamais une colonne, un agrégat ou une expression dans SELECT. N'encadre jamais les noms de table ou de colonne avec des guillemets ni des accents graves ; les chaînes de texte utilisent exclusivement des apostrophes simples. Pour « différent de », utilise !=, jamais <>. N'utilise jamais WITH/CTE, LEFT/RIGHT/OUTER JOIN, IS NULL/IS NOT NULL, EXISTS/NOT EXISTS, CASE, CAST, fenêtres OVER/RANK, UNION/INTERSECT/EXCEPT, ALL/ANY, expressions arithmétiques, sous-requête dans SELECT/FROM ou comme opérande de comparaison, ni IN avec une liste de valeurs. N'ajoute ni commentaire SQL, ni libellé de difficulté, ni texte hors de la requête."""
HARDNESS_RECIPES = {
    "easy": """Forme obligatoire easy : une seule table, une seule expression simple dans SELECT, sans agrégat, jointure, GROUP BY, HAVING, ORDER BY, LIMIT ni sous-requête ; au plus un filtre simple dans WHERE.""",
    "medium": """Forme obligatoire medium : une seule table et exactement SELECT colonne_de_groupe, COUNT(*) FROM table GROUP BY colonne_de_groupe. Aucun alias de résultat, filtre, jointure, HAVING, ORDER BY, LIMIT ni sous-requête.""",
    "hard": """Forme obligatoire hard : copie exactement cette structure, en choisissant seulement les noms existants : SELECT colonne_affichee FROM table WHERE cle IN (SELECT cle FROM table). La table et la clé de la sous-requête doivent être les mêmes que dans la requête externe. Aucun alias, DISTINCT, agrégat, jointure, GROUP BY, HAVING, ORDER BY, LIMIT, second filtre ni autre sous-requête.""",
    "extra": """Forme obligatoire extra : exactement SELECT colonne_de_groupe, COUNT(*) FROM table WHERE filtre_simple_1 AND filtre_simple_2 GROUP BY colonne_de_groupe. Les deux filtres sont des comparaisons simples compatibles avec le type des colonnes, sans sous-requête. Aucun alias de résultat, jointure, HAVING, ORDER BY ni LIMIT.""",
}
def rows(path: Path) -> list[dict]: return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def largest_remainder(weights: dict[str, int], total: int) -> dict[str, int]:
    raw = {key: total * value / sum(weights.values()) for key, value in weights.items()}; quotas = {key: int(value) for key, value in raw.items()}
    for key in sorted(weights, key=lambda item: (raw[item] - quotas[item], item), reverse=True)[:total - sum(quotas.values())]: quotas[key] += 1
    return quotas
def difficulty_quotas(row: dict, experiment: str, total: int) -> dict[str, int]:
    counts = {key: int(value) for key, value in row.get("hardness_counts", {}).items() if key in {"easy", "medium", "hard", "extra"} and int(value) > 0}
    if experiment == "A" and counts: return largest_remainder(counts, total)
    return {}
def quota_plan(seeds: list[dict], experiment: str, totals: dict[str, int]) -> dict[str, dict[str, int]]:
    """Répartit les quotas par difficulté, avec un 50/50 global pour C/D."""
    if experiment == "A": return {row["db_id"]: difficulty_quotas(row, experiment, totals[row["db_id"]]) for row in seeds}
    if experiment not in {"C", "D"}: return {row["db_id"]: {} for row in seeds}
    plan = {row["db_id"]: {"hard": totals[row["db_id"]] // 2, "extra": totals[row["db_id"]] // 2} for row in seeds}
    odd_databases = sorted(row["db_id"] for row in seeds if totals[row["db_id"]] % 2)
    split = (len(odd_databases) + 1) // 2
    for db_id in odd_databases[:split]: plan[db_id]["hard"] += 1
    for db_id in odd_databases[split:]: plan[db_id]["extra"] += 1
    return plan
def complement_plan(seeds: list[dict], experiment: str, selected_path: Path, oversample_factor: float) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    selected = rows(selected_path) if selected_path.is_file() else []
    selected_by_db_hardness = Counter((row["db_id"], row["hardness"]) for row in selected)
    request_quotas, distributions = {}, {}
    target_plan = quota_plan(seeds, experiment, {row["db_id"]: row["target_quota"] for row in seeds})
    for row in seeds:
        target = target_plan[row["db_id"]]
        missing = {level: quota - selected_by_db_hardness[row["db_id"], level] for level, quota in target.items() if quota > selected_by_db_hardness[row["db_id"], level]}
        if not missing: continue
        request_quota = math.ceil(sum(missing.values()) * oversample_factor)
        request_quotas[row["db_id"]] = request_quota
        distributions[row["db_id"]] = largest_remainder(missing, request_quota)
    return request_quotas, distributions
def request_chunks(total: int, max_examples: int, quotas: dict[str, int]) -> list[tuple[int, dict[str, int]]]:
    """Découpe une demande sans perdre la répartition de difficulté demandée."""
    if not quotas:
        return [(min(max_examples, total - start), {}) for start in range(0, total, max_examples)]
    remaining = dict(quotas); chunks = []
    while sum(remaining.values()):
        size = min(max_examples, sum(remaining.values()))
        chunk = largest_remainder(remaining, size)
        chunks.append((size, chunk))
        remaining = {hardness: remaining[hardness] - chunk.get(hardness, 0) for hardness in remaining}
    return chunks
def key() -> str:
    if os.environ.get("OPENAI_API_KEY"): return os.environ["OPENAI_API_KEY"]
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY="): return line.partition("=")[2].strip().strip('"')
    raise RuntimeError("OPENAI_API_KEY est absent.")
def post(url: str, payload: dict, token: str) -> dict:
    req = request.Request(url, data=json.dumps(payload).encode(), method="POST", headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with request.urlopen(req) as response: return json.load(response)
def upload(path: Path, token: str) -> dict:
    boundary = f"----codex{uuid.uuid4().hex}"; data = bytearray(); data.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'.encode())
    mime = mimetypes.guess_type(path.name)[0] or "application/jsonl"; data.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\nContent-Type: {mime}\r\n\r\n'.encode()); data.extend(path.read_bytes()); data.extend(f"\r\n--{boundary}--\r\n".encode())
    req = request.Request("https://api.openai.com/v1/files", data=bytes(data), method="POST", headers={"Authorization": f"Bearer {token}", "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with request.urlopen(req) as response: return json.load(response)
def ask_experiment(value: str | None) -> str:
    choice = value or input("Expérience à préparer pour Batch [A/B/C/D] : ").strip().upper()
    if choice not in "ABCD": raise ValueError("Choix attendu : A, B, C ou D.")
    return choice
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--experiment", choices="ABCD"); parser.add_argument("--model"); parser.add_argument("--request-count", type=int, help="Nombre total d'exemples à répartir entre les bases."); parser.add_argument("--complement", action="store_true", help="Ne demande que les quotas Spider manquants après la sélection courante."); parser.add_argument("--max-examples-per-request", type=int, default=80, help="Taille maximale d'une réponse ciblant une base et une difficulté."); parser.add_argument("--oversample-factor", type=float, default=1.15, help="Marge appliquée au complément ciblé."); parser.add_argument("--submit", action="store_true"); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); args.experiment = ask_experiment(args.experiment); args.model = args.model or input(f"Modèle [{DEFAULT_MODEL}] : ").strip() or DEFAULT_MODEL
    seed_source = "existing" if args.experiment in {"A", "C"} else "new"
    split_by_hardness = args.experiment in {"A", "C", "D"}
    seeds = rows(PREPARE / f"{seed_source}_sql_bases.jsonl"); output = BATCH / "03_03_02_01_requests" / f"{args.experiment}_requests.jsonl"
    if args.request_count is not None and args.request_count < 1: raise ValueError("--request-count doit être positif.")
    if args.complement and args.request_count is not None: raise ValueError("--request-count est incompatible avec --complement.")
    if args.max_examples_per_request < 1: raise ValueError("--max-examples-per-request doit être positif.")
    if args.complement and args.oversample_factor < 1: raise ValueError("--oversample-factor doit être >= 1.")
    selected_path = QUALITY / "03_03_03_01_deterministic_checks" / args.experiment / "selected.jsonl"
    if args.complement:
        request_quotas, complement_distributions = complement_plan(seeds, args.experiment, selected_path, args.oversample_factor)
    else:
        request_quotas, complement_distributions = (largest_remainder({row["db_id"]: row["source_examples"] for row in seeds}, args.request_count) if args.request_count else {row["db_id"]: row["request_quota"] for row in seeds}), {}
    request_plan = quota_plan(seeds, args.experiment, request_quotas)
    if output.exists() and not args.overwrite:
        if input(f"{output.name} existe déjà. Le remplacer ? [o/N] ").strip().lower() not in {"o", "oui"}: return
    lines = []
    if split_by_hardness:
        for row in seeds:
            if args.complement and row["db_id"] not in complement_distributions: continue
            target = complement_distributions[row["db_id"]] if args.complement else request_plan[row["db_id"]]
            for hardness, quota in target.items():
                if not quota: continue
                requested_for_hardness = quota
                for chunk_number, start in enumerate(range(0, requested_for_hardness, args.max_examples_per_request), start=1):
                    request_quota = min(args.max_examples_per_request, requested_for_hardness - start)
                    prompt = (f"Génère exactement {request_quota} paires indépendantes question/SQL SQLite de difficulté Spider {hardness} pour ce schéma. Chaque paire doit être classée {hardness} par l'évaluateur officiel Spider ; ne produis aucun autre niveau. {HARDNESS_RECIPES[hardness]} Chaque question est naturelle, en français ; chaque SQL est exécutable et en lecture seule. N'invente ni table ni colonne et n'ajoute aucune explication. Diversifie les tables et formulations dans cette forme obligatoire. {SPIDER_SQL_RULES}\n\nSCHÉMA ({row['db_id']}):\n{row['schema']}")
                    body = {"model": args.model, "input": [{"role": "system", "content": "Tu produis des données text-to-SQL françaises fiables."}, {"role": "user", "content": prompt}], "max_output_tokens": MAX_OUTPUT_TOKENS, "text": {"format": {"type": "json_schema", "name": "synthetic_examples", "strict": True, "schema": SCHEMA}}}
                    lines.append(json.dumps({"custom_id": f"{args.experiment}:{row['db_id']}:{hardness}:{chunk_number}", "method": "POST", "url": "/v1/responses", "body": body}, ensure_ascii=False))
        requested_total = sum(quota for quotas in (complement_distributions.values() if args.complement else request_plan.values()) for quota in quotas.values())
    else:
        requested_total = sum(request_quotas.values())
    for row in seeds:
        if split_by_hardness: continue
        if row["db_id"] not in request_quotas: continue
        total_quota = request_quotas[row["db_id"]]
        quotas = complement_distributions.get(row["db_id"], request_plan[row["db_id"]])
        if args.experiment == "A":
            difficulty = "Suis strictement la répartition de difficulté demandée."
        elif args.experiment in {"C", "D"}:
            difficulty = "Utilise jointures, agrégations, sous-requêtes ou opérations ensemblistes quand cela est pertinent."
        else:
            difficulty = "Varie librement la difficulté et les constructions SQL."
        for chunk_number, (request_quota, chunk_quotas) in enumerate(request_chunks(total_quota, args.max_examples_per_request, quotas), start=1):
            distribution = f" Respecte exactement cette répartition de difficulté Spider : {json.dumps(chunk_quotas, ensure_ascii=False)}." if chunk_quotas else ""
            prompt = (f"Génère exactement {request_quota} paires indépendantes question/SQL SQLite pour ce schéma. Chaque question est naturelle, en français ; chaque SQL est exécutable et en lecture seule. N'invente ni table ni colonne et n'ajoute aucune explication. Diversifie les tables et formulations. {SPIDER_SQL_RULES} {difficulty}{distribution}\n\nSCHÉMA ({row['db_id']}):\n{row['schema']}")
            body = {"model": args.model, "input": [{"role": "system", "content": "Tu produis des données text-to-SQL françaises fiables."}, {"role": "user", "content": prompt}], "max_output_tokens": MAX_OUTPUT_TOKENS, "text": {"format": {"type": "json_schema", "name": "synthetic_examples", "strict": True, "schema": SCHEMA}}}
            lines.append(json.dumps({"custom_id": f"{args.experiment}:{row['db_id']}:{chunk_number}", "method": "POST", "url": "/v1/responses", "body": body}, ensure_ascii=False))
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text("\n".join(lines) + "\n", encoding="utf-8"); print(f"{len(lines)} prompts / {requested_total} exemples demandés écrits dans {output}")
    if not args.submit and input("Soumettre ce Batch à OpenAI maintenant ? [o/N] ").strip().lower() not in {"o", "oui"}: return
    token = key(); uploaded = upload(output, token); batch = post("https://api.openai.com/v1/batches", {"input_file_id": uploaded["id"], "endpoint": "/v1/responses", "completion_window": "24h", "metadata": {"pipeline": "more-data", "experiment": args.experiment, "model": args.model}}, token)
    state_path = BATCH / "03_03_02_02_submissions" / f"{args.experiment}_batch_state.json"; state_path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(state_path.read_text(encoding="utf-8")) if args.complement and state_path.is_file() else {}
    prior_raw_output_files = list(previous.get("raw_output_files", []))
    if previous.get("raw_output_file") and previous["raw_output_file"] not in prior_raw_output_files: prior_raw_output_files.append(previous["raw_output_file"])
    prior_requested = previous.get("cumulative_requested_examples", previous.get("requested_examples", 0))
    mode = "split_by_hardness_complement" if args.complement and split_by_hardness else "complement" if args.complement else "split_by_hardness" if split_by_hardness else "full"
    state = {"experiment": args.experiment, "model": args.model, "mode": mode, "requested_examples": requested_total, "cumulative_requested_examples": prior_requested + requested_total, "raw_output_files": prior_raw_output_files, "submitted_at": datetime.now(timezone.utc).isoformat(), "request_file": output.name, "input_file_id": uploaded["id"], "batch_id": batch["id"], "status": batch.get("status")}
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8"); print(f"Batch soumis : {batch['id']}\nEntrée envoyée : {output}")
if __name__ == "__main__": main()
