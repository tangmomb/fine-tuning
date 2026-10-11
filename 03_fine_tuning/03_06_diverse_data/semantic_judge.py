"""Juge sémantique v2 : vérifie que chaque question permet d'écrire le SQL associé.

Différences avec 03_03_03_02_semantic_judge :
- toutes les paires valides sont jugées (et pas seulement 5 000 présélectionnées) : l'export
  choisit ensuite parmi les paires acceptées, il n'y a donc pas besoin de boucle de réécriture ;
- le critère n'est plus « la question décrit exactement le SQL », qui poussait vers des
  questions littérales, mais « un expert écrirait une requête au résultat identique » ;
- les paires sont envoyées par paquets de 40 au plus.

Étapes :
  python semantic_judge.py --input A_v2 --prepare      # écrit judge_requests.jsonl
  python semantic_judge.py --input A_v2 --submit       # envoie le Batch (gpt-5.6-terra)
  python semantic_judge.py --input A_v2 --collect      # statut, puis judged.jsonl une fois terminé
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib import request

HERE = Path(__file__).resolve().parent
FINE_TUNING = HERE.parent
sys.path.insert(0, str(FINE_TUNING / "03_03_more_data" / "03_03_02_batch"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FINE_TUNING / "03_03_more_data" / "03_03_03_quality" / "03_03_03_01_deterministic_checks"))
from create_batch import key, post, upload  # noqa: E402
from deterministic_check import output_text  # noqa: E402

PREPARE = FINE_TUNING / "03_03_more_data" / "03_03_01_prepare"
DEFAULT_MODEL = "gpt-5.6-terra"
CHUNK = 40
VERDICT = {"type": "object", "additionalProperties": False,
           "properties": {"id": {"type": "string"}, "verdict": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
                          "issues": {"type": "array", "items": {"type": "string"}}},
           "required": ["id", "verdict", "issues"]}
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"verdicts": {"type": "array", "items": VERDICT}}, "required": ["verdicts"]}
SYSTEM = "Tu es un juge qualité text-to-SQL strict et indépendant."
INSTRUCTIONS = """Évalue chaque paire question/SQL pour ce schéma SQLite.

Question à te poser : un expert SQL qui lit seulement la question française et le schéma écrirait-il une requête dont le résultat est identique à celui du SQL fourni ?

- pass : oui. La question peut être formulée naturellement, sans nommer les colonnes techniques ni décrire la syntaxe ; les choix conventionnels d'un expert (par exemple afficher le nom pour « quel client ») sont acceptables.
- fail : la question ajoute, retire ou modifie une contrainte ; le SQL renvoie des colonnes que la question ne demande pas, ou oublie une colonne demandée ; l'agrégat, le tri, la limite, la négation, la jointure ou une valeur ne correspond pas ; la question est absurde, tautologique ou mal écrite en français.
- uncertain : la question admet plusieurs lectures raisonnables qui donnent des résultats différents.

Conventions du jeu Spider, à accepter sans les pénaliser : jointure interne même pour « chaque » ou « tous » ; regroupement ou INTERSECT/EXCEPT sur un nom affiché plutôt que sur un identifiant ; doublons possibles sans DISTINCT quand la question ne demande pas explicitement des valeurs distinctes ; cas limites théoriques (homonymes, valeurs NULL, ex æquo) que la question n'évoque pas. Ne rejette que les écarts de sens réels.

Retourne exactement un verdict pour chaque identifiant fourni ; dans issues, décris brièvement le problème (liste vide si pass)."""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def schemas() -> dict[str, str]:
    return {row["db_id"]: row["schema"] for name in ("existing_sql_bases.jsonl", "new_sql_bases.jsonl")
            for row in read_jsonl(PREPARE / name)}


def prepare(folder: Path, model: str) -> None:
    candidates = read_jsonl(folder / "validated.jsonl")
    schema_by_db = schemas()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        grouped[row["db_id"]].append(row)
    lines = []
    for db_id, rows in sorted(grouped.items()):
        for start in range(0, len(rows), CHUNK):
            chunk = rows[start:start + CHUNK]
            payload = {"schema": schema_by_db[db_id],
                       "examples": [{"id": row["id"], "question": row["question"], "sql": row["sql"]} for row in chunk]}
            body = {"model": model,
                    "input": [{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": INSTRUCTIONS + "\n\n" + json.dumps(payload, ensure_ascii=False)}],
                    "max_output_tokens": 16000,
                    "text": {"format": {"type": "json_schema", "name": "semantic_verdicts", "strict": True, "schema": SCHEMA}}}
            lines.append({"custom_id": f"judge:{db_id}:{start // CHUNK + 1}", "method": "POST", "url": "/v1/responses", "body": body})
    write_jsonl(folder / "judge_requests.jsonl", lines)
    print(f"{len(candidates)} paires à juger dans {len(lines)} requêtes -> {folder / 'judge_requests.jsonl'}")


def submit(folder: Path, model: str) -> None:
    token = key()
    uploaded = upload(folder / "judge_requests.jsonl", token)
    batch = post("https://api.openai.com/v1/batches", {"input_file_id": uploaded["id"], "endpoint": "/v1/responses", "completion_window": "24h",
                                                        "metadata": {"pipeline": "diverse-data-v2-judge", "output": folder.name, "model": model}}, token)
    state = {"batch_id": batch["id"], "input_file_id": uploaded["id"], "model": model, "status": batch.get("status")}
    (folder / "judge_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Batch de jugement soumis : {batch['id']} ({batch.get('status')})")


def collect(folder: Path) -> None:
    token = key()
    state_path = folder / "judge_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))

    def get(url: str) -> bytes:
        with request.urlopen(request.Request(url, headers={"Authorization": f"Bearer {token}"})) as response:
            return response.read()

    batch = json.loads(get(f"https://api.openai.com/v1/batches/{state['batch_id']}"))
    counts = batch.get("request_counts", {})
    state["status"] = batch.get("status")
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Statut : {batch.get('status')} · {counts.get('completed', 0)}/{counts.get('total', 0)} terminées · {counts.get('failed', 0)} échouées")
    if not batch.get("output_file_id"):
        return
    (folder / "judge_responses.jsonl").write_bytes(get(f"https://api.openai.com/v1/files/{batch['output_file_id']}/content"))
    consolidate(folder)


def run_sync(folder: Path) -> None:
    """Exécute les requêtes directement (pilote), au même format que la sortie Batch."""
    from concurrent.futures import ThreadPoolExecutor
    from create_requests import call
    token = key()
    lines = read_jsonl(folder / "judge_requests.jsonl")
    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(lambda line: {"custom_id": line["custom_id"], "response": call(line["body"], token)}, lines))
    write_jsonl(folder / "judge_responses.jsonl", responses)
    consolidate(folder)


def consolidate(folder: Path) -> None:
    verdicts = {}
    for line in read_jsonl(folder / "judge_responses.jsonl"):
        try:
            answers = json.loads(output_text(line) or "")["verdicts"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        for answer in answers:
            if answer.get("verdict") in {"pass", "fail", "uncertain"} and isinstance(answer.get("id"), str):
                verdicts[answer["id"]] = answer
    candidates = read_jsonl(folder / "validated.jsonl")
    judged = [{**row, "verdict": verdicts.get(row["id"], {}).get("verdict", "missing"),
               "issues": verdicts.get(row["id"], {}).get("issues", ["Verdict absent."])} for row in candidates]
    write_jsonl(folder / "judged.jsonl", judged)
    summary = Counter(row["verdict"] for row in judged)
    print("Verdicts :", dict(summary))
    print("Pass par difficulté :", dict(Counter(row["hardness"] for row in judged if row["verdict"] == "pass")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Dossier contenant validated.jsonl (sortie de check_pilot.py).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--submit", action="store_true")
    action.add_argument("--collect", action="store_true")
    action.add_argument("--run-sync", action="store_true", help="Exécute directement les requêtes préparées (pilote).")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.input, args.model)
    elif args.submit:
        submit(args.input, args.model)
    elif args.run_sync:
        run_sync(args.input)
    else:
        collect(args.input)


if __name__ == "__main__":
    main()
