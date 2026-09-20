"""Juge sémantiquement les traductions pilot ou production avec gpt-5.6-sol."""

import json
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
SOURCE = WORK = STATE = OUTPUT = SELECTION = ENVIRONMENT = REQUESTS_DIR = RAW_DIR = None
SELECTED_SPLITS: tuple[str, ...] | None = None
MODEL, SIZE, CHUNK = "gpt-5.6-sol", 100, 50
PRODUCTION_SAMPLE_SIZE, RANDOM_SEED = 200, 20260919
PROMPT = """Tu es le juge qualité d'un dataset text-to-SQL traduit de l'anglais vers le français.
Évalue la fidélité de la traduction française. Le SQL et le schéma servent seulement de garde-fous. Ne reformule jamais.
Un verdict pass conserve le sens ; fail ajoute, retire ou change une information ; uncertain ne permet pas de décider.
Vérifie nombres, dates, pourcentages, noms propres, comparaisons, négations, superlatifs, minimums, maximums, tris et classements.
Retourne uniquement : {"id":"<id reçu>","verdict":"pass"|"fail"|"uncertain","issues":["<raison concise>"]}.
issues est vide pour pass."""


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def key():
    value = os.environ.get("OPENAI_API_KEY")
    env = ROOT / ".env"
    if not value and env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, sep, candidate = line.partition("=")
            if name.strip() == "OPENAI_API_KEY" and sep:
                value = candidate.strip().strip('"').strip("'")
                break
    if not value:
        raise RuntimeError("OPENAI_API_KEY n'est pas configurée.")
    return value


def api(path, method="GET", payload=None):
    headers = {"Authorization": f"Bearer {key()}"}
    data = None
    if payload is not None:
        data, headers["Content-Type"] = json.dumps(payload).encode(), "application/json"
    request = Request("https://api.openai.com/v1" + path, data=data, headers=headers, method=method)
    with urlopen(request) as response:
        return json.loads(response.read().decode())


def request_row(record, identifier):
    fields = ("question_original_en", "question", "sql", "schema")
    bad = [field for field in fields if not isinstance(record.get(field), str) or not record[field].strip()]
    if bad:
        raise ValueError(f"{identifier} : champs invalides : {', '.join(bad)}")
    return {"custom_id": f"judge:{identifier}", "method": "POST", "url": "/v1/responses", "body": {
        "model": MODEL, "reasoning": {"effort": "low"}, "instructions": PROMPT,
        "input": json.dumps({"id": identifier, **{field: record[field] for field in fields}}, ensure_ascii=False),
        "max_output_tokens": 1024,
    }}


def selection_records():
    if ENVIRONMENT == "pilot":
        records = rows(SOURCE)[:SIZE]
        if len(records) != SIZE:
            raise ValueError(f"{SIZE} traductions requises, {len(records)} trouvées.")
        return [{"id": f"train_spider:{index}", "selection_reason": "pilot_full", **record} for index, record in enumerate(records)]

    selected, passing = [], {}
    splits = SELECTED_SPLITS or ("train_spider", "train_others", "dev")
    for split in splits:
        translations = rows(ROOT / "data" / "04_merge_translations" / "03_production" / f"{split}.jsonl")
        checks = {row["id"]: row for row in rows(WORK / "01_deterministic_checks" / f"{split}_deterministic_checks.jsonl")}
        if len(checks) != len(translations):
            raise ValueError(f"Contrôles incomplets pour {split}.")
        passing[split] = []
        for index, record in enumerate(translations):
            identifier = f"{split}:{index}"
            check = checks.get(identifier)
            if not check:
                raise ValueError(f"Contrôle absent : {identifier}")
            entry = {"id": identifier, **record}
            if check["status"] == "fail":
                selected.append({"selection_reason": "deterministic_fail", **entry})
            else:
                passing[split].append(entry)

    total = sum(map(len, passing.values()))
    quotas = {split: len(candidates) * PRODUCTION_SAMPLE_SIZE // total for split, candidates in passing.items()}
    for split in sorted(passing, key=lambda name: len(passing[name]) * PRODUCTION_SAMPLE_SIZE % total, reverse=True)[:PRODUCTION_SAMPLE_SIZE - sum(quotas.values())]:
        quotas[split] += 1
    generator = random.Random(RANDOM_SEED)
    for split, candidates in passing.items():
        selected.extend({"selection_reason": "random_pass", **entry} for entry in generator.sample(candidates, quotas[split]))
    return sorted(selected, key=lambda row: (row["id"].split(":")[0], int(row["id"].split(":")[1])))


def prepare():
    records = selection_records()
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    SELECTION.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    failures = sum(record["selection_reason"] == "deterministic_fail" for record in records)
    print(f"{len(records)} traductions sélectionnées : {failures} signaux mécaniques + {len(records) - failures} échantillons aléatoires.")
    requests = [request_row(row, row["id"]) for row in records]
    REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    for number, start in enumerate(range(0, len(requests), CHUNK), 1):
        path = REQUESTS_DIR / f"requests_{number:02d}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests[start:start + CHUNK]), encoding="utf-8")
        print(f"{len(requests[start:start + CHUNK])} jugements écrits dans {path}")


def upload(path):
    boundary = "----Judge" + uuid.uuid4().hex
    body = b"".join((
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\nContent-Type: application/jsonl\r\n\r\n".encode(),
        path.read_bytes(), f"\r\n--{boundary}--\r\n".encode(),
    ))
    request = Request("https://api.openai.com/v1/files", data=body, headers={"Authorization": f"Bearer {key()}", "Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    with urlopen(request) as response:
        return json.loads(response.read().decode())


def submit():
    paths = sorted(REQUESTS_DIR.glob("requests_*.jsonl"))
    if not paths:
        prepare()
        paths = sorted(REQUESTS_DIR.glob("requests_*.jsonl"))
    batches = []
    for path in paths:
        file = upload(path)
        batch = api("/batches", "POST", {"input_file_id": file["id"], "endpoint": "/v1/responses", "completion_window": "24h", "metadata": {"job": f"spider-judge-{ENVIRONMENT}", "model": MODEL}})
        batches.append({"batch_id": batch["id"], "input_file_id": file["id"]})
        print(f"Batch de jugement créé : {batch['id']}")
    WORK.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"environment": ENVIRONMENT, "model": MODEL, "selection": str(SELECTION.relative_to(ROOT)), "submitted_at": datetime.now(timezone.utc).isoformat(), "batches": batches}, indent=2) + "\n", encoding="utf-8")


def state():
    if not STATE.is_file():
        raise RuntimeError("Aucun Batch de jugement connu.")
    return json.loads(STATE.read_text(encoding="utf-8"))


def status():
    result = []
    for number, saved in enumerate(state()["batches"], 1):
        batch = api(f"/batches/{saved['batch_id']}")
        print(f"- Lot {number} : {batch['id']} — {batch['status']} — {batch.get('request_counts')}")
        result.append(batch)
    return result


def extract(line):
    response = line.get("response", {})
    body = response.get("body", {})
    text = "".join(item.get("text", "") for output in body.get("output", []) if isinstance(output, dict) for item in output.get("content", []) if isinstance(item, dict) and item.get("type") == "output_text")
    answer = json.loads(text)
    if set(answer) != {"id", "verdict", "issues"} or answer["verdict"] not in {"pass", "fail", "uncertain"} or not isinstance(answer["issues"], list):
        raise RuntimeError(f"Verdict invalide : {answer!r}")
    return answer["id"], answer["verdict"], answer["issues"]


def collect():
    verdicts = {}
    for number, saved in enumerate(state()["batches"], 1):
        batch = api(f"/batches/{saved['batch_id']}")
        if batch.get("status") != "completed":
            raise RuntimeError(f"Le batch {batch['id']} n'est pas terminé : {batch.get('status')}")
        request = Request(f"https://api.openai.com/v1/files/{batch['output_file_id']}/content", headers={"Authorization": f"Bearer {key()}"})
        with urlopen(request) as response:
            raw = response.read().decode()
        raw_path = RAW_DIR / f"output_{number:02d}.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw, encoding="utf-8")
        verdicts.update({identifier: (verdict, issues) for identifier, verdict, issues in map(extract, map(json.loads, filter(None, raw.splitlines())))})
    judged = []
    for record in rows(SELECTION):
        verdict, issues = verdicts[record["id"]]
        judged.append({**record, "verdict": verdict, "issues": issues})
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in judged), encoding="utf-8")
    print(f"{len(judged)} jugements écrits dans {OUTPUT}")


def main():
    global SOURCE, WORK, STATE, OUTPUT, SELECTION, ENVIRONMENT, SIZE, SELECTED_SPLITS, REQUESTS_DIR, RAW_DIR
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=("train_spider", "train_others", "dev", "test"), help="juge uniquement les splits indiqués en production")
    args = parser.parse_args()
    choice = input("Dossier à traiter — 1) pilot  2) production [1/2] : ").strip()
    environments = {"1": "pilot", "2": "production"}
    if choice not in environments:
        raise ValueError("Choix attendu : 1 (pilot) ou 2 (production).")
    environment = environments[choice]
    if environment == "pilot" and args.splits:
        raise ValueError("--splits est réservé au mode production.")
    ENVIRONMENT = environment
    SELECTED_SPLITS = tuple(args.splits) if args.splits else None
    folder = {"pilot": "02_pilot", "production": "03_production"}[environment]
    SOURCE = ROOT / "data" / "04_merge_translations" / folder / "train_spider.jsonl"
    WORK = ROOT / "data" / "05_quality_control" / folder
    suffix = "" if not SELECTED_SPLITS else "_" + "_".join(SELECTED_SPLITS)
    STATE = WORK / "08_judge_batch_state" / f"judge_batch_state{suffix}.json"
    OUTPUT = WORK / "09_judgments" / f"sol_judgments{suffix}.jsonl"
    SELECTION = WORK / "03_judge_selection" / f"judge_selection{suffix}.jsonl"
    REQUESTS_DIR = WORK / ("05_judge_requests_test" if suffix == "_test" else "04_judge_requests")
    RAW_DIR = WORK / ("07_judge_raw_response_test" if suffix == "_test" else "06_judge_raw_response")
    SIZE = 100 if environment == "pilot" else None
    if not STATE.is_file():
        prepared_paths = sorted(REQUESTS_DIR.glob("requests_*.jsonl"))
        if prepared_paths:
            print(f"{len(prepared_paths)} lots du juge sont déjà préparés.")
            if input("Envoyer ces lots ? [o/N] ").strip().lower() in {"o", "oui"}:
                submit()
            return
        if input("Préparer les lots du juge ? [o/N] ").strip().lower() in {"o", "oui"}:
            prepare()
            if input("Envoyer ces lots ? [o/N] ").strip().lower() in {"o", "oui"}:
                submit()
        return
    batches = status()
    if all(batch.get("status") == "completed" for batch in batches) and input("Les jugements sont prêts. Les récupérer ? [o/N] ").strip().lower() in {"o", "oui"}:
        collect()


if __name__ == "__main__":
    main()
