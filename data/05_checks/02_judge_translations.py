"""Juge sémantiquement les traductions pilot ou production avec gpt-5.6-sol."""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
SOURCE = WORK = STATE = OUTPUT = None
MODEL, SIZE, CHUNK = "gpt-5.6-sol", 100, 50
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


def request_row(record, index):
    fields = ("question_original_en", "question", "sql", "schema")
    bad = [field for field in fields if not isinstance(record.get(field), str) or not record[field].strip()]
    if bad:
        raise ValueError(f"Ligne {index + 1} : champs invalides : {', '.join(bad)}")
    identifier = f"train_spider:{index}"
    return {"custom_id": f"judge:{identifier}", "method": "POST", "url": "/v1/responses", "body": {
        "model": MODEL, "reasoning": {"effort": "low"}, "instructions": PROMPT,
        "input": json.dumps({"id": identifier, **{field: record[field] for field in fields}}, ensure_ascii=False),
        "max_output_tokens": 1024,
    }}


def prepare():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Traductions absentes : {SOURCE}. Récupérez-les avant de lancer le juge.")
    records = rows(SOURCE)[:SIZE]
    if SIZE is not None and len(records) != SIZE:
        raise ValueError(f"{SIZE} traductions requises, {len(records)} trouvées.")
    requests = [request_row(row, index) for index, row in enumerate(records)]
    target = WORK / "judge_requests"
    target.mkdir(parents=True, exist_ok=True)
    for number, start in enumerate(range(0, len(requests), CHUNK), 1):
        path = target / f"requests_{number:02d}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests[start:start + CHUNK]), encoding="utf-8")
        print(f"{CHUNK} jugements écrits dans {path}")


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
    paths = sorted((WORK / "judge_requests").glob("requests_*.jsonl"))
    if not paths:
        prepare()
        paths = sorted((WORK / "judge_requests").glob("requests_*.jsonl"))
    batches = []
    for path in paths:
        file = upload(path)
        batch = api("/batches", "POST", {"input_file_id": file["id"], "endpoint": "/v1/responses", "completion_window": "24h", "metadata": {"job": "spider-judge-pilot", "model": MODEL}})
        batches.append({"batch_id": batch["id"], "input_file_id": file["id"]})
        print(f"Batch de jugement créé : {batch['id']}")
    WORK.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"model": MODEL, "source": str(SOURCE.relative_to(ROOT)), "submitted_at": datetime.now(timezone.utc).isoformat(), "batches": batches}, indent=2) + "\n", encoding="utf-8")


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
        raw_path = WORK / "judge_raw_response" / f"output_{number:02d}.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw, encoding="utf-8")
        verdicts.update({identifier: (verdict, issues) for identifier, verdict, issues in map(extract, map(json.loads, filter(None, raw.splitlines())))})
    judged = []
    for index, record in enumerate(rows(SOURCE)[:SIZE]):
        verdict, issues = verdicts[f"train_spider:{index}"]
        judged.append({**record, "verdict": verdict, "issues": issues})
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in judged), encoding="utf-8")
    print(f"{len(judged)} jugements écrits dans {OUTPUT}")


def main():
    global SOURCE, WORK, STATE, OUTPUT, SIZE
    environment = input("Dossier à traiter [pilot/production] : ").strip().lower()
    if environment not in {"pilot", "production"}:
        raise ValueError("Dossier attendu : pilot ou production.")
    SOURCE = ROOT / "data" / "04_translated_fr" / environment / "train_spider.jsonl"
    WORK = ROOT / "data" / "05_checks" / environment
    STATE = WORK / "judge_batch_state.json"
    OUTPUT = WORK / "sol_judgments.jsonl"
    SIZE = 100 if environment == "pilot" else None
    if not STATE.is_file():
        if input("Préparer les deux lots du juge ? [o/N] ").strip().lower() in {"o", "oui"}:
            prepare()
            if input("Envoyer ces lots ? [o/N] ").strip().lower() in {"o", "oui"}:
                submit()
        return
    batches = status()
    if all(batch.get("status") == "completed" for batch in batches) and input("Les jugements sont prêts. Les récupérer ? [o/N] ").strip().lower() in {"o", "oui"}:
        collect()


if __name__ == "__main__":
    main()
