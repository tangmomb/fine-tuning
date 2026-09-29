"""Corrige en Batch les questions rejetées par le juge sémantique."""
from __future__ import annotations

import argparse, json, mimetypes, os, uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
MODEL = "gpt-5.6-terra"
CORRECTION = {
    "type": "object", "additionalProperties": False,
    "properties": {"id": {"type": "string"}, "question": {"type": "string"}},
    "required": ["id", "question"],
}
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"corrections": {"type": "array", "items": CORRECTION}},
    "required": ["corrections"],
}

def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

def api_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    for env_path in (ROOT / ".env", ROOT.parent / ".env", ROOT.parents[1] / ".env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.partition("=")[2].strip().strip('"')
    raise RuntimeError("OPENAI_API_KEY est absent.")

def get(url: str, token: str) -> bytes:
    req = request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with request.urlopen(req) as response:
        return response.read()

def post(url: str, payload: dict, token: str) -> dict:
    req = request.Request(url, data=json.dumps(payload).encode(), method="POST", headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with request.urlopen(req) as response:
        return json.load(response)

def upload(path: Path, token: str) -> dict:
    boundary = f"----correction{uuid.uuid4().hex}"
    data = bytearray()
    data.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode())
    data.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\nContent-Type: {mimetypes.guess_type(path.name)[0] or "application/jsonl"}\r\n\r\n'.encode())
    data.extend(path.read_bytes())
    data.extend(f"\r\n--{boundary}--\r\n".encode())
    req = request.Request("https://api.openai.com/v1/files", data=bytes(data), method="POST", headers={"Authorization": f"Bearer {token}", "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with request.urlopen(req) as response:
        return json.load(response)

def output_text(response: dict) -> str | None:
    for item in response.get("response", {}).get("body", {}).get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    return None

def prepare(experiment: str, model: str, retry: bool = False) -> Path:
    source = (ROOT / "03_03_03_quality" / "03_03_03_02_semantic_judge" / "03_03_03_02_06_semantic_rejudge" / "03_03_03_02_06_04_rejudgments" / f"{experiment}_rejudged.jsonl") if retry else (ROOT / "03_03_03_quality" / "03_03_03_02_semantic_judge" / "03_03_03_02_04_judgments" / f"{experiment}_judged.jsonl")
    verdict_key, issues_key = ("recheck_verdict", "recheck_issues") if retry else ("verdict", "issues")
    rejected = [row for row in read(source) if row[verdict_key] in {"fail", "uncertain"}]
    if not rejected:
        raise RuntimeError("Aucun exemple fail ou uncertain à corriger.")
    prefix = "retry_" if retry else ""
    write(WORK / "03_03_03_02_05_01_requests" / f"{experiment}_{prefix}to_correct.jsonl", rejected)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rejected:
        grouped[row["db_id"]].append(row)
    lines = []
    for db_id, examples in sorted(grouped.items()):
        payload = {
            "schema": examples[0]["schema"],
            "examples": [{"id": row["id"], "question_originale": row["question"], "sql": row["sql"], "diagnostic": row[issues_key]} for row in examples],
        }
        prompt = (
            "Réécris chaque question française pour qu'elle décrive exactement le SQL associé. "
            "Le SQL est la vérité : ne le modifie jamais. Utilise le schéma et le diagnostic du juge. "
            "Conserve une formulation naturelle, sans mentionner le diagnostic, le SQL ni les tables si cela n'est pas nécessaire. "
            "Retourne une correction pour chaque id, y compris si le diagnostic semble discutable.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        body = {"model": model, "input": [{"role": "system", "content": "Tu corriges strictement des paires text-to-SQL en français."}, {"role": "user", "content": prompt}], "max_output_tokens": 8000, "text": {"format": {"type": "json_schema", "name": "semantic_corrections", "strict": True, "schema": SCHEMA}}}
        lines.append(json.dumps({"custom_id": f"correct:{experiment}:{db_id}", "method": "POST", "url": "/v1/responses", "body": body}, ensure_ascii=False))
    path = WORK / "03_03_03_02_05_01_requests" / f"{experiment}_{prefix}correction_requests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rejected)} exemples à corriger dans {len(lines)} requêtes : {path}")
    return path

def submit(experiment: str, model: str, path: Path, retry: bool = False) -> None:
    token = api_key()
    uploaded = upload(path, token)
    batch = post("https://api.openai.com/v1/batches", {"input_file_id": uploaded["id"], "endpoint": "/v1/responses", "completion_window": "24h", "metadata": {"pipeline": "more-data-semantic-correction", "experiment": experiment, "model": model}}, token)
    state = {"experiment": experiment, "model": model, "retry": retry, "submitted_at": datetime.now(timezone.utc).isoformat(), "request_file": path.name, "input_file_id": uploaded["id"], "batch_id": batch["id"], "status": batch.get("status")}
    prefix = "retry_" if retry else ""
    state_path = WORK / "03_03_03_02_05_02_submissions" / f"{experiment}_{prefix}correction_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Batch de correction soumis : {batch['id']}")

def collect(experiment: str, state_path: Path, state: dict, batch: dict, token: str) -> None:
    raw = WORK / "03_03_03_02_05_03_responses" / f"{experiment}_{state['batch_id']}_correction_output.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        raw.write_bytes(get(f"https://api.openai.com/v1/files/{batch['output_file_id']}/content", token))
    corrections = {}
    for line in read(raw):
        try:
            answers = json.loads(output_text(line) or "")["corrections"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        for answer in answers:
            if isinstance(answer.get("id"), str) and isinstance(answer.get("question"), str) and answer["question"].strip():
                corrections[answer["id"]] = answer["question"].strip()
    prefix = "retry_" if state.get("retry") else ""
    originals = read(WORK / "03_03_03_02_05_01_requests" / f"{experiment}_{prefix}to_correct.jsonl")
    corrected = [{**row, "question_originale": row.get("question_originale", row["question"]), "question": corrections.get(row["id"], row["question"]), "correction_status": "corrected" if row["id"] in corrections else "missing"} for row in originals]
    output_dir = WORK / "03_03_03_02_05_04_corrections"
    write(output_dir / f"{experiment}_{prefix}corrected.jsonl", corrected)
    manifest = {"experiment": experiment, "input_count": len(originals), "correction_status": dict(Counter(row["correction_status"] for row in corrected)), "raw_response": raw.name}
    (output_dir / f"{experiment}_{prefix}manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state["status"] = "collected"
    state["raw_output_file"] = raw.name
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"{len(corrected)} corrections écrites dans {output_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices="ABCD")
    parser.add_argument("--model")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--retry", action="store_true", help="Corrige les échecs du re-jugement.")
    args = parser.parse_args()
    experiment = args.experiment or input("Expérience à corriger [A/B/C/D] : ").strip().upper()
    if experiment not in "ABCD":
        raise ValueError("Choix attendu : A, B, C ou D.")
    model = args.model or input(f"Modèle correcteur [{MODEL}] : ").strip() or MODEL
    prefix = "retry_" if args.retry else ""
    state_path = WORK / "03_03_03_02_05_02_submissions" / f"{experiment}_{prefix}correction_state.json"
    if not state_path.is_file():
        path = prepare(experiment, model, args.retry)
        if args.submit or input("Soumettre le Batch de correction à OpenAI ? [o/N] ").strip().lower() in {"o", "oui"}:
            submit(experiment, model, path, args.retry)
        return
    token = api_key()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    batch = json.loads(get(f"https://api.openai.com/v1/batches/{state['batch_id']}", token))
    counts = batch.get("request_counts", {})
    print(f"Statut : {batch.get('status')} · {counts.get('completed', 0)}/{counts.get('total', 0)} terminées · {counts.get('failed', 0)} échouées")
    if batch.get("status") == "completed" and input("Télécharger et consolider les corrections ? [o/N] ").strip().lower() in {"o", "oui"}:
        collect(experiment, state_path, state, batch, token)

if __name__ == "__main__":
    main()
