"""Juge sémantiquement les paires déterministiquement valides, une base par requête Batch."""
from __future__ import annotations

import argparse, json, mimetypes, os, random, uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
MODEL = "gpt-5.6-terra"
SAMPLE_SIZE = 200
RANDOM_SEED = 20260928
VERDICT = {"type": "object", "additionalProperties": False, "properties": {"id": {"type": "string"}, "verdict": {"type": "string", "enum": ["pass", "fail", "uncertain"]}, "issues": {"type": "array", "items": {"type": "string"}}}, "required": ["id", "verdict", "issues"]}
SCHEMA = {"type": "object", "additionalProperties": False, "properties": {"verdicts": {"type": "array", "items": VERDICT}}, "required": ["verdicts"]}

def read(path: Path) -> list[dict]: return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
def ask_experiment(value: str | None) -> str:
    choice = value or input("Expérience à juger [A/B/C/D] : ").strip().upper()
    if choice not in "ABCD": raise ValueError("Choix attendu : A, B, C ou D.")
    return choice
def api_key() -> str:
    if os.environ.get("OPENAI_API_KEY"): return os.environ["OPENAI_API_KEY"]
    for env_path in (ROOT / ".env", ROOT.parent / ".env", ROOT.parents[1] / ".env"):
        if not env_path.is_file(): continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="): return line.partition("=")[2].strip().strip('"')
    raise RuntimeError("OPENAI_API_KEY est absent.")
def get(url: str, token: str) -> bytes:
    req = request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with request.urlopen(req) as response: return response.read()
def post(url: str, payload: dict, token: str) -> dict:
    req = request.Request(url, data=json.dumps(payload).encode(), method="POST", headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with request.urlopen(req) as response: return json.load(response)
def upload(path: Path, token: str) -> dict:
    boundary = f"----judge{uuid.uuid4().hex}"; data = bytearray()
    data.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'.encode())
    data.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\nContent-Type: {mimetypes.guess_type(path.name)[0] or "application/jsonl"}\r\n\r\n'.encode()); data.extend(path.read_bytes()); data.extend(f"\r\n--{boundary}--\r\n".encode())
    req = request.Request("https://api.openai.com/v1/files", data=bytes(data), method="POST", headers={"Authorization": f"Bearer {token}", "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with request.urlopen(req) as response: return json.load(response)
def output_text(response: dict) -> str | None:
    for item in response.get("response", {}).get("body", {}).get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str): return content["text"]
    return None

def stratified_sample(candidates: list[dict], size: int, rng: random.Random) -> list[dict]:
    """Sample proportionally by difficulty and distribute each quota across databases."""
    target = min(size, len(candidates))
    by_hardness: dict[str, list[dict]] = defaultdict(list)
    for row in candidates: by_hardness[row["hardness"]].append(row)
    total = len(candidates)
    quotas = {hardness: target * len(rows) // total for hardness, rows in by_hardness.items()}
    remaining = target - sum(quotas.values())
    fractions = sorted(
        by_hardness,
        key=lambda hardness: (target * len(by_hardness[hardness]) % total, hardness),
        reverse=True,
    )
    for hardness in fractions[:remaining]: quotas[hardness] += 1

    sample = []
    for hardness, quota in quotas.items():
        by_database: dict[str, list[dict]] = defaultdict(list)
        for row in by_hardness[hardness]: by_database[row["db_id"]].append(row)
        for rows in by_database.values(): rng.shuffle(rows)
        while quota:
            database_ids = [db_id for db_id, rows in by_database.items() if rows]
            rng.shuffle(database_ids)
            for db_id in database_ids:
                if not quota: break
                sample.append(by_database[db_id].pop())
                quota -= 1
    rng.shuffle(sample)
    return sample

def prepare(experiment: str, model: str, exhaustive: bool = False) -> Path:
    source = ROOT / "03_quality" / "01_deterministic_checks" / experiment / "selected.jsonl"
    grouped: dict[str, list[dict]] = defaultdict(list)
    candidates = read(source)
    audited = candidates if exhaustive else stratified_sample(candidates, SAMPLE_SIZE, random.Random(RANDOM_SEED))
    audit_name = "audit" if exhaustive else "sample"
    write(WORK / "01_requests" / f"{experiment}_{audit_name}.jsonl", audited)
    for row in audited: grouped[row["db_id"]].append(row)
    lines = []
    for db_id, examples in sorted(grouped.items()):
        prompt = ("Évalue chaque paire question/SQL pour ce schéma SQLite. Une paire est pass seulement si la question française décrit exactement le SQL ; "
                  "fail si elle ajoute, retire ou modifie une contrainte ; uncertain si tu ne peux pas décider. Vérifie agrégations, filtres, jointures, négations, tris, limites et nombres.\n\n"
                  + json.dumps({"schema": examples[0]["schema"], "examples": [{"id": row["id"], "question": row["question"], "sql": row["sql"]} for row in examples]}, ensure_ascii=False))
        body = {"model": model, "input": [{"role": "system", "content": "Tu es un juge qualité text-to-SQL strict et indépendant."}, {"role": "user", "content": prompt}], "max_output_tokens": 16000, "text": {"format": {"type": "json_schema", "name": "semantic_verdicts", "strict": True, "schema": SCHEMA}}}
        lines.append(json.dumps({"custom_id": f"judge:{experiment}:{db_id}", "method": "POST", "url": "/v1/responses", "body": body}, ensure_ascii=False))
    path = WORK / "01_requests" / f"{experiment}_judge_requests.jsonl"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    scope = "exemples à vérifier exhaustivement" if exhaustive else "exemples échantillonnés"
    print(f"{len(audited)} {scope} dans {len(lines)} requêtes de jugement : {path}"); return path

def submit(experiment: str, model: str, path: Path, exhaustive: bool) -> None:
    token = api_key(); uploaded = upload(path, token); batch = post("https://api.openai.com/v1/batches", {"input_file_id": uploaded["id"], "endpoint": "/v1/responses", "completion_window": "24h", "metadata": {"pipeline": "more-data-semantic-judge", "experiment": experiment, "model": model}}, token)
    state = {"experiment": experiment, "model": model, "exhaustive": exhaustive, "submitted_at": datetime.now(timezone.utc).isoformat(), "request_file": path.name, "input_file_id": uploaded["id"], "batch_id": batch["id"], "status": batch.get("status")}
    state_path = WORK / "02_submissions" / f"{experiment}_judge_state.json"; state_path.parent.mkdir(parents=True, exist_ok=True); state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8"); print(f"Batch de jugement soumis : {batch['id']}")

def collect(experiment: str, state_path: Path, state: dict, batch: dict, token: str) -> None:
    raw = WORK / "03_responses" / f"{experiment}_{state['batch_id']}_judge_output.jsonl"; raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists(): raw.write_bytes(get(f"https://api.openai.com/v1/files/{batch['output_file_id']}/content", token))
    verdicts = {}
    for line in read(raw):
        text = output_text(line)
        try: answers = json.loads(text or "")["verdicts"]
        except (json.JSONDecodeError, KeyError, TypeError): continue
        for answer in answers:
            if answer.get("verdict") in {"pass", "fail", "uncertain"} and isinstance(answer.get("id"), str) and isinstance(answer.get("issues"), list): verdicts[answer["id"]] = answer
    audit_name = "audit" if state.get("exhaustive") else "sample"
    audited = read(WORK / "01_requests" / f"{experiment}_{audit_name}.jsonl")
    judged = [{**row, **verdicts.get(row["id"], {"verdict": "uncertain", "issues": ["Verdict absent."]})} for row in audited]
    output_dir = WORK / "04_judgments"
    judged_name = "judged" if state.get("exhaustive") else "judged_sample"
    write(output_dir / f"{experiment}_{judged_name}.jsonl", judged)
    manifest = {"experiment": experiment, "exhaustive": bool(state.get("exhaustive")), "audited_size": len(judged), "sample_seed": None if state.get("exhaustive") else RANDOM_SEED, "verdicts": dict(Counter(row["verdict"] for row in judged)), "raw_response": raw.name}; (output_dir / f"{experiment}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state["status"] = "collected"; state["raw_output_file"] = raw.name; state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8"); print(f"{len(judged)} jugements écrits dans {output_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--experiment", choices="ABCD"); parser.add_argument("--model"); parser.add_argument("--submit", action="store_true"); parser.add_argument("--all", action="store_true", help="Juge exhaustivement les exemples sélectionnés."); args = parser.parse_args()
    experiment = ask_experiment(args.experiment); model = args.model or input(f"Modèle juge [{MODEL}] : ").strip() or MODEL; state_path = WORK / "02_submissions" / f"{experiment}_judge_state.json"
    if not state_path.is_file():
        path = prepare(experiment, model, args.all)
        if args.submit or input("Soumettre le Batch de jugement à OpenAI ? [o/N] ").strip().lower() in {"o", "oui"}: submit(experiment, model, path, args.all)
        return
    token = api_key(); state = json.loads(state_path.read_text(encoding="utf-8")); batch = json.loads(get(f"https://api.openai.com/v1/batches/{state['batch_id']}", token)); counts = batch.get("request_counts", {})
    print(f"Statut : {batch.get('status')} · {counts.get('completed', 0)}/{counts.get('total', 0)} terminées · {counts.get('failed', 0)} échouées")
    if batch.get("status") == "completed" and input("Télécharger et consolider les jugements ? [o/N] ").strip().lower() in {"o", "oui"}: collect(experiment, state_path, state, batch, token)

if __name__ == "__main__": main()
