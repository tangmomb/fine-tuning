"""Prépare un seul Batch OpenAI : une requête Luna par base de données."""
from __future__ import annotations
import argparse, json, mimetypes, os, uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

HERE = Path(__file__).resolve().parents[1]; ROOT = HERE.parents[1]; DEFAULT_MODEL = "gpt-5.6-luna"; MAX_OUTPUT_TOKENS = 128000
PAIR = {"type": "object", "additionalProperties": False, "properties": {"question": {"type": "string"}, "sql": {"type": "string"}}, "required": ["question", "sql"]}
SCHEMA = {"type": "object", "additionalProperties": False, "properties": {"examples": {"type": "array", "items": PAIR}}, "required": ["examples"]}
def rows(path: Path) -> list[dict]: return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
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
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--experiment", choices="ABCD"); parser.add_argument("--model"); parser.add_argument("--submit", action="store_true"); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); args.experiment = ask_experiment(args.experiment); args.model = args.model or input(f"Modèle [{DEFAULT_MODEL}] : ").strip() or DEFAULT_MODEL
    seed_source = "existing" if args.experiment in {"A", "C"} else "new"
    seeds = rows(HERE / "01_prepare" / f"{seed_source}_sql_bases.jsonl"); output = HERE / "02_batch" / "01_requests" / f"{args.experiment}_requests.jsonl"
    if output.exists() and not args.overwrite:
        if input(f"{output.name} existe déjà. Le remplacer ? [o/N] ").strip().lower() not in {"o", "oui"}: return
    lines = []
    for row in seeds:
        difficulty = "Utilise jointures, agrégations, sous-requêtes ou opérations ensemblistes quand cela est pertinent." if args.experiment in {"C", "D"} else "Varie librement la difficulté et les constructions SQL."
        prompt = (f"Génère exactement {row['request_quota']} paires indépendantes question/SQL SQLite pour ce schéma. Chaque question est naturelle, en français ; chaque SQL est exécutable et en lecture seule. N'invente ni table ni colonne et n'ajoute aucune explication. Diversifie les tables et formulations. {difficulty}\n\nSCHÉMA ({row['db_id']}):\n{row['schema']}")
        body = {"model": args.model, "input": [{"role": "system", "content": "Tu produis des données text-to-SQL françaises fiables."}, {"role": "user", "content": prompt}], "max_output_tokens": MAX_OUTPUT_TOKENS, "text": {"format": {"type": "json_schema", "name": "synthetic_examples", "strict": True, "schema": SCHEMA}}}
        lines.append(json.dumps({"custom_id": f"{args.experiment}:{row['db_id']}", "method": "POST", "url": "/v1/responses", "body": body}, ensure_ascii=False))
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text("\n".join(lines) + "\n", encoding="utf-8"); print(f"{len(lines)} prompts (un par base) écrits dans {output}")
    if not args.submit and input("Soumettre ce Batch à OpenAI maintenant ? [o/N] ").strip().lower() not in {"o", "oui"}: return
    token = key(); uploaded = upload(output, token); batch = post("https://api.openai.com/v1/batches", {"input_file_id": uploaded["id"], "endpoint": "/v1/responses", "completion_window": "24h", "metadata": {"pipeline": "more-data", "experiment": args.experiment, "model": args.model}}, token)
    state_path = HERE / "02_batch" / "02_submissions" / f"{args.experiment}_batch_state.json"; state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"experiment": args.experiment, "model": args.model, "submitted_at": datetime.now(timezone.utc).isoformat(), "request_file": output.name, "input_file_id": uploaded["id"], "batch_id": batch["id"], "status": batch.get("status")}
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8"); print(f"Batch soumis : {batch['id']}\nEntrée envoyée : {output}")
if __name__ == "__main__": main()
