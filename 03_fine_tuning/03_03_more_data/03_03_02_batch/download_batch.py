"""Affiche le statut d'un batch et télécharge son output lorsqu'il est disponible."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from urllib import request
from create_batch import key

HERE = Path(__file__).resolve().parents[1]

def get(url: str, token: str) -> bytes:
    req = request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with request.urlopen(req) as response:
        return response.read()

def ask_experiment(value: str | None) -> str:
    choice = value or input("Expérience dont télécharger le Batch [A/B/C/D] : ").strip().upper()
    if choice not in "ABCD": raise ValueError("Choix attendu : A, B, C ou D.")
    return choice

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--experiment", choices="ABCD"); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args(); args.experiment = ask_experiment(args.experiment)
    state_path = HERE / "02_batch" / "02_submissions" / f"{args.experiment}_batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")); token = key()
    batch = json.loads(get(f"https://api.openai.com/v1/batches/{state['batch_id']}", token))
    state["status"] = batch.get("status"); state["checked_at"] = batch.get("completed_at") or batch.get("in_progress_at")
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    counts = batch.get("request_counts", {})
    print(f"Statut : {batch.get('status')} · {counts.get('completed', 0)}/{counts.get('total', 0)} terminées · {counts.get('failed', 0)} échouées")
    file_id = batch.get("output_file_id")
    if not file_id:
        return
    output = HERE / "02_batch" / "03_responses" / f"{args.experiment}_{state['batch_id']}_output.jsonl"; output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        try:
            replace = input(f"La sortie brute {output.name} existe déjà. La re-télécharger ? [o/N] ").strip().lower()
        except EOFError:
            return
        if replace not in {"o", "oui"}: return
    output.write_bytes(get(f"https://api.openai.com/v1/files/{file_id}/content", token))
    state["raw_output_file"] = output.name
    raw_output_files = list(state.get("raw_output_files", []))
    if output.name not in raw_output_files: raw_output_files.append(output.name)
    state["raw_output_files"] = raw_output_files
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Réponse téléchargée : {output}")
    error_file_id = batch.get("error_file_id")
    if error_file_id:
        errors = HERE / "02_batch" / "03_responses" / f"{args.experiment}_{state['batch_id']}_errors.jsonl"
        errors.write_bytes(get(f"https://api.openai.com/v1/files/{error_file_id}/content", token))
        print(f"Erreurs téléchargées : {errors}")

if __name__ == "__main__":
    main()
