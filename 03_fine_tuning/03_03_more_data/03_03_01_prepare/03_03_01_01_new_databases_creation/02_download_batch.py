"""Vérifie le Batch de création et télécharge sa sortie lorsqu'elle est prête."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request

from importlib.util import module_from_spec, spec_from_file_location

HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("new_database_requests", HERE / "01_create_requests.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Impossible de charger 01_create_requests.py")
module = module_from_spec(spec); spec.loader.exec_module(module)

def get(url: str, token: str) -> bytes:
    req = request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with request.urlopen(req) as response:
        return response.read()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--state", type=Path, default=HERE / "02_batch" / "batch_state.json"); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    state_path = args.state
    state = json.loads(state_path.read_text(encoding="utf-8")); token = module.api_key()
    batch = json.loads(get(f"https://api.openai.com/v1/batches/{state['batch_id']}", token))
    state["status"] = batch.get("status"); state["checked_at"] = batch.get("completed_at") or batch.get("in_progress_at")
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = batch.get("request_counts", {})
    print(f"Statut : {batch.get('status')} · {counts.get('completed', 0)}/{counts.get('total', 0)} terminées · {counts.get('failed', 0)} échouées")
    if not batch.get("output_file_id"):
        return
    output = HERE / "03_responses" / f"{state['batch_id']}_output.jsonl"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} existe déjà ; utilisez --overwrite pour le remplacer.")
    output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(get(f"https://api.openai.com/v1/files/{batch['output_file_id']}/content", token))
    state["raw_output_file"] = output.name
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Réponse téléchargée : {output}")

if __name__ == "__main__":
    main()
