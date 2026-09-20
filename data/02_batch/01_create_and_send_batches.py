"""Crée et envoie les lots de traduction Mistral pour pilot ou production."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from translation_common import MODEL, ROOT, choose_environment, config, extract_question, load_jsonl, load_state, normalize_identifier, request_row, state_path, upload, api, write_jsonl


def prepare(environment: str) -> list[dict]:
    splits, limit, chunk_size = config(environment)
    work, batches = state_path(environment).parent, []
    for split in splits:
        records = load_jsonl(ROOT / "data" / "01_processed" / f"{split}.jsonl")[:limit]
        requests = [request_row(row, split, index) for index, row in enumerate(records)]
        if not requests:
            raise ValueError(f"Aucune donnée source pour {split}.")
        chunks = [requests] if not chunk_size else [requests[start:start + chunk_size] for start in range(0, len(requests), chunk_size)]
        for number, chunk in enumerate(chunks, 1):
            suffix = f"_{number:02d}" if len(chunks) > 1 else ""
            path = work / f"{split}_requests{suffix}.jsonl"
            write_jsonl(path, chunk)
            batches.append({"split": split, "request_file": path.name})
            print(f"{split} : {len(chunk)} requêtes écrites dans {path}")
    return batches


def submit(environment: str, batches: list[dict], append: bool) -> None:
    work, submitted = state_path(environment).parent, []
    for item in batches:
        file = upload(work / item["request_file"])
        batch = api("/batch/jobs", "POST", {"input_files": [file["id"]], "model": MODEL, "endpoint": "/v1/chat/completions", "metadata": {"job": f"spider-en-to-fr-{environment}", "split": item["split"], "model": MODEL}})
        submitted.append({**item, "batch_id": batch["id"], "input_file_id": file["id"]})
        print(f"{item['split']} : Batch Mistral créé : {batch['id']}")
    existing = load_state(environment)["batches"] if append else []
    state_path(environment).write_text(json.dumps({"environment": environment, "model": MODEL, "submitted_at": datetime.now(timezone.utc).isoformat(), "batches": [*existing, *submitted]}, indent=2) + "\n", encoding="utf-8")


def retry_missing(environment: str) -> None:
    missing, completed = set(), set()
    raw_dir = ROOT / "data" / "03_mistral_response" / environment
    for number, item in enumerate(load_state(environment)["batches"], 1):
        path = raw_dir / f"output_{number:02d}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            response = json.loads(line)
            try:
                identifier, _ = extract_question(response)
                completed.add(normalize_identifier(identifier, item["split"]))
            except (json.JSONDecodeError, RuntimeError):
                identifier = response.get("custom_id")
                if isinstance(identifier, str):
                    missing.add(normalize_identifier(identifier, item["split"]))
    overrides_path = state_path(environment).parent / "manual_translations.jsonl"
    overridden = {row.get("id") for row in load_jsonl(overrides_path) if isinstance(row.get("id"), str)} if overrides_path.is_file() else set()
    identifiers = sorted(missing - completed - overridden)
    if not identifiers:
        raise RuntimeError("Aucune réponse incomplète n'a été trouvée dans les sorties brutes téléchargées.")
    rows_by_split = {}
    for identifier in identifiers:
        split, separator, index_text = identifier.partition(":")
        if not separator or not index_text.isdigit():
            raise RuntimeError(f"Identifiant de réponse invalide : {identifier}")
        rows_by_split.setdefault(split, load_jsonl(ROOT / "data" / "01_processed" / f"{split}.jsonl"))
    requests = [request_row(rows_by_split[identifier.partition(":")[0]][int(identifier.partition(":")[2])], identifier.partition(":")[0], int(identifier.partition(":")[2])) for identifier in identifiers]
    path = state_path(environment).parent / "retry_missing_requests.jsonl"
    write_jsonl(path, requests)
    print(f"{len(requests)} requêtes incomplètes écrites dans {path}")
    submit(environment, [{"split": "retry_missing", "request_file": path.name}], append=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resubmit", action="store_true", help="ajoute de nouveaux lots à l'état existant")
    parser.add_argument("--retry-missing", action="store_true", help="renvoie les réponses invalides des sorties brutes téléchargées")
    args = parser.parse_args()
    environment = choose_environment()
    if args.retry_missing:
        if input("Renvoyer les réponses incomplètes déjà téléchargées ? [o/N] ").strip().lower() in {"o", "oui"}:
            retry_missing(environment)
        return
    exists = state_path(environment).is_file()
    if exists and not args.resubmit:
        raise RuntimeError("Des lots existent déjà. Utilisez --resubmit pour en créer de nouveaux.")
    if input("Préparer et envoyer les lots Mistral ? [o/N] ").strip().lower() not in {"o", "oui"}:
        return
    submit(environment, prepare(environment), append=exists)


if __name__ == "__main__":
    main()
