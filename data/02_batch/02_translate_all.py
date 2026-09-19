"""Prépare, soumet et récupère les traductions GLM de tous les splits Spider."""

import json
import runpy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PILOT = runpy.run_path(ROOT / "data" / "02_batch" / "pilot" / "01_pilot_translate_to_french.py")
SPLITS = ("train_spider", "train_others", "dev")
REQUEST_DIR = ROOT / "data" / "02_batch" / "production"
STATE = REQUEST_DIR / "batch_state.json"
OUTPUT_DIR = ROOT / "data" / "04_translated_fr" / "production"
RAW_DIR = ROOT / "data" / "03_mistral_response" / "production"


def prepare():
    REQUEST_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        records = PILOT["load_jsonl"](ROOT / "data" / "01_processed" / f"{split}.jsonl")
        requests = [PILOT["batch_request"](record, index) for index, record in enumerate(records)]
        path = REQUEST_DIR / f"{split}_requests.jsonl"
        PILOT["write_jsonl"](path, requests)
        print(f"{split} : {len(requests)} requêtes écrites dans {path}")


def submit():
    if not all((REQUEST_DIR / f"{split}_requests.jsonl").is_file() for split in SPLITS):
        prepare()
    batches = {}
    for split in SPLITS:
        file = PILOT["upload"](REQUEST_DIR / f"{split}_requests.jsonl")
        batch = PILOT["api_request"]("/batch/jobs", "POST", {
            "input_files": [file["id"]], "model": PILOT["MODEL"],
            "endpoint": "/v1/chat/completions",
            "metadata": {"job": "spider-en-to-fr-production", "split": split, "model": PILOT["MODEL"]},
        })
        batches[split] = {"batch_id": batch["id"], "input_file_id": file["id"]}
        print(f"{split} : Batch Mistral créé : {batch['id']}")
    STATE.write_text(json.dumps({"model": PILOT["MODEL"], "submitted_at": datetime.now(timezone.utc).isoformat(), "batches": batches}, indent=2) + "\n", encoding="utf-8")


def load_state():
    if not STATE.is_file():
        raise RuntimeError("Aucun Batch de production connu.")
    return json.loads(STATE.read_text(encoding="utf-8"))


def status():
    jobs = {}
    for split, saved in load_state()["batches"].items():
        job = PILOT["api_request"](f"/batch/jobs/{saved['batch_id']}")
        print(f"{split} : {job['status']} — {job.get('succeeded_requests', 0)}/{job.get('total_requests', '?')}")
        jobs[split] = job
    return jobs


def collect():
    for split, job in status().items():
        if job.get("status") != "SUCCESS" or not isinstance(job.get("output_file"), str):
            raise RuntimeError(f"{split} n'est pas prêt : {job.get('status')}")
        raw = PILOT["download_output"](job["output_file"])
        raw_path = RAW_DIR / f"{split}.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw, encoding="utf-8")
        questions = {}
        for line in raw.splitlines():
            identifier, question = PILOT["extract_question"](json.loads(line))
            questions[identifier] = question
        records = PILOT["load_jsonl"](ROOT / "data" / "01_processed" / f"{split}.jsonl")
        translated = [{**record, "question": questions[f"train_spider:{index}"]} for index, record in enumerate(records)]
        output = OUTPUT_DIR / f"{split}.jsonl"
        PILOT["write_jsonl"](output, translated)
        print(f"{split} : {len(translated)} traductions écrites dans {output}")


def main():
    if not STATE.is_file():
        if input("Préparer et envoyer les trois lots Mistral ? [o/N] ").strip().lower() in {"o", "oui"}:
            prepare()
            submit()
        return
    jobs = status()
    if all(job.get("status") == "SUCCESS" for job in jobs.values()):
        if input("Les traductions sont prêtes. Les récupérer ? [o/N] ").strip().lower() in {"o", "oui"}:
            collect()


if __name__ == "__main__":
    main()
