"""Prépare, soumet et récupère les traductions Mistral pour pilot ou production."""

import argparse
import json
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
MODEL, PILOT_SIZE, PILOT_BATCH_SIZE, MAX_OUTPUT_TOKENS = "zai-glm-5-3", 100, 50, 2048
SYSTEM_PROMPT = """Tu traduis en français des questions anglaises text-to-SQL. Conserve exactement le sens,
les nombres, dates, pourcentages, noms propres, comparaisons, négations, superlatifs et classements.
N'ajoute ni ne retire aucune information. Le SQL et le schéma sont des garde-fous.
Retourne uniquement la traduction française, sans JSON, SQL, commentaire ni balise Markdown."""


def choose_environment():
    environment = input("Dossier à traiter [pilot/production] : ").strip().lower()
    if environment not in {"pilot", "production"}:
        raise ValueError("Dossier attendu : pilot ou production.")
    return environment


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def key():
    value = os.environ.get("MISTRAL_API_KEY")
    env = ROOT / ".env"
    if not value and env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, separator, candidate = line.partition("=")
            if name.strip() == "MISTRAL_API_KEY" and separator:
                value = candidate.strip().strip('"').strip("'")
                break
    if not value:
        raise RuntimeError("MISTRAL_API_KEY n'est pas configurée.")
    return value


def api(path, method="GET", payload=None):
    headers, data = {"Authorization": f"Bearer {key()}"}, None
    if payload is not None:
        data, headers["Content-Type"] = json.dumps(payload).encode("utf-8"), "application/json"
    try:
        with urlopen(Request("https://api.mistral.ai/v1" + path, data=data, headers=headers, method=method)) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(f"API Mistral : HTTP {error.code} — {error.read().decode(errors='replace')}") from error


def upload(path):
    boundary = "----MistralBatch" + uuid.uuid4().hex
    mime = mimetypes.guess_type(path.name)[0] or "application/jsonl"
    body = b"".join((
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\nContent-Type: {mime}\r\n\r\n".encode(),
        path.read_bytes(), f"\r\n--{boundary}--\r\n".encode(),
    ))
    request = Request("https://api.mistral.ai/v1/files", data=body, headers={"Authorization": f"Bearer {key()}", "Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def download(file_id):
    with urlopen(Request(f"https://api.mistral.ai/v1/files/{file_id}/content", headers={"Authorization": f"Bearer {key()}"})) as response:
        return response.read().decode("utf-8")


def config(environment):
    if environment == "pilot":
        return ("train_spider",), PILOT_SIZE, PILOT_BATCH_SIZE
    return ("train_spider", "train_others", "dev"), None, 0


def request_row(record, split, index):
    fields = ("question_original_en", "sql", "schema")
    if not all(isinstance(record.get(field), str) and record[field].strip() for field in fields):
        raise ValueError(f"{split}:{index} : question, SQL ou schéma invalide.")
    identifier = f"{split}:{index}"
    return {"custom_id": identifier, "body": {"max_tokens": MAX_OUTPUT_TOKENS, "temperature": 0, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"id": identifier, **{field: record[field] for field in fields}}, ensure_ascii=False)},
    ]}}


def state_path(environment):
    return ROOT / "data" / "02_batch" / environment / "batch_state.json"


def prepare(environment):
    splits, limit, chunk_size = config(environment)
    work, batches = state_path(environment).parent, []
    for split in splits:
        requests = [request_row(row, split, index) for index, row in enumerate(load_jsonl(ROOT / "data" / "01_processed" / f"{split}.jsonl")[:limit])]
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


def submit(environment, batches, append=False):
    work, submitted = state_path(environment).parent, []
    for item in batches:
        file = upload(work / item["request_file"])
        batch = api("/batch/jobs", "POST", {"input_files": [file["id"]], "model": MODEL, "endpoint": "/v1/chat/completions", "metadata": {"job": f"spider-en-to-fr-{environment}", "split": item["split"], "model": MODEL}})
        submitted.append({**item, "batch_id": batch["id"], "input_file_id": file["id"]})
        print(f"{item['split']} : Batch Mistral créé : {batch['id']}")
    existing = load_state(environment)["batches"] if append and state_path(environment).is_file() else []
    state_path(environment).write_text(json.dumps({"environment": environment, "model": MODEL, "submitted_at": datetime.now(timezone.utc).isoformat(), "batches": [*existing, *submitted]}, indent=2) + "\n", encoding="utf-8")


def load_state(environment):
    path = state_path(environment)
    if not path.is_file():
        raise RuntimeError(f"Aucun Batch {environment} connu.")
    state = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(state.get("batches"), dict):
        state["batches"] = [{"split": split, **item} for split, item in state["batches"].items()]
    if environment == "pilot":
        for item in state.get("batches", []):
            item.setdefault("split", "train_spider")
    return state


def status(environment):
    jobs = []
    for item in load_state(environment)["batches"]:
        job = api(f"/batch/jobs/{item['batch_id']}")
        print(f"{item['split']} : {job['status']} — {job.get('succeeded_requests', 0)}/{job.get('total_requests', '?')}")
        jobs.append((item, job))
    return jobs


def extract_question(line):
    identifier = line.get("custom_id")
    if not isinstance(identifier, str) or not identifier:
        raise RuntimeError("Réponse Mistral sans custom_id.")
    body = line.get("response", {}).get("body", {})
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
    if not isinstance(content, str):
        raise RuntimeError(f"Contenu Mistral invalide : {content!r}")
    question = content.strip()
    if not question:
        raise RuntimeError("Réponse Mistral sans texte final.")
    # Compatibilité avec les lots déjà soumis avant le passage en texte simple.
    if question.startswith("{"):
        answer = json.loads(question)
        if not isinstance(answer.get("id"), str) or not isinstance(answer.get("question"), str):
            raise RuntimeError(f"Réponse JSON Mistral invalide : {answer!r}")
        if answer["id"] != identifier:
            raise RuntimeError(f"id JSON incohérent : {answer['id']} au lieu de {identifier}.")
        question = answer["question"].strip()
    return identifier, question


def missing_response_ids(environment):
    missing, completed = set(), set()
    raw_dir = ROOT / "data" / "03_mistral_response" / environment
    for path in raw_dir.glob("output_*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            response = json.loads(line)
            try:
                identifier, _ = extract_question(response)
                completed.add(identifier)
            except (json.JSONDecodeError, RuntimeError):
                identifier = response.get("custom_id")
                if isinstance(identifier, str):
                    missing.add(identifier)
    return sorted(missing - completed)


def retry_missing(environment):
    identifiers = missing_response_ids(environment)
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


def collect(environment, jobs):
    questions, invalid_responses = {}, []
    for number, (item, job) in enumerate(jobs, 1):
        if job.get("status") != "SUCCESS" or not isinstance(job.get("output_file"), str):
            raise RuntimeError(f"{item['split']} n'est pas prêt : {job.get('status')}")
        raw = download(job["output_file"])
        raw_path = ROOT / "data" / "03_mistral_response" / environment / f"output_{number:02d}.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw, encoding="utf-8")
        for line in raw.splitlines():
            response = json.loads(line)
            try:
                identifier, question = extract_question(response)
            except (json.JSONDecodeError, RuntimeError) as error:
                invalid_responses.append((response.get("custom_id"), str(error)))
                continue
            split, _ = identifier.split(":", 1)
            questions.setdefault(split, {})[identifier] = question
    unresolved = []
    for identifier, error in invalid_responses:
        if not isinstance(identifier, str) or ":" not in identifier:
            unresolved.append(f"{identifier or '<id inconnu>'} ({error})")
            continue
        split, _ = identifier.split(":", 1)
        if identifier not in questions.get(split, {}):
            unresolved.append(f"{identifier} ({error})")
    if unresolved:
        details = "; ".join(unresolved[:20])
        suffix = "" if len(unresolved) <= 20 else f" ; … ({len(unresolved)} au total)"
        raise RuntimeError(f"{len(unresolved)} réponses Mistral ne contiennent pas de JSON final : {details}{suffix}")
    splits, limit, _ = config(environment)
    for split in splits:
        records = load_jsonl(ROOT / "data" / "01_processed" / f"{split}.jsonl")[:limit]
        missing = [f"{split}:{index}" for index in range(len(records)) if f"{split}:{index}" not in questions.get(split, {})]
        if missing:
            raise RuntimeError(f"Traductions absentes pour {split} : {', '.join(missing[:20])}")
        translated = [{**record, "question": questions[split][f"{split}:{index}"]} for index, record in enumerate(records)]
        output = ROOT / "data" / "04_translated_fr" / environment / f"{split}.jsonl"
        write_jsonl(output, translated)
        print(f"{split} : {len(translated)} traductions écrites dans {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resubmit", action="store_true", help="prépare et soumet de nouveaux lots après confirmation")
    parser.add_argument("--retry-missing", action="store_true", help="renvoie uniquement les réponses sans JSON final déjà téléchargées")
    args = parser.parse_args()
    environment = choose_environment()
    if args.resubmit:
        if input("Préparer et soumettre de nouveaux lots Mistral ? [o/N] ").strip().lower() in {"o", "oui"}:
            submit(environment, prepare(environment))
        return
    if args.retry_missing:
        if input("Renvoyer uniquement les réponses incomplètes déjà téléchargées ? [o/N] ").strip().lower() in {"o", "oui"}:
            retry_missing(environment)
        return
    if not state_path(environment).is_file():
        if input("Préparer et envoyer les lots Mistral ? [o/N] ").strip().lower() in {"o", "oui"}:
            submit(environment, prepare(environment))
        return
    jobs = status(environment)
    if all(job.get("status") == "SUCCESS" for _, job in jobs) and input("Les traductions sont prêtes. Les récupérer ? [o/N] ").strip().lower() in {"o", "oui"}:
        collect(environment, jobs)


if __name__ == "__main__":
    main()
