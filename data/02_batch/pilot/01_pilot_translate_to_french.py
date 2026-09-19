"""Lance et récupère un pilote Batch Mistral de 100 traductions vers le français."""

import argparse
import json
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PROJECT_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_DIR / ".env"
SOURCE_PATH = PROJECT_DIR / "data" / "01_processed" / "train_spider.jsonl"
PILOT_DIR = PROJECT_DIR / "data" / "02_batch" / "pilot"
STATE_PATH = PILOT_DIR / "batch_state.json"
RAW_OUTPUT_DIR = PROJECT_DIR / "data" / "03_mistral_response" / "pilot"
OUTPUT_DIR = PROJECT_DIR / "data" / "04_translated_fr" / "pilot"
API_BASE_URL = "https://api.mistral.ai/v1"
MODEL = "zai-glm-5-3"
PILOT_SIZE = 100
BATCH_SIZE = 50
MAX_OUTPUT_TOKENS = 2048
OBJECTIVE = (
    "Traduire directement avec Z.ai GLM 5.3, hébergé par Mistral, 100 questions anglaises "
    "Spider en français naturel sans changer le sens SQL, en utilisant le SQL et le schéma "
    "comme garde-fous."
)
SYSTEM_PROMPT = """Tu es chargé de traduire en français des questions anglaises issues d'un dataset text-to-SQL.

Ton unique tâche est de produire une traduction française naturelle, grammaticalement correcte et fluide, SANS modifier le sens de la question anglaise.

Règles impératives :

1. Conserve exactement le sens de la question originale.
2. Ne change aucune contrainte logique.
3. Ne change jamais :
   - les nombres ;
   - les dates ;
   - les pourcentages ;
   - les noms propres ;
   - les comparaisons ;
   - les négations ;
   - les superlatifs ;
   - les notions de minimum ou maximum ;
   - les conditions d'ordre ou de classement.
4. Respecte exactement les distinctions suivantes :
   - « plus de X » ≠ « au moins X »
   - « moins de X » ≠ « au plus X »
   - « avant X » ≠ « jusqu'à X »
   - « après X » ≠ « à partir de X »
5. N'ajoute aucune information qui n'est pas demandée dans la question originale.
6. Ne supprime aucune information.
7. Ne réponds jamais à la question.
8. Ne produis jamais de SQL.
9. Le SQL cible et le schéma sont fournis uniquement pour vérifier que ta traduction conserve le sens. Utilise-les pour traduire correctement les noms d'entités, mais ne les utilise jamais pour ajouter des précisions absentes de la question originale.
10. Ne traduis pas les requêtes SQL ; ne produis que la question française.
11. Utilise un français naturel et moderne, sans chercher à rendre la phrase inutilement sophistiquée.
12. Préserve le niveau de précision de la question originale.

Pour chaque élément, retourne uniquement un objet JSON valide de cette forme :
{"id": "<id reçu>", "question": "<question française traduite>"}

N'ajoute aucun commentaire, explication ou texte supplémentaire."""


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as source_file:
        return [json.loads(line) for line in source_file if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def batch_request(record: dict[str, object], index: int) -> dict[str, object]:
    question, sql, schema = (record.get(key) for key in ("question_original_en", "sql", "schema"))
    if not all(isinstance(value, str) and value.strip() for value in (question, sql, schema)):
        raise ValueError(f"Ligne {index + 1} : question, SQL ou schéma invalide.")
    request_id = f"train_spider:{index}"
    user_input = json.dumps(
        {"id": request_id, "question_original_en": question, "sql": sql, "schema": schema},
        ensure_ascii=False,
    )
    # Mistral Batch attend custom_id et le corps brut de chat/completions.
    return {
        "custom_id": request_id,
        "body": {
            # GLM émet une trace de raisonnement avant le JSON final. Une marge
            # généreuse évite qu'elle ne tronque la traduction finale.
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
        },
    }


def prepare() -> None:
    """Écrit deux fichiers Batch Mistral de 50 requêtes, sans les envoyer."""
    records = load_jsonl(SOURCE_PATH)[:PILOT_SIZE]
    if len(records) != PILOT_SIZE:
        raise ValueError(f"Le pilote requiert {PILOT_SIZE} exemples, {len(records)} trouvés.")
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    requests = [batch_request(record, index) for index, record in enumerate(records)]
    for number, start in enumerate(range(0, PILOT_SIZE, BATCH_SIZE), start=1):
        path = PILOT_DIR / f"mistral_requests_{number:02d}.jsonl"
        write_jsonl(path, requests[start : start + BATCH_SIZE])
        print(f"{BATCH_SIZE} requêtes écrites dans {path}")


def api_key() -> str:
    key = os.environ.get("MISTRAL_API_KEY") or read_env_api_key()
    if not key:
        raise RuntimeError("MISTRAL_API_KEY n'est pas configurée.")
    return key


def read_env_api_key() -> str | None:
    """Lit MISTRAL_API_KEY depuis .env sans l'afficher."""
    if not ENV_PATH.is_file():
        return None
    with ENV_PATH.open("r", encoding="utf-8") as env_file:
        for line in env_file:
            name, separator, value = line.strip().partition("=")
            if name == "MISTRAL_API_KEY" and separator:
                return value.strip().strip('"').strip("'") or None
    return None


def api_request(path: str, method: str = "GET", payload: object | None = None) -> dict[str, object]:
    headers = {"Authorization": f"Bearer {api_key()}"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"{API_BASE_URL}{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API Mistral : HTTP {error.code} — {details}") from error


def upload(path: Path) -> dict[str, object]:
    boundary = f"----MistralBatch{uuid.uuid4().hex}"
    file_bytes = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0] or "application/jsonl"
    body = b"".join((
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode(),
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
         f"filename=\"{path.name}\"\r\nContent-Type: {mime_type}\r\n\r\n").encode(),
        file_bytes, f"\r\n--{boundary}--\r\n".encode(),
    ))
    request = Request(
        f"{API_BASE_URL}/files", data=body,
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Téléversement Mistral : HTTP {error.code} — {details}") from error


def submit() -> None:
    """Soumet les deux lots pilotes à Mistral, chacun limité à 50 requêtes."""
    paths = [PILOT_DIR / "mistral_requests_01.jsonl", PILOT_DIR / "mistral_requests_02.jsonl"]
    if not all(path.is_file() for path in paths):
        prepare()
    batches: list[dict[str, str]] = []
    for path in paths:
        uploaded_file = upload(path)
        batch = api_request("/batch/jobs", method="POST", payload={
            "input_files": [uploaded_file["id"]], "model": MODEL,
            "endpoint": "/v1/chat/completions",
            "metadata": {"job": "spider-en-to-fr-glm-5-3-pilot", "model": MODEL},
        })
        batches.append({"batch_id": batch["id"], "input_file_id": uploaded_file["id"]})
        print(f"Batch Mistral créé : {batch['id']}")
    state = {
        "provider": "mistral", "objective": OBJECTIVE, "model": MODEL,
        "source": str(SOURCE_PATH.relative_to(PROJECT_DIR)), "examples": PILOT_SIZE,
        "requests_per_batch": BATCH_SIZE, "schema_included": True,
        "submitted_at": datetime.now(timezone.utc).isoformat(), "batches": batches,
    }
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_state() -> dict[str, object]:
    if not STATE_PATH.is_file():
        raise RuntimeError("Aucun batch pilote Mistral connu. Exécutez d'abord submit.")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("provider") != "mistral" or not isinstance(state.get("batches"), list):
        raise RuntimeError(f"État Batch Mistral invalide : {STATE_PATH}")
    return state


def status() -> list[dict[str, object]]:
    """Affiche et retourne le statut des deux Batch Mistral pilotes."""
    saved_state = load_state()
    print("Dernier envoi Batch Mistral :")
    print(f"- Objectif : {saved_state.get('objective', OBJECTIVE)}")
    print(f"- Modèle : {saved_state.get('model', MODEL)}")
    print(f"- Source : {saved_state.get('source', SOURCE_PATH.relative_to(PROJECT_DIR))}")
    print(f"- Schéma envoyé : {saved_state.get('schema_included', True)}")
    print(f"- Envoyé le : {saved_state.get('submitted_at', 'date non enregistrée')}")
    batches: list[dict[str, object]] = []
    for number, batch_state in enumerate(saved_state["batches"], start=1):
        if not isinstance(batch_state, dict):
            raise RuntimeError("Entrée Batch invalide dans l'état local.")
        batch = api_request(f"/batch/jobs/{batch_state['batch_id']}")
        print(f"- Lot {number} : {batch['id']} — {batch['status']} — "
              f"{batch.get('succeeded_requests', 0)}/{batch.get('total_requests', '?')} réussies")
        batches.append(batch)
    return batches


def extract_question(batch_line: dict[str, object]) -> tuple[str, str]:
    response_id, response = batch_line.get("custom_id"), batch_line.get("response")
    if not isinstance(response_id, str) or not isinstance(response, dict):
        raise RuntimeError(f"Réponse Batch Mistral invalide : {batch_line}")
    if response.get("status_code") not in (None, 200):
        raise RuntimeError(f"Échec d'une requête Batch Mistral : {batch_line}")
    body = response.get("body", response)
    if not isinstance(body, dict):
        raise RuntimeError(f"Corps de réponse Mistral invalide : {batch_line}")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise RuntimeError(f"Réponse chat Mistral inattendue : {body}")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"Message chat Mistral invalide : {body}")
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
        )
    else:
        text = ""
    if not text.strip():
        finish_reason = choices[0].get("finish_reason")
        raise RuntimeError(
            "Réponse GLM sans texte final "
            f"(finish_reason={finish_reason!r}). Relancez avec --resubmit."
        )
    answer = json.loads(text)
    if not isinstance(answer, dict) or set(answer) != {"id", "question"}:
        raise RuntimeError(f"Réponse JSON inattendue : {answer!r}")
    if answer["id"] != response_id or not isinstance(answer["question"], str) or not answer["question"].strip():
        raise RuntimeError(f"Réponse de traduction invalide : {answer!r}")
    return response_id, answer["question"].strip()


def download_output(file_id: str) -> str:
    request = Request(f"{API_BASE_URL}/files/{file_id}/content", headers={"Authorization": f"Bearer {api_key()}"})
    with urlopen(request) as response:
        return response.read().decode("utf-8")


def collect() -> None:
    """Construit le JSONL pilote nettoyé quand les deux lots Mistral sont terminés."""
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions: dict[str, str] = {}
    for number, batch_state in enumerate(load_state()["batches"], start=1):
        if not isinstance(batch_state, dict):
            raise RuntimeError("Entrée Batch invalide dans l'état local.")
        batch = api_request(f"/batch/jobs/{batch_state['batch_id']}")
        if batch.get("status") != "SUCCESS":
            raise RuntimeError(f"Le batch {batch['id']} n'est pas terminé avec succès : {batch.get('status')}")
        output_file_id = batch.get("output_file")
        if not isinstance(output_file_id, str):
            raise RuntimeError(f"Le batch {batch['id']} ne fournit pas de résultat.")
        raw_output = download_output(output_file_id)
        raw_path = RAW_OUTPUT_DIR / f"output_{number:02d}.jsonl"
        raw_path.write_text(raw_output, encoding="utf-8")
        print(f"Réponse Mistral brute enregistrée dans {raw_path}")
        for line in raw_output.splitlines():
            response_id, question = extract_question(json.loads(line))
            if response_id in questions:
                raise RuntimeError(f"id dupliqué dans les réponses : {response_id}")
            questions[response_id] = question
    records = load_jsonl(SOURCE_PATH)[:PILOT_SIZE]
    cleaned_records = []
    for index, record in enumerate(records):
        request_id = f"train_spider:{index}"
        if request_id not in questions:
            raise RuntimeError(f"Réponse absente pour {request_id}")
        cleaned_records.append({**record, "question": questions[request_id]})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "train_spider.jsonl"
    write_jsonl(output_path, cleaned_records)
    print(f"{len(cleaned_records)} exemples écrits dans {output_path}")


def main() -> None:
    """Affiche le dernier lot Mistral et propose l'action adaptée à son état."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resubmit",
        action="store_true",
        help="prépare et soumet de nouveaux lots, en remplaçant l'état local après confirmation",
    )
    args = parser.parse_args()
    if args.resubmit:
        if input("Préparer et soumettre deux nouveaux lots Mistral ? [o/N] ").strip().lower() in {"o", "oui"}:
            prepare()
            submit()
        return
    if not STATE_PATH.is_file():
        print("Aucun Batch pilote Mistral n'a encore été envoyé.")
        if input("Préparer les deux lots de 50 requêtes ? [o/N] ").strip().lower() in {"o", "oui"}:
            prepare()
            if input("Envoyer ces lots à Mistral ? [o/N] ").strip().lower() in {"o", "oui"}:
                submit()
        return
    batches = status()
    if all(batch.get("status") == "SUCCESS" for batch in batches):
        if input("Les résultats sont prêts. Les récupérer ? [o/N] ").strip().lower() in {"o", "oui"}:
            collect()
    else:
        print("Le traitement est toujours en cours. Relancez ce script plus tard.")


if __name__ == "__main__":
    main()
