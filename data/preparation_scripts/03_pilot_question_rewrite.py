"""Lance et récupère un pilote Batch de 100 reformulations Spider-FR."""

import argparse
import json
import mimetypes
import os
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_DIR / ".env"
SOURCE_PATH = PROJECT_DIR / "data" / "01_processed" / "train_spider.jsonl"
PILOT_DIR = PROJECT_DIR / "data" / "02_batch" / "pilot"
STATE_PATH = PILOT_DIR / "batch_state.json"
RAW_OUTPUT_DIR = PROJECT_DIR / "data" / "03_open_ai_response" / "pilot"
OUTPUT_DIR = PROJECT_DIR / "data" / "04_cleaned" / "pilot"
MODEL = "gpt-5.6-luna"
PILOT_SIZE = 100
BATCH_SIZE = 50
SYSTEM_PROMPT = """Tu es chargé de corriger linguistiquement des questions françaises issues d'un dataset text-to-SQL.

Ton unique tâche est de reformuler chaque question en français naturel, grammaticalement correct et fluide, SANS modifier son sens.

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
9. Le SQL cible et le schéma sont fournis uniquement pour vérifier que ta reformulation conserve le sens. Utilise-les pour corriger les traductions littérales manifestement absurdes de noms d'entités, mais ne les utilise jamais pour ajouter des précisions absentes de la question originale.
10. Si la question est déjà correcte et naturelle, conserve-la avec seulement les éventuelles corrections typographiques nécessaires.
11. Utilise un français naturel et moderne, sans chercher à rendre la phrase inutilement sophistiquée.
12. Préserve le niveau de précision de la question originale.

Pour chaque élément, retourne uniquement un objet JSON valide de cette forme :
{"id": "<id reçu>", "question": "<question française corrigée>"}

N'ajoute aucun commentaire, explication ou texte supplémentaire."""


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as source_file:
        return [json.loads(line) for line in source_file if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def batch_request(record: dict[str, object], index: int) -> dict[str, object]:
    question = record.get("question_original")
    sql = record.get("sql")
    schema = record.get("schema")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Ligne {index + 1} : question_original invalide.")
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError(f"Ligne {index + 1} : sql invalide.")
    if not isinstance(schema, str) or not schema.strip():
        raise ValueError(f"Ligne {index + 1} : schema invalide.")

    request_id = f"train_spider:{index}"
    user_input = {
        "id": request_id,
        "question_original": question,
        "sql": sql,
        "schema": schema,
    }
    return {
        "custom_id": request_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "reasoning": {"effort": "none"},
            "instructions": SYSTEM_PROMPT,
            "input": json.dumps(user_input, ensure_ascii=False),
            "max_output_tokens": 128,
        },
    }


def prepare() -> None:
    """Écrit deux fichiers Batch de 50 requêtes, sans les envoyer."""
    records = load_jsonl(SOURCE_PATH)[:PILOT_SIZE]
    if len(records) != PILOT_SIZE:
        raise ValueError(f"Le pilote requiert {PILOT_SIZE} exemples, {len(records)} trouvés.")

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    requests = [batch_request(record, index) for index, record in enumerate(records)]
    for batch_number, start in enumerate(range(0, PILOT_SIZE, BATCH_SIZE), start=1):
        batch_path = PILOT_DIR / f"requests_{batch_number:02d}.jsonl"
        write_jsonl(batch_path, requests[start : start + BATCH_SIZE])
        print(f"{BATCH_SIZE} requêtes écrites dans {batch_path}")


def api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY") or read_env_api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY n'est pas configurée.")
    return key


def read_env_api_key() -> str | None:
    """Lit OPENAI_API_KEY depuis le fichier .env local, sans l'afficher."""
    if not ENV_PATH.is_file():
        return None
    with ENV_PATH.open("r", encoding="utf-8") as env_file:
        for line in env_file:
            name, separator, value = line.strip().partition("=")
            if name == "OPENAI_API_KEY" and separator:
                return value.strip().strip('"').strip("'") or None
    return None


def api_request(path: str, method: str = "GET", payload: object | None = None) -> dict[str, object]:
    headers = {"Authorization": f"Bearer {api_key()}"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"https://api.openai.com/v1{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API OpenAI : HTTP {error.code} — {details}") from error


def upload(path: Path) -> dict[str, object]:
    boundary = f"----OpenAIBatch{uuid.uuid4().hex}"
    file_bytes = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0] or "application/jsonl"
    body = b"".join(
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode(),
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"{path.name}\"\r\nContent-Type: {mime_type}\r\n\r\n"
            ).encode(),
            file_bytes,
            f"\r\n--{boundary}--\r\n".encode(),
        )
    )
    request = Request(
        "https://api.openai.com/v1/files",
        data=body,
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Téléversement : HTTP {error.code} — {details}") from error


def submit() -> None:
    """Soumet les deux lots pilotes, chacun limité à 50 requêtes."""
    paths = [PILOT_DIR / "requests_01.jsonl", PILOT_DIR / "requests_02.jsonl"]
    if not all(path.is_file() for path in paths):
        prepare()

    batches: list[dict[str, str]] = []
    for path in paths:
        uploaded_file = upload(path)
        batch = api_request(
            "/batches",
            method="POST",
            payload={
                "input_file_id": uploaded_file["id"],
                "endpoint": "/v1/responses",
                "completion_window": "24h",
                "metadata": {"job": "spider-fr-rewrite-pilot", "model": MODEL},
            },
        )
        batches.append({"batch_id": batch["id"], "input_file_id": uploaded_file["id"]})
        print(f"Batch créé : {batch['id']}")
    STATE_PATH.write_text(json.dumps({"batches": batches}, indent=2) + "\n", encoding="utf-8")


def load_state() -> list[dict[str, str]]:
    if not STATE_PATH.is_file():
        raise RuntimeError("Aucun batch pilote connu. Exécutez d'abord submit.")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return state["batches"]


def status() -> None:
    """Affiche le statut des deux Batch pilotes."""
    for batch_number, state in enumerate(load_state(), start=1):
        batch = api_request(f"/batches/{state['batch_id']}")
        print(f"{batch['id']} : {batch['status']} — {batch.get('request_counts')}")


def extract_question(batch_line: dict[str, object]) -> tuple[str, str]:
    response = batch_line.get("response")
    if not isinstance(response, dict) or response.get("status_code") != 200:
        raise RuntimeError(f"Réponse Batch invalide : {batch_line}")
    body = response.get("body")
    if not isinstance(body, dict):
        raise RuntimeError(f"Corps de réponse invalide : {batch_line}")

    text_parts: list[str] = []
    for output_item in body.get("output", []):
        if isinstance(output_item, dict):
            for content_item in output_item.get("content", []):
                if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                    text = content_item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
    answer = json.loads("".join(text_parts))
    if not isinstance(answer, dict) or set(answer) != {"id", "question"}:
        raise RuntimeError(f"Réponse JSON inattendue : {answer!r}")
    if not isinstance(answer["id"], str) or not isinstance(answer["question"], str) or not answer["question"].strip():
        raise RuntimeError(f"Réponse de reformulation invalide : {answer!r}")
    return answer["id"], answer["question"].strip()


def download_output(file_id: str) -> str:
    request = Request(
        f"https://api.openai.com/v1/files/{file_id}/content",
        headers={"Authorization": f"Bearer {api_key()}"},
    )
    with urlopen(request) as response:
        return response.read().decode("utf-8")


def collect() -> None:
    """Construit le JSONL pilote nettoyé quand les deux lots sont terminés."""
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions: dict[str, str] = {}
    for batch_number, state in enumerate(load_state(), start=1):
        batch = api_request(f"/batches/{state['batch_id']}")
        if batch.get("status") != "completed":
            raise RuntimeError(f"Le batch {batch['id']} n'est pas terminé : {batch.get('status')}")
        output_file_id = batch.get("output_file_id")
        if not isinstance(output_file_id, str):
            raise RuntimeError(f"Le batch {batch['id']} ne fournit pas de résultat.")
        raw_output = download_output(output_file_id)
        raw_output_path = RAW_OUTPUT_DIR / f"output_{batch_number:02d}.jsonl"
        raw_output_path.write_text(raw_output, encoding="utf-8")
        print(f"Réponse OpenAI brute enregistrée dans {raw_output_path}")
        for line in raw_output.splitlines():
            response_id, question = extract_question(json.loads(line))
            if response_id in questions:
                raise RuntimeError(f"id dupliqué dans les réponses : {response_id}")
            questions[response_id] = question

    records = load_jsonl(SOURCE_PATH)[:PILOT_SIZE]
    cleaned_records: list[dict[str, object]] = []
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
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "submit", "status", "collect"))
    args = parser.parse_args()
    {"prepare": prepare, "submit": submit, "status": status, "collect": collect}[args.command]()


if __name__ == "__main__":
    main()
