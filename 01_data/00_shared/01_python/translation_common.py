"""Fonctions communes au pipeline de traduction Spider vers le français."""

import json
import mimetypes
import os
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[3]
MODEL, PILOT_SIZE, PILOT_BATCH_SIZE, MAX_OUTPUT_TOKENS = "zai-glm-5-3", 100, 50, 4096
SYSTEM_PROMPT = """Tu traduis en français des questions anglaises text-to-SQL. Conserve exactement le sens,
les nombres, dates, pourcentages, noms propres, comparaisons, négations, superlatifs et classements.
N'ajoute ni ne retire aucune information. Le SQL et le schéma sont des garde-fous.
Retourne uniquement la traduction française, sans JSON, SQL, commentaire ni balise Markdown."""


def choose_environment() -> str:
    choice = input("Dossier à traiter — 1) pilot  2) production [1/2] : ").strip()
    environments = {"1": "pilot", "2": "production"}
    if choice not in environments:
        raise ValueError("Choix attendu : 1 (pilot) ou 2 (production).")
    return environments[choice]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def config(environment: str) -> tuple[tuple[str, ...], int | None, int]:
    if environment == "pilot":
        return ("train_spider",), PILOT_SIZE, PILOT_BATCH_SIZE
    return ("train_spider", "train_others", "dev", "test"), None, 0


def state_path(environment: str) -> Path:
    return environment_dir("02_translation_batches", environment) / "02_batch_state" / "batch_state.json"


def environment_dir(stage: str, environment: str) -> Path:
    """Retourne le dossier numéroté correspondant à un environnement."""
    folder = {"pilot": "02_pilot", "production": "03_production"}[environment]
    return ROOT / "01_data" / stage / folder


def load_state(environment: str) -> dict:
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


def request_row(record: dict, split: str, index: int) -> dict:
    fields = ("question_original_en", "sql", "schema")
    if not all(isinstance(record.get(field), str) and record[field].strip() for field in fields):
        raise ValueError(f"{split}:{index} : question, SQL ou schéma invalide.")
    identifier = f"{split}:{index}"
    return {"custom_id": identifier, "body": {"max_tokens": MAX_OUTPUT_TOKENS, "temperature": 0, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({field: record[field] for field in fields}, ensure_ascii=False)},
    ]}}


def extract_question(line: dict) -> tuple[str, str]:
    identifier = line.get("custom_id")
    if not isinstance(identifier, str) or not identifier:
        raise RuntimeError("Réponse Mistral sans custom_id.")
    content = line.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str))
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Réponse Mistral sans texte final.")
    question = content.strip()
    if question.startswith("{"):
        answer = json.loads(question)
        if not isinstance(answer.get("id"), str) or not isinstance(answer.get("question"), str):
            raise RuntimeError(f"Réponse JSON Mistral invalide : {answer!r}")
        if answer["id"] != identifier:
            raise RuntimeError(f"id JSON incohérent : {answer['id']} au lieu de {identifier}.")
        question = answer["question"].strip()
    return identifier, question


def normalize_identifier(identifier: str, expected_split: str) -> str:
    """Corrige le préfixe erroné des lots de production historiques."""
    if expected_split in {"train_spider", "train_others", "dev", "test"}:
        _, separator, index = identifier.partition(":")
        if separator and index.isdigit():
            return f"{expected_split}:{index}"
    return identifier


def _key() -> str:
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


def api(path: str, method: str = "GET", payload: dict | None = None) -> dict:
    headers, data = {"Authorization": f"Bearer {_key()}"}, None
    if payload is not None:
        data, headers["Content-Type"] = json.dumps(payload).encode("utf-8"), "application/json"
    try:
        with urlopen(Request("https://api.mistral.ai/v1" + path, data=data, headers=headers, method=method)) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(f"API Mistral : HTTP {error.code} — {error.read().decode(errors='replace')}") from error


def upload(path: Path) -> dict:
    boundary = "----MistralBatch" + uuid.uuid4().hex
    mime = mimetypes.guess_type(path.name)[0] or "application/jsonl"
    body = b"".join((
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\nContent-Type: {mime}\r\n\r\n".encode(),
        path.read_bytes(), f"\r\n--{boundary}--\r\n".encode(),
    ))
    request = Request("https://api.mistral.ai/v1/files", data=body, headers={"Authorization": f"Bearer {_key()}", "Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def download(file_id: str) -> str:
    with urlopen(Request(f"https://api.mistral.ai/v1/files/{file_id}/content", headers={"Authorization": f"Bearer {_key()}"})) as response:
        return response.read().decode("utf-8")
