"""Produit les contrôles mécaniques gratuits des traductions."""

import json
import re
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
REQUIRED_FIELDS = ("db_id", "question_original_en", "question", "sql", "schema")
NUMBER_PATTERN = re.compile(
    r"(?<![\w])(?:\d{1,3}(?:[ \u00a0]\d{3})+|\d+)(?:[.,]\d+)?(?:st|nd|rd|th|e|er|ème|s)?(?![\w])",
    re.IGNORECASE,
)
ONE_IN_COMPARISON_PATTERN = re.compile(r"\b(?:plus|moins)\s+d[’'](?:un|une)\b", re.IGNORECASE)
ONE_IN_ENGLISH_COMPARISON_PATTERN = re.compile(r"\b(?:more|less)\s+than\s+one\b", re.IGNORECASE)
ONE_IN_ENGLISH_SINGLE_PATTERN = re.compile(r"\b(?:a\s+)?single\b", re.IGNORECASE)
SQL_PATTERN = re.compile(
    r"\b(?:SELECT|FROM|WHERE|JOIN|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b",
    re.IGNORECASE,
)
LETTER_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")


def normalize_numbers(text: str) -> Counter[str]:
    """Compare les nombres sans distinguer 3,5 de 3.5."""
    numbers = Counter(
        re.sub(r"(?:st|nd|rd|th|e|er|ème|s)$", "", match.group(), flags=re.IGNORECASE)
        .replace("\u00a0", "").replace(" ", "").replace(",", ".")
        for match in NUMBER_PATTERN.finditer(text)
    )
    # Une traduction naturelle peut rendre « more/less than 1 » par « plus/moins d'un ».
    numbers.update("1" for _ in ONE_IN_COMPARISON_PATTERN.finditer(text))
    numbers.update("1" for _ in ONE_IN_ENGLISH_COMPARISON_PATTERN.finditer(text))
    numbers.update("1" for _ in ONE_IN_ENGLISH_SINGLE_PATTERN.finditer(text))
    return numbers


def check_record(record: object, index: int) -> dict[str, object]:
    issues: list[str] = []
    if not isinstance(record, dict):
        return {
            "id": f"train_spider:{index}",
            "status": "fail",
            "issues": ["La ligne source n'est pas un objet JSON."],
        }

    invalid_fields = [
        field
        for field in REQUIRED_FIELDS
        if not isinstance(record.get(field), str) or not record[field].strip()
    ]
    if invalid_fields:
        issues.append(f"Champs absents ou vides : {', '.join(invalid_fields)}.")

    source_question = record.get("question_original_en", "")
    translated_question = record.get("question", "")
    if isinstance(source_question, str) and isinstance(translated_question, str):
        if normalize_numbers(source_question) != normalize_numbers(translated_question):
            issues.append("Les nombres ou dates numériques ne correspondent pas à la question source.")
        if source_question.count("%") != translated_question.count("%"):
            issues.append("Le nombre de pourcentages ne correspond pas à la question source.")
        if SQL_PATTERN.search(translated_question):
            issues.append("La traduction semble contenir du SQL.")
        if translated_question.strip() and not LETTER_PATTERN.search(translated_question):
            issues.append("La traduction ne contient aucun caractère alphabétique.")

    return {
        "id": f"train_spider:{index}",
        "db_id": record.get("db_id"),
        "question_original_en": source_question,
        "question": translated_question,
        "status": "pass" if not issues else "fail",
        "issues": issues,
    }


def manual_corrections(environment: str) -> dict[str, str]:
    path = PROJECT_DIR / "data" / "05_checks" / environment / "manual_corrections.jsonl"
    if not path.is_file():
        return {}
    corrections = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row.get("id"), str) or not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Correction manuelle invalide : {row!r}")
        corrections[row["id"]] = row["question"].strip()
    return corrections


def main() -> None:
    choice = input("Dossier à traiter — 1) pilot  2) production [1/2] : ").strip()
    environments = {"1": "pilot", "2": "production"}
    if choice not in environments:
        raise ValueError("Choix attendu : 1 (pilot) ou 2 (production).")
    environment = environments[choice]
    source_dir = PROJECT_DIR / "data" / "04_translated_fr" / environment
    output_dir = PROJECT_DIR / "data" / "05_checks" / environment
    corrections = manual_corrections(environment)
    sources = sorted(source_dir.glob("*.jsonl"))
    if not sources:
        raise FileNotFoundError(f"Traductions absentes dans {source_dir}. Récupérez d'abord les Batchs.")
    for source_path in sources:
        with source_path.open("r", encoding="utf-8") as source_file:
            records = [json.loads(line) for line in source_file if line.strip()]
        checks = []
        for index, record in enumerate(records):
            identifier = f"{source_path.stem}:{index}"
            if identifier in corrections and isinstance(record, dict):
                record = {**record, "question": corrections[identifier]}
            checks.append({**check_record(record, index), "id": identifier})
        output_path = output_dir / f"{source_path.stem}_deterministic_checks.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for check in checks:
                output_file.write(json.dumps(check, ensure_ascii=False) + "\n")
        failed = sum(check["status"] == "fail" for check in checks)
        print(f"{len(checks)} contrôles écrits dans {output_path} — pass={len(checks) - failed}, fail={failed}")


if __name__ == "__main__":
    main()
