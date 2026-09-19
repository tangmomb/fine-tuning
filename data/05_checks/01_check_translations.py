"""Produit les contrôles mécaniques gratuits des traductions Luna."""

import json
import re
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
SOURCE_PATH = PROJECT_DIR / "data" / "04_translated_fr" / "pilot" / "train_spider.jsonl"
OUTPUT_DIR = PROJECT_DIR / "data" / "05_checks" / "pilot"
OUTPUT_PATH = OUTPUT_DIR / "deterministic_checks.jsonl"
REQUIRED_FIELDS = ("db_id", "question_original_en", "question", "sql", "schema")
NUMBER_PATTERN = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?(?![\w])")
SQL_PATTERN = re.compile(
    r"\b(?:SELECT|FROM|WHERE|JOIN|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b",
    re.IGNORECASE,
)
LETTER_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")


def normalize_numbers(text: str) -> Counter[str]:
    """Compare les nombres sans distinguer 3,5 de 3.5."""
    return Counter(match.group().replace(",", ".") for match in NUMBER_PATTERN.finditer(text))


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


def main() -> None:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(
            f"Traductions absentes : {SOURCE_PATH}. Récupérez d'abord le Batch Luna."
        )
    with SOURCE_PATH.open("r", encoding="utf-8") as source_file:
        records = [json.loads(line) for line in source_file if line.strip()]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checks = [check_record(record, index) for index, record in enumerate(records)]
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output_file:
        for check in checks:
            output_file.write(json.dumps(check, ensure_ascii=False) + "\n")

    failed = sum(check["status"] == "fail" for check in checks)
    print(f"{len(checks)} contrôles écrits dans {OUTPUT_PATH}")
    print(f"pass={len(checks) - failed}, fail={failed}")


if __name__ == "__main__":
    main()
