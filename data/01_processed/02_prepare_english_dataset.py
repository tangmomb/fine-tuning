"""Prépare les splits Spider originaux anglais en JSONL pour traduction."""

import json
import sqlite3
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_DIR / "BRUT_spider-original" / "data" / "spider_data"
DATABASE_DIR = PROJECT_DIR / "BRUT_spider-original" / "data" / "spider_data" / "database"
OUTPUT_DIR = PROJECT_DIR / "data" / "01_processed"
SPLITS = ("train_spider", "train_others", "dev")
REQUIRED_FIELDS = ("db_id", "question", "query")


def validate_example(example: object, source_path: Path, index: int) -> dict[str, str]:
    """Valide les champs nécessaires d'un exemple source."""
    if not isinstance(example, dict):
        raise ValueError(f"{source_path} : l'exemple {index} n'est pas un objet JSON.")

    missing_fields = [field for field in REQUIRED_FIELDS if field not in example]
    if missing_fields:
        fields = ", ".join(missing_fields)
        raise ValueError(f"{source_path} : l'exemple {index} manque : {fields}.")

    invalid_fields = [
        field
        for field in REQUIRED_FIELDS
        if not isinstance(example[field], str) or not example[field].strip()
    ]
    if invalid_fields:
        fields = ", ".join(invalid_fields)
        raise ValueError(
            f"{source_path} : l'exemple {index} contient des champs non textuels : {fields}."
        )

    return {field: example[field] for field in REQUIRED_FIELDS}


def extract_schema(database_path: Path) -> str:
    """Retourne les CREATE TABLE non internes d'une base SQLite."""
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()

    create_statements = [
        create_sql.rstrip().rstrip(";")
        for table_name, create_sql in rows
        if not table_name.startswith("sqlite_") and create_sql is not None
    ]
    return ";\n\n".join(create_statements) + ";" if create_statements else ""


def prepare_split(split_name: str, schema_cache: dict[str, str]) -> tuple[int, dict[str, str]]:
    """Transforme un split source en JSONL et retourne son premier exemple."""
    source_path = SOURCE_DIR / f"{split_name}.json"
    output_path = OUTPUT_DIR / f"{split_name}.jsonl"

    with source_path.open("r", encoding="utf-8") as source_file:
        source_examples = json.load(source_file)

    if not isinstance(source_examples, list):
        raise ValueError(f"{source_path} doit contenir une liste d'exemples.")

    validated_examples = [
        validate_example(example, source_path, index)
        for index, example in enumerate(source_examples)
    ]

    for example in validated_examples:
        db_id = example["db_id"]
        database_path = DATABASE_DIR / db_id / f"{db_id}.sqlite"
        if not database_path.is_file():
            raise FileNotFoundError(
                f"Base SQLite introuvable pour db_id={db_id!r} : {database_path}"
            )
        if db_id not in schema_cache:
            schema_cache[db_id] = extract_schema(database_path)
            if not schema_cache[db_id].strip():
                raise ValueError(
                    f"Schéma SQLite vide pour db_id={db_id!r} : {database_path}"
                )

    first_output_example: dict[str, str] | None = None
    seen_examples: set[tuple[str, str, str]] = set()
    generated_count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for example in validated_examples:
            example_key = (example["db_id"], example["question"], example["query"])
            if example_key in seen_examples:
                continue
            seen_examples.add(example_key)

            output_example = {
                "db_id": example["db_id"],
                "question_original_en": example["question"],
                "schema": schema_cache[example["db_id"]],
                "sql": example["query"],
            }
            output_file.write(json.dumps(output_example, ensure_ascii=False) + "\n")
            generated_count += 1
            if first_output_example is None:
                first_output_example = output_example

    if first_output_example is None:
        raise ValueError(f"{source_path} ne contient aucun exemple.")

    return generated_count, first_output_example


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    schema_cache: dict[str, str] = {}
    results: dict[str, tuple[int, dict[str, str]]] = {}

    for split_name in SPLITS:
        results[split_name] = prepare_split(split_name, schema_cache)

    print("Fichiers générés :")
    for split_name in SPLITS:
        count, example = results[split_name]
        output_path = OUTPUT_DIR / f"{split_name}.jsonl"
        print(f"- {split_name} : {count} exemples — {output_path}")
        print(json.dumps(example, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
