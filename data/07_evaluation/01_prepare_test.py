"""Prépare les entrées et les SQL gold séparés pour évaluer le modèle sur test."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sql_utils import normalize_sql


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT = """Tu génères une requête SQL SQLite à partir d'une question en français et du schéma fourni.
Retourne uniquement la requête SQL valide, sans explication ni balise Markdown."""
SOURCE = ROOT / "data" / "04_translated_fr" / "production" / "test.jsonl"
OUTPUT_DIR = ROOT / "data" / "07_evaluation" / "production"
REQUIRED_FIELDS = ("db_id", "question", "schema", "sql")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    records = load_jsonl(SOURCE)
    if not records:
        raise ValueError(f"Aucun exemple dans {SOURCE}.")

    inputs, gold = [], []
    for index, record in enumerate(records):
        invalid = [field for field in REQUIRED_FIELDS if not isinstance(record.get(field), str) or not record[field].strip()]
        if invalid:
            raise ValueError(f"test:{index} : champs invalides : {', '.join(invalid)}")
        identifier = f"test:{index}"
        user_content = json.dumps(
            {"question": record["question"], "schema": record["schema"]},
            ensure_ascii=False,
        )
        input_record = {
            "id": identifier,
            "db_id": record["db_id"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        if set(input_record) != {"id", "db_id", "messages"} or [message["role"] for message in input_record["messages"]] != ["system", "user"]:
            raise AssertionError(f"{identifier} : l'entrée modèle ne doit contenir que le prompt, sans SQL gold.")
        inputs.append(input_record)
        gold.append({"id": identifier, "db_id": record["db_id"], "sql": normalize_sql(record["sql"])})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs_path = OUTPUT_DIR / "test_inputs.jsonl"
    gold_path = OUTPUT_DIR / "test_gold.jsonl"
    manifest_path = OUTPUT_DIR / "manifest.json"
    inputs_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in inputs), encoding="utf-8")
    gold_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gold), encoding="utf-8")
    manifest_path.write_text(json.dumps({
        "source": str(SOURCE.relative_to(ROOT)),
        "inputs": str(inputs_path.relative_to(ROOT)),
        "gold": str(gold_path.relative_to(ROOT)),
        "examples": len(inputs),
        "prompt_contains_gold_sql": False,
        "gold_sql_normalized": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(inputs)} entrées d'évaluation écrites dans {inputs_path}")
    print(f"{len(gold)} SQL gold normalisés écrits dans {gold_path}")


if __name__ == "__main__":
    main()
