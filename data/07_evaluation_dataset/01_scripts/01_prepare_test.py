"""Prépare les entrées et les SQL gold séparés pour évaluer le modèle sur test."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "00_shared" / "01_python"))
from sql_utils import normalize_sql


ROOT = Path(__file__).resolve().parents[3]
SYSTEM_PROMPT = """Tu génères une requête SQL SQLite à partir d'une question en français et du schéma fourni.
Retourne uniquement la requête SQL valide, sans explication ni balise Markdown."""
SOURCE = ROOT / "data" / "04_merge_translations" / "03_production" / "test.jsonl"
OUTPUT_DIR = ROOT / "data" / "07_evaluation_dataset" / "03_production"
REQUIRED_FIELDS = ("db_id", "question", "schema", "sql")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def correction_map() -> dict[str, str]:
    path = ROOT / "data" / "05_quality_control" / "03_production" / "02_manual_corrections" / "manual_corrections.jsonl"
    if not path.is_file():
        return {}
    return {
        row["id"]: row["question"]
        for row in load_jsonl(path)
        if isinstance(row.get("id"), str) and isinstance(row.get("question"), str) and row["question"].strip()
    }


def main() -> None:
    records = load_jsonl(SOURCE)
    if not records:
        raise ValueError(f"Aucun exemple dans {SOURCE}.")

    checks_path = ROOT / "data" / "05_quality_control" / "03_production" / "01_deterministic_checks" / "test_deterministic_checks.jsonl"
    judgments_path = ROOT / "data" / "05_quality_control" / "03_production" / "09_judgments" / "sol_judgments_test.jsonl"
    checks = {row["id"]: row for row in load_jsonl(checks_path)}
    judgments = {row["id"]: row for row in load_jsonl(judgments_path)}
    corrections = correction_map()
    inputs, gold = [], []
    rejected = []
    for index, record in enumerate(records):
        identifier = f"test:{index}"
        corrected = identifier in corrections
        if corrected:
            record = {**record, "question": corrections[identifier]}
        judgment = judgments.get(identifier)
        accepted_by_judge = judgment and judgment.get("verdict") == "pass"
        if not corrected and judgment and judgment.get("verdict") != "pass":
            rejected.append({"id": identifier, "reason": f"judge_{judgment.get('verdict')}"})
            continue
        if not corrected and checks[identifier].get("status") != "pass" and not accepted_by_judge:
            rejected.append({"id": identifier, "reason": "deterministic_fail"})
            continue
        invalid = [field for field in REQUIRED_FIELDS if not isinstance(record.get(field), str) or not record[field].strip()]
        if invalid:
            raise ValueError(f"{identifier} : champs invalides : {', '.join(invalid)}")
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
        "source_checks": str(checks_path.relative_to(ROOT)),
        "source_judgments": str(judgments_path.relative_to(ROOT)),
        "inputs": str(inputs_path.relative_to(ROOT)),
        "gold": str(gold_path.relative_to(ROOT)),
        "total_candidates": len(records),
        "examples": len(inputs),
        "rejected": rejected,
        "manual_corrections": sorted(corrections),
        "prompt_contains_gold_sql": False,
        "gold_sql_normalized": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(inputs)} entrées d'évaluation écrites dans {inputs_path}")
    print(f"{len(gold)} SQL gold normalisés écrits dans {gold_path}")


if __name__ == "__main__":
    main()
