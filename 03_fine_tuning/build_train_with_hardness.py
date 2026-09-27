"""Construit une copie SFT enrichie avec les niveaux de difficulté Spider.

Le fichier produit ajoute ``id``, ``db_id`` et ``hardness`` aux ``messages``
du train final. Il ne modifie jamais le train.jsonl source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATASET = ROOT / "01_data" / "06_training_dataset" / "03_production" / "train.jsonl"
TRANSLATIONS_DIR = ROOT / "01_data" / "04_merge_translations" / "03_production"
RAW_SPIDER_DIR = ROOT / "BRUT_spider-original" / "data" / "spider_data"
CHECKS_DIR = ROOT / "01_data" / "05_quality_control" / "03_production" / "01_deterministic_checks"
JUDGMENTS = ROOT / "01_data" / "05_quality_control" / "03_production" / "09_judgments" / "sol_judgments.jsonl"
CORRECTIONS = ROOT / "01_data" / "05_quality_control" / "03_production" / "02_manual_corrections" / "manual_corrections.jsonl"
ENRICHED_TRAIN_DATASET = ROOT / "03_fine_tuning" / "train_with_hardness.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def spider_hardness(sql: dict[str, Any]) -> str:
    """Applique les règles ``Evaluator.eval_hardness`` de Spider au SQL AST gold."""
    where_ops = ("not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists")

    def count_agg(units: list[Any]) -> int:
        return sum(unit[0] != 0 for unit in units)

    def nested_sqls(value: dict[str, Any]) -> list[dict[str, Any]]:
        nested = []
        conditions = value["from"]["conds"][::2] + value["where"][::2] + value["having"][::2]
        for condition in conditions:
            nested.extend(item for item in condition[3:5] if isinstance(item, dict))
        nested.extend(value[key] for key in ("intersect", "except", "union") if value[key] is not None)
        return nested

    component1 = int(bool(sql["where"])) + int(bool(sql["groupBy"])) + int(bool(sql["orderBy"]))
    component1 += int(sql["limit"] is not None) + len(sql["from"]["table_units"]) - 1
    boolean_operators = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    conditions = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    component1 += sum(token == "or" for token in boolean_operators)
    component1 += sum(condition[1] == where_ops.index("like") for condition in conditions)
    component2 = len(nested_sqls(sql))
    agg_count = count_agg(sql["select"][1]) + count_agg(sql["where"][::2]) + count_agg(sql["groupBy"])
    if sql["orderBy"]:
        order_units = [unit[index] for unit in sql["orderBy"][1] for index in (1, 2) if unit[index]]
        agg_count += count_agg(order_units)
    agg_count += count_agg(sql["having"])
    others = int(agg_count > 1) + int(len(sql["select"][1]) > 1) + int(len(sql["where"]) > 1) + int(len(sql["groupBy"]) > 1)
    if component1 <= 1 and others == 0 and component2 == 0:
        return "easy"
    if (others <= 2 and component1 <= 1 and component2 == 0) or (component1 <= 2 and others < 2 and component2 == 0):
        return "medium"
    if (others > 2 and component1 <= 2 and component2 == 0) or (2 < component1 <= 3 and others <= 2 and component2 == 0) or (component1 <= 1 and others == 0 and component2 <= 1):
        return "hard"
    return "extra"


def accepted_training_metadata() -> list[dict[str, Any]]:
    judgments = {row["id"]: row for row in read_jsonl(JUDGMENTS)}
    corrections = {row["id"]: row["question"] for row in read_jsonl(CORRECTIONS)} if CORRECTIONS.is_file() else {}
    accepted: list[dict[str, Any]] = []
    for split in ("train_spider", "train_others"):
        translations = read_jsonl(TRANSLATIONS_DIR / f"{split}.jsonl")
        raw_rows = json.loads((RAW_SPIDER_DIR / f"{split}.json").read_text(encoding="utf-8"))
        raw_sql_by_example = {(row["db_id"], row["question"], row["query"]): row["sql"] for row in raw_rows}
        checks = {row["id"]: row for row in read_jsonl(CHECKS_DIR / f"{split}_deterministic_checks.jsonl")}
        for index, translation in enumerate(translations):
            identifier = f"{split}:{index}"
            judgment = judgments.get(identifier)
            corrected = identifier in corrections
            if not corrected and judgment and judgment.get("verdict") != "pass":
                continue
            if not corrected and checks[identifier].get("status") != "pass" and not (judgment and judgment.get("verdict") == "pass"):
                continue
            if corrected:
                translation = {**translation, "question": corrections[identifier]}
            db_id = translation.get("db_id")
            raw_sql = raw_sql_by_example.get((db_id, translation.get("question_original_en"), translation.get("sql")))
            if not isinstance(db_id, str) or not db_id or not isinstance(raw_sql, dict):
                raise ValueError(f"SQL AST Spider invalide pour {identifier}.")
            accepted.append({"id": identifier, "db_id": db_id, "hardness": spider_hardness(raw_sql),
                             "question": translation["question"], "schema": translation["schema"]})
    return accepted


def build_enriched_dataset(output: Path) -> int:
    train_rows = read_jsonl(TRAIN_DATASET)
    metadata = accepted_training_metadata()
    if len(train_rows) != len(metadata):
        raise ValueError(f"Train SFT ({len(train_rows)}) et métadonnées ({len(metadata)}) divergent.")
    enriched = []
    for index, (row, source) in enumerate(zip(train_rows, metadata)):
        try:
            user = json.loads(row["messages"][1]["content"])
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Exemple SFT invalide à l'index {index}.") from error
        if user != {"question": source["question"], "schema": source["schema"]}:
            raise ValueError(f"L'ordre du train SFT ne correspond plus à la source, index {index}.")
        enriched.append({"id": source["id"], "db_id": source["db_id"], "hardness": source["hardness"], "messages": row["messages"]})
    write_jsonl(output, enriched)
    return len(enriched)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ENRICHED_TRAIN_DATASET)
    parser.add_argument("--overwrite", action="store_true", help="Autorise le remplacement du fichier enrichi existant.")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Le fichier existe déjà : {args.output}. Utilisez --overwrite pour le reconstruire.")
    count = build_enriched_dataset(args.output)
    print(f"{count} exemples enrichis écrits dans {args.output}")


if __name__ == "__main__":
    main()
