"""Calcule l'Execution Accuracy de validation par difficulté Spider.

Les prédictions archivées contiennent déjà le verdict d'exécution de chaque
exemple. Ce script les rattache aux AST SQL gold de Spider, applique la règle
officielle de difficulté déjà utilisée par le projet, puis agrège les résultats
pour le baseline et les expériences A à D.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
RAW_DEV = ROOT / "BRUT_spider-original" / "data" / "spider_data" / "dev.json"
CLASSIFIER = ROOT / "03_fine_tuning" / "03_02_train_data_scaling" / "build_train_with_hardness.py"
CAMPAIGN = ROOT / "03_fine_tuning" / "artifacts" / "data-scaling-new-data-20261010"
RUNS = {
    "Train set Spider initial uniquement": ROOT / "03_fine_tuning" / "artifacts" / "qwen3.5-2b-20260926-125948Z_r8_alpha16_lr1e-4" / "eval" / "epoch-3" / "val_predictions.jsonl",
    "Dataset A + Train set Spider initial": ROOT / "03_fine_tuning" / "artifacts" / "data-scaling-new-data-20261009" / "A" / "more-data-a-20261009-210634Z" / "eval" / "epoch-3" / "val_predictions.jsonl",
    "Dataset B + Train set Spider initial": CAMPAIGN / "B" / "more-data-b-20261010-143623Z" / "eval" / "epoch-3" / "val_predictions.jsonl",
    "Dataset C + Train set Spider initial": CAMPAIGN / "C" / "more-data-c-20261010-160253Z" / "eval" / "epoch-2" / "val_predictions.jsonl",
    "Dataset D + Train set Spider initial": ROOT / "03_fine_tuning" / "artifacts" / "data-scaling-new-data-20261009" / "D" / "more-data-d-20261009-222330Z" / "eval" / "epoch-3" / "val_predictions.jsonl",
}
HARDNESS_LEVELS = ("easy", "medium", "hard", "extra")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_hardness_classifier() -> Callable[[dict[str, Any]], str]:
    spec = importlib.util.spec_from_file_location("spider_hardness_classifier", CLASSIFIER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger le classifieur : {CLASSIFIER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.spider_hardness


def validation_metadata() -> list[dict[str, str]]:
    classifier = load_hardness_classifier()
    raw_rows = read_json(RAW_DEV)
    if len(raw_rows) != 1034:
        raise ValueError(f"{RAW_DEV} : 1 034 exemples attendus, {len(raw_rows)} trouvés.")
    return [
        {"id": f"dev:{index}", "db_id": row["db_id"], "hardness": classifier(row["sql"])}
        for index, row in enumerate(raw_rows)
    ]


def summarize_run(label: str, prediction_path: Path, metadata: list[dict[str, str]]) -> dict[str, Any]:
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prédictions introuvables pour {label} : {prediction_path}")
    predictions = {row.get("id"): row for row in read_jsonl(prediction_path)}
    expected_ids = {row["id"] for row in metadata}
    if set(predictions) != expected_ids:
        raise ValueError(f"{label} : les identifiants de prédiction ne correspondent pas au split validation Spider.")

    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for item in metadata:
        prediction = predictions[item["id"]]
        if prediction.get("db_id") != item["db_id"]:
            raise ValueError(f"{label}, {item['id']} : db_id de prédiction incohérent.")
        if not isinstance(prediction.get("execution_match"), bool):
            raise ValueError(f"{label}, {item['id']} : execution_match booléen attendu.")
        grouped[item["hardness"]]["total"] += 1
        grouped[item["hardness"]]["correct"] += int(prediction["execution_match"])

    by_hardness = {
        level: {
            "correct": grouped[level]["correct"],
            "total": grouped[level]["total"],
            "execution_accuracy": grouped[level]["correct"] / grouped[level]["total"],
        }
        for level in HARDNESS_LEVELS
    }
    correct = sum(row["correct"] for row in by_hardness.values())
    total = sum(row["total"] for row in by_hardness.values())
    return {
        "label": label,
        "prediction_file": str(prediction_path.relative_to(ROOT)),
        "overall": {"correct": correct, "total": total, "execution_accuracy": correct / total},
        "by_hardness": by_hardness,
    }


def print_table(results: list[dict[str, Any]]) -> None:
    print("| Entraînement | Easy | Medium | Hard | Extra | Total |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        cells = []
        for level in HARDNESS_LEVELS:
            score = result["by_hardness"][level]
            cells.append(f"{score['execution_accuracy']:.2%} ({score['correct']}/{score['total']})")
        overall = result["overall"]
        print(f"| {result['label']} | {' | '.join(cells)} | {overall['execution_accuracy']:.2%} ({overall['correct']}/{overall['total']}) |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CAMPAIGN / "validation_by_hardness.json")
    args = parser.parse_args()

    metadata = validation_metadata()
    results = [summarize_run(label, path, metadata) for label, path in RUNS.items()]
    payload = {
        "purpose": "Validation Execution Accuracy by official Spider hardness",
        "validation_examples": len(metadata),
        "validation_examples_by_hardness": dict(Counter(row["hardness"] for row in metadata)),
        "runs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_table(results)
    print(f"\nRésultats écrits dans {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
