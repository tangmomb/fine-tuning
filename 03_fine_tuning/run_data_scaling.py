"""Lance un data-scaling LoRA sur des sous-ensembles Spider imbriqués.

Les sous-ensembles 25 %, 50 %, 75 % et 100 % sont imbriqués et répartis
proportionnellement par ``db_id``. Chaque base conserve, à un exemple
d'arrondi près, sa part dans le jeu complet — plutôt qu'un simple ``head``.
Le split validation, le modèle, la seed et le protocole d'évaluation restent
identiques entre les quatre entraînements. Le split test n'est jamais lu.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATASET = ROOT / "01_data" / "06_training_dataset" / "03_production" / "train.jsonl"
VALIDATION_DATASET = ROOT / "01_data" / "06_training_dataset" / "03_production" / "validation.jsonl"
TRANSLATIONS_DIR = ROOT / "01_data" / "04_merge_translations" / "03_production"
CHECKS_DIR = ROOT / "01_data" / "05_quality_control" / "03_production" / "01_deterministic_checks"
JUDGMENTS = ROOT / "01_data" / "05_quality_control" / "03_production" / "09_judgments" / "sol_judgments.jsonl"
CORRECTIONS = ROOT / "01_data" / "05_quality_control" / "03_production" / "02_manual_corrections" / "manual_corrections.jsonl"
TRAIN_SCRIPT = Path(__file__).with_name("train.py")
PERCENTAGES = (25, 50, 75, 100)
TRAINING_PERCENTAGES = (25, 50, 75)
EXISTING_FULL_RUN = ROOT / "03_fine_tuning" / "artifacts" / "qwen3.5-2b-20260926-125948Z_r8_alpha16_lr1e-4"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def accepted_training_metadata() -> list[dict[str, Any]]:
    """Reconstruit l'ordre des exemples SFT finaux en conservant leur db_id."""
    judgments = {row["id"]: row for row in read_jsonl(JUDGMENTS)}
    corrections = (
        {row["id"]: row["question"] for row in read_jsonl(CORRECTIONS)} if CORRECTIONS.is_file() else {}
    )
    accepted: list[dict[str, Any]] = []
    for split in ("train_spider", "train_others"):
        translations = read_jsonl(TRANSLATIONS_DIR / f"{split}.jsonl")
        checks = {row["id"]: row for row in read_jsonl(CHECKS_DIR / f"{split}_deterministic_checks.jsonl")}
        for index, translation in enumerate(translations):
            identifier = f"{split}:{index}"
            judgment = judgments.get(identifier)
            corrected = identifier in corrections
            if not corrected and judgment and judgment.get("verdict") != "pass":
                continue
            if not corrected and checks[identifier].get("status") != "pass" and not (
                judgment and judgment.get("verdict") == "pass"
            ):
                continue
            if corrected:
                translation = {**translation, "question": corrections[identifier]}
            db_id = translation.get("db_id")
            if not isinstance(db_id, str) or not db_id:
                raise ValueError(f"db_id manquant pour {identifier}.")
            accepted.append({"id": identifier, "db_id": db_id, "question": translation["question"], "schema": translation["schema"]})
    return accepted


def attach_db_ids(train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata = accepted_training_metadata()
    if len(train_rows) != len(metadata):
        raise ValueError(f"Train SFT ({len(train_rows)}) et métadonnées ({len(metadata)}) divergent.")
    rows = []
    for index, (row, source) in enumerate(zip(train_rows, metadata)):
        try:
            user = json.loads(row["messages"][1]["content"])
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Exemple SFT invalide à l'index {index}.") from error
        if user != {"question": source["question"], "schema": source["schema"]}:
            raise ValueError(f"L'ordre du train SFT ne correspond plus à la source, index {index}.")
        rows.append({"row": row, "db_id": source["db_id"], "id": source["id"]})
    return rows


def proportional_subsets(rows: list[dict[str, Any]], seed: int) -> dict[int, list[dict[str, Any]]]:
    """Construit des subsets imbriqués avec un quota proportionnel par base."""
    by_db: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_db[row["db_id"]].append(row)
    rng = random.Random(seed)
    database_ids = sorted(by_db)
    tie_breakers = {db_id: rng.random() for db_id in database_ids}
    for db_id in database_ids:
        examples = by_db[db_id]
        rng.shuffle(examples)
        for row in examples:
            row["_sampling_order"] = rng.random()

    total = len(rows)
    previous = {db_id: 0 for db_id in database_ids}
    subsets: dict[int, list[dict[str, Any]]] = {}
    for percentage in PERCENTAGES:
        target = round(total * percentage / 100)
        expected = {db_id: target * len(by_db[db_id]) / total for db_id in database_ids}
        quotas = {db_id: max(previous[db_id], int(expected[db_id])) for db_id in database_ids}
        remaining = target - sum(quotas.values())
        if remaining < 0:
            raise AssertionError("Les quotas précédents empêchent un subset imbriqué.")
        for _ in range(remaining):
            candidates = [db_id for db_id in database_ids if quotas[db_id] < len(by_db[db_id])]
            db_id = max(candidates, key=lambda value: (expected[value] - quotas[value], tie_breakers[value]))
            quotas[db_id] += 1
        selected = [row for db_id in database_ids for row in by_db[db_id][:quotas[db_id]]]
        subsets[percentage] = sorted(selected, key=lambda row: row["_sampling_order"])
        previous = quotas
    return subsets


def subset_manifest(rows: list[dict[str, Any]], total: int, percentage: int) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["db_id"]] += 1
    return {
        "percentage": percentage,
        "examples": len(rows),
        "total_examples": total,
        "databases": len(counts),
        "examples_by_db": dict(sorted(counts.items())),
        "source_ids": [row["id"] for row in rows],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen3.5-2B", choices=("Qwen3.5-2B",))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="Écrit et vérifie les subsets sans entraîner.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs doit être strictement positif.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    output_root = args.output_root or ROOT / "03_fine_tuning" / "artifacts" / f"data-scaling-{timestamp}"
    if output_root.exists():
        raise FileExistsError(f"Le dossier de sortie existe déjà : {output_root}")
    train_rows = read_jsonl(TRAIN_DATASET)
    validation_rows = read_jsonl(VALIDATION_DATASET)
    if not validation_rows:
        raise ValueError("Le split validation est vide.")
    subsets_by_percentage = proportional_subsets(attach_db_ids(train_rows), args.seed)
    output_root.mkdir(parents=True)
    subsets_dir = output_root / "subsets"
    subsets_dir.mkdir()
    manifest: dict[str, Any] = {
        "purpose": "nested, db_id-proportional data scaling evaluated on the fixed validation split",
        "base_model": args.model,
        "seed": args.seed,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05},
        "learning_rate": 1e-4,
        "epochs": args.epochs,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 32,
        "effective_batch_size": 64,
        "validation_dataset": str(VALIDATION_DATASET),
        "validation_examples": len(validation_rows),
        "test_dataset_used": False,
        "existing_full_run": str(EXISTING_FULL_RUN),
        "subsets": [],
    }
    subsets: list[tuple[int, Path]] = []
    for percentage in PERCENTAGES:
        subset = subsets_by_percentage[percentage]
        path = subsets_dir / f"train_{percentage:03d}pct.jsonl"
        write_jsonl(path, [item["row"] for item in subset])
        entry = {**subset_manifest(subset, len(train_rows), percentage), "dataset": str(path)}
        if percentage == 100:
            entry["run"] = str(EXISTING_FULL_RUN)
            entry["run_reused"] = True
        manifest["subsets"].append(entry)
        subsets.append((percentage, path))
    # Contrat central de l'expérience : chaque subset est inclus dans le suivant.
    ids = [entry["source_ids"] for entry in manifest["subsets"]]
    if any(not set(earlier).issubset(later) for earlier, later in zip(ids, ids[1:])):
        raise AssertionError("Les subsets ne sont pas strictement imbriqués.")
    write_json(output_root / "manifest.json", manifest)
    print(f"Subsets écrits dans {subsets_dir} ; validation fixe : {len(validation_rows)} exemples.")
    for entry in manifest["subsets"]:
        print(f"  {entry['percentage']:>3} % : {entry['examples']} exemples, {entry['databases']} bases")
    if args.prepare_only:
        return
    for percentage, dataset in subsets:
        if percentage not in TRAINING_PERCENTAGES:
            continue
        run_name = f"{args.model.lower()}-data-scaling-{percentage:03d}pct"
        command = [
            sys.executable, str(TRAIN_SCRIPT), "--models", args.model,
            "--dataset", str(dataset), "--validation-dataset", str(VALIDATION_DATASET),
            "--output-root", str(output_root / "runs"), "--run-name", run_name,
            "--epochs", str(args.epochs), "--lora-preset", "A", "--learning-rate", "1e-4",
            "--per-device-batch-size", "2", "--gradient-accumulation-steps", "32",
            "--seed", str(args.seed),
        ]
        print(f"\n=== Data scaling {percentage} % ===")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
