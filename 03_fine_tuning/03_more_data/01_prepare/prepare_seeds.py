"""Prépare un plan JSONL de génération, avec un quota par base de données.

A/C utilisent les bases existantes, B/D un catalogue externe. Les quotas,
répartis selon ``source_examples``,
totalisent 5 000 exemples validés par défaut. ``request_quota`` applique une
surproduction (1,25×) pour couvrir les rejets de validation.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
SOURCES = {"existing", "new"}

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def largest_remainder(weights: dict[str, int], total: int) -> dict[str, int]:
    raw = {key: total * value / sum(weights.values()) for key, value in weights.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    for key in sorted(weights, key=lambda item: (raw[item] - quotas[item], item), reverse=True)[:total - sum(quotas.values())]: quotas[key] += 1
    return quotas

def existing_databases() -> list[dict[str, Any]]:
    train = ROOT / "03_fine_tuning" / "02_train_data_scaling" / "train_with_hardness.jsonl"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(train):
        grouped[row["db_id"]].append(row)
    result = []
    for db_id, rows in grouped.items():
        user = json.loads(rows[0]["messages"][1]["content"])
        database = ROOT / "BRUT_spider-original" / "data" / "spider_data" / "database" / db_id / f"{db_id}.sqlite"
        hardness_counts = Counter(row.get("hardness") for row in rows if row.get("hardness") in {"easy", "medium", "hard", "extra"})
        result.append({"db_id": db_id, "schema": user["schema"], "database_path": str(database), "source_examples": len(rows), "hardness_counts": dict(hardness_counts)})
    return result

def external_databases(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    required = {"db_id", "schema", "source_examples"}
    if any(not required <= row.keys() for row in rows): raise ValueError("Corpus externe : db_id, schema et source_examples sont obligatoires.")
    return rows

def ask_source(value: str | None) -> str:
    choice = value or input("Type de bases [existing/new] : ").strip().lower()
    if choice not in SOURCES: raise ValueError("Choix attendu : existing ou new.")
    return choice

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES); parser.add_argument("--count", type=int, help="Nombre final visé après validation.")
    parser.add_argument("--oversample-factor", type=float, help="Marge de génération avant rejets.")
    parser.add_argument("--input", type=Path); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); args.source = ask_source(args.source)
    args.count = args.count if args.count is not None else int(input("Nombre final visé [5000] : ").strip() or "5000")
    args.oversample_factor = args.oversample_factor if args.oversample_factor is not None else float(input("Facteur de surproduction [1.25] : ").strip() or "1.25")
    default_input = HERE / "01_prepare" / "01_new_databases_creation" / "04_final" / "new_databases.jsonl"
    input_path = args.input
    if args.source == "new" and input_path is None:
        input_path = Path(input(f"Catalogue des nouvelles bases [{default_input}] : ").strip() or default_input)
    databases = existing_databases() if args.source == "existing" else external_databases(input_path)
    if not databases: raise ValueError(f"Aucune base disponible pour {args.source}.")
    weights = {row["db_id"]: int(row["source_examples"]) for row in databases}
    if len(weights) != len(databases) or any(value < 1 for value in weights.values()): raise ValueError("Chaque db_id doit être unique et source_examples positif.")
    if args.count < 1 or args.oversample_factor < 1: raise ValueError("--count doit être positif et --oversample-factor >= 1.")
    target_quotas = largest_remainder(weights, args.count)
    request_quotas = largest_remainder(weights, round(args.count * args.oversample_factor))
    output_rows = [{**row, "target_quota": target_quotas[row["db_id"]], "request_quota": request_quotas[row["db_id"]]} for row in sorted(databases, key=lambda item: item["db_id"])]
    output = HERE / "01_prepare" / f"{args.source}_sql_bases.jsonl"
    if output.exists() and not args.overwrite:
        if input(f"{output.name} existe déjà. Le remplacer ? [o/N] ").strip().lower() not in {"o", "oui"}: return
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    print(f"{len(output_rows)} bases, {sum(request_quotas.values())} demandés pour {sum(target_quotas.values())} exemples finaux : {output}")

if __name__ == "__main__": main()
