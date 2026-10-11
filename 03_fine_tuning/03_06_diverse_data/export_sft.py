"""Sélectionne les paires acceptées par le juge et les exporte au format SFT du fine-tuning.

La sélection reproduit la pondération de l'ancien A : chaque base reçoit une part de la cible
proportionnelle à son nombre d'exemples dans le train Spider. Les paires d'une base sont tirées
au hasard (graine fixe) ; si une base manque de paires acceptées, son déficit est redistribué
aux bases qui en ont en réserve.

Le fichier est écrit dans 03_03_04_export/synthetic_data_<NOM>.jsonl, directement utilisable par
03_04_train_more_data/train.py --experiment <NOM>.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINE_TUNING = HERE.parent
sys.path.insert(0, str(FINE_TUNING / "03_03_more_data" / "03_03_02_batch"))
sys.path.insert(0, str(FINE_TUNING / "03_03_more_data" / "03_03_04_export"))
from build_sft import SYSTEM_PROMPT  # noqa: E402
from create_batch import largest_remainder  # noqa: E402

PREPARE = FINE_TUNING / "03_03_more_data" / "03_03_01_prepare"
EXPORT = FINE_TUNING / "03_03_more_data" / "03_03_04_export"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select(accepted: list[dict], weights: dict[str, int], target: int, rng: random.Random) -> list[dict]:
    by_db: dict[str, list[dict]] = defaultdict(list)
    for row in accepted:
        by_db[row["db_id"]].append(row)
    for rows in by_db.values():
        rng.shuffle(rows)
    target = min(target, len(accepted))
    quotas = largest_remainder({db: weights.get(db, 1) for db in by_db}, target)
    taken = {db: min(quota, len(by_db[db])) for db, quota in quotas.items()}
    # Redistribue le déficit aux bases qui ont encore des paires, au prorata de leur poids.
    while sum(taken.values()) < target:
        spare = {db: weights.get(db, 1) for db in by_db if taken[db] < len(by_db[db])}
        extra = largest_remainder(spare, target - sum(taken.values()))
        for db, count in extra.items():
            taken[db] = min(len(by_db[db]), taken[db] + count)
    return [row for db in sorted(by_db) for row in by_db[db][:taken[db]]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Dossier contenant judged.jsonl.")
    parser.add_argument("--name", default="A2", help="Nom de l'expérience (synthetic_data_<NOM>.jsonl).")
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--output-dir", type=Path, default=EXPORT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output_dir / f"synthetic_data_{args.name}.jsonl"
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} existe déjà (utiliser --overwrite).")
    judged = read_jsonl(args.input / "judged.jsonl")
    accepted = [row for row in judged if row["verdict"] == "pass"]
    schemas, weights = {}, {}
    for name in ("existing_sql_bases.jsonl", "new_sql_bases.jsonl"):
        for seed in read_jsonl(PREPARE / name):
            schemas[seed["db_id"]] = seed["schema"]
            weights[seed["db_id"]] = seed.get("source_examples", 1)
    selected = select(accepted, weights, args.target, random.Random(args.seed))

    examples = [{"id": row["id"], "source_id": row["id"], "db_id": row["db_id"], "hardness": row["hardness"],
                 "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": json.dumps({"question": row["question"], "schema": schemas[row["db_id"]]}, ensure_ascii=False)},
                              {"role": "assistant", "content": row["sql"].strip()}]}
                for row in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8")
    manifest = {"experiment": args.name, "pipeline": "03_06_diverse_data", "source": str(args.input), "examples": len(examples),
                "target": args.target, "judged": len(judged), "verdicts": dict(Counter(row["verdict"] for row in judged)),
                "databases": len({row["db_id"] for row in selected}), "hardness": dict(Counter(row["hardness"] for row in selected)),
                "format": "chat_messages_jsonl_text_to_sql"}
    output.with_name(f"synthetic_data_{args.name}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(examples)} exemples SFT écrits dans {output}")
    print(json.dumps({key: manifest[key] for key in ("verdicts", "databases", "hardness")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
