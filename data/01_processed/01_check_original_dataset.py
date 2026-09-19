"""Vérifie la cohérence entre les splits Spider anglais et les bases SQLite."""

import json
import sqlite3
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "BRUT_spider-original" / "data" / "spider_data" / "train_spider.json"
DATABASE_DIR = PROJECT_DIR / "BRUT_spider-original" / "data" / "spider_data" / "database"


def main() -> None:
    with DATASET_PATH.open("r", encoding="utf-8") as dataset_file:
        examples = json.load(dataset_file)

    db_ids = sorted({example["db_id"] for example in examples})
    missing_databases = [
        db_id
        for db_id in db_ids
        if not (DATABASE_DIR / db_id / f"{db_id}.sqlite").is_file()
    ]

    print(f"Nombre total d'exemples : {len(examples)}")
    print(f"Nombre de bases uniques : {len(db_ids)}")
    print(f"Nombre de bases manquantes : {len(missing_databases)}")
    if missing_databases:
        print("Bases manquantes :")
        for db_id in missing_databases:
            print(f"- {db_id}")

    first_example = examples[0]
    db_id = first_example["db_id"]
    database_path = DATABASE_DIR / db_id / f"{db_id}.sqlite"

    print("\nPremier exemple :")
    print(f"db_id : {db_id}")
    print(f"question : {first_example['question']}")
    print(f"query : {first_example['query']}")

    print("\nSchéma SQLite :")
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' ORDER BY name"
        )
        tables = cursor.fetchall()

    for table_name, create_table_sql in tables:
        if table_name.startswith("sqlite_"):
            continue
        print(f"\n{create_table_sql};")


if __name__ == "__main__":
    main()
