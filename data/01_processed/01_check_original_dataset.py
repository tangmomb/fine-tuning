"""Vérifie tous les splits Spider anglais et leurs bases SQLite associées."""

import json
import sqlite3
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_DIR / "BRUT_spider-original" / "data" / "spider_data"
DATABASE_DIR = PROJECT_DIR / "BRUT_spider-original" / "data" / "spider_data" / "database"
SPLIT_FILENAMES = (
    "train_spider.json",
    "train_others.json",
    "dev.json",
)


def main() -> None:
    all_db_ids: set[str] = set()
    existing_db_ids: set[str] = set()
    invalid_databases: list[str] = []

    for split_filename in SPLIT_FILENAMES:
        dataset_path = DATASET_DIR / split_filename
        with dataset_path.open("r", encoding="utf-8") as dataset_file:
            examples = json.load(dataset_file)

        db_ids = sorted({example["db_id"] for example in examples})
        all_db_ids.update(db_ids)
        missing_databases = [
            db_id
            for db_id in db_ids
            if not (DATABASE_DIR / db_id / f"{db_id}.sqlite").is_file()
        ]
        existing_db_ids.update(set(db_ids) - set(missing_databases))

        print(f"\nSplit : {split_filename}")
        print(f"Nombre d'exemples : {len(examples)}")
        print(f"Nombre de bases uniques : {len(db_ids)}")
        print(f"Nombre de bases manquantes : {len(missing_databases)}")
        for db_id in missing_databases:
            print(f"- Base manquante : {db_id}")

    for db_id in sorted(existing_db_ids):
        database_path = DATABASE_DIR / db_id / f"{db_id}.sqlite"
        try:
            with sqlite3.connect(database_path) as connection:
                integrity_result = connection.execute("PRAGMA integrity_check").fetchone()[0]
                if integrity_result != "ok":
                    invalid_databases.append(f"{db_id} ({integrity_result})")
        except sqlite3.DatabaseError as error:
            invalid_databases.append(f"{db_id} ({error})")

    print(f"\nTotal de bases uniques référencées : {len(all_db_ids)}")
    print(f"Total de bases SQLite vérifiées : {len(existing_db_ids)}")
    print(f"Bases SQLite invalides : {len(invalid_databases)}")
    for database in invalid_databases:
        print(f"- Base invalide : {database}")


if __name__ == "__main__":
    main()
