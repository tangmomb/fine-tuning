"""Vérifie tous les splits Spider anglais et leurs bases SQLite associées."""

import json
import sqlite3
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_DIR / "BRUT_spider-original" / "data" / "spider_data"
DATABASE_DIR = PROJECT_DIR / "BRUT_spider-original" / "data" / "spider_data" / "database"
TEST_DATABASE_DIR = PROJECT_DIR / "BRUT_spider-original" / "data" / "spider_data" / "test_database"
SPLITS = (
    ("train_spider.json", DATABASE_DIR),
    ("train_others.json", DATABASE_DIR),
    ("dev.json", DATABASE_DIR),
    ("test.json", TEST_DATABASE_DIR),
)


def main() -> None:
    all_db_ids: set[str] = set()
    database_paths: dict[str, Path] = {}
    invalid_databases: list[str] = []

    for split_filename, database_dir in SPLITS:
        dataset_path = DATASET_DIR / split_filename
        with dataset_path.open("r", encoding="utf-8") as dataset_file:
            examples = json.load(dataset_file)

        db_ids = sorted({example["db_id"] for example in examples})
        all_db_ids.update(db_ids)
        missing_databases = [
            db_id
            for db_id in db_ids
            if not (database_dir / db_id / f"{db_id}.sqlite").is_file()
        ]
        database_paths.update({
            db_id: database_dir / db_id / f"{db_id}.sqlite"
            for db_id in db_ids
            if db_id not in missing_databases
        })

        print(f"\nSplit : {split_filename}")
        print(f"Nombre d'exemples : {len(examples)}")
        print(f"Nombre de bases uniques : {len(db_ids)}")
        print(f"Nombre de bases manquantes : {len(missing_databases)}")
        for db_id in missing_databases:
            print(f"- Base manquante : {db_id}")

    for db_id, database_path in sorted(database_paths.items()):
        try:
            with sqlite3.connect(database_path) as connection:
                integrity_result = connection.execute("PRAGMA integrity_check").fetchone()[0]
                if integrity_result != "ok":
                    invalid_databases.append(f"{db_id} ({integrity_result})")
        except sqlite3.DatabaseError as error:
            invalid_databases.append(f"{db_id} ({error})")

    print(f"\nTotal de bases uniques référencées : {len(all_db_ids)}")
    print(f"Total de bases SQLite vérifiées : {len(database_paths)}")
    print(f"Bases SQLite invalides : {len(invalid_databases)}")
    for database in invalid_databases:
        print(f"- Base invalide : {database}")


if __name__ == "__main__":
    main()
