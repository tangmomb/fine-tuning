"""Convertit les réponses Batch validées en SQLite et catalogue JSONL pour B/D."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FINAL = HERE / "04_final"
DATABASES = FINAL / "databases"
CATALOG = FINAL / "new_databases.jsonl"

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def output_text(response: dict[str, Any]) -> str | None:
    for item in response.get("response", {}).get("body", {}).get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    return None

def authorizer(action: int, arg1: str | None, arg2: str | None, _db: str | None, _source: str | None) -> int:
    allowed = {sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_INSERT, sqlite3.SQLITE_READ, sqlite3.SQLITE_TRANSACTION}
    # SQLite met à jour son catalogue interne pendant CREATE TABLE/INDEX.
    if action == sqlite3.SQLITE_UPDATE and arg1 == "sqlite_master":
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

def statement_policy(sql: str, prefix: str) -> None:
    forbidden = r"\b(?:pragma|attach|detach|drop|alter|create\s+(?:view|trigger)|replace|update|delete|vacuum)\b"
    if re.search(forbidden, sql, flags=re.I) or not all(part.strip().lower().startswith(prefix) for part in sql.split(";") if part.strip()):
        raise ValueError(f"SQL non autorisé : seules les instructions {prefix.upper()} sont acceptées.")

def build_database(path: Path, ddl: str, data: str) -> tuple[str, dict[str, int]]:
    if not isinstance(ddl, str) or not isinstance(data, str):
        raise ValueError("ddl et data doivent être des chaînes SQL.")
    statement_policy(ddl, "create")
    statement_policy(data, "insert into")
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(ddl); connection.executescript(data)
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        if not 4 <= len(tables) <= 8:
            raise ValueError(f"{len(tables)} tables : 4 à 8 attendues.")
        counts = {table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in tables}
        if any(count < 10 for count in counts.values()):
            raise ValueError("Chaque table doit contenir au moins 10 lignes.")
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise ValueError(f"{len(violations)} violation(s) de clé étrangère.")
        schema = "\n\n".join(row[0] + ";" for row in connection.execute("SELECT sql FROM sqlite_master WHERE type IN ('table', 'index') AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY type, name"))
        connection.commit()
        return schema, counts
    finally:
        connection.close()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Sortie Batch JSONL ; par défaut, celle enregistrée dans batch_state.json.")
    parser.add_argument("--source-examples", type=int, default=1, help="Poids uniforme de chaque base dans le catalogue.")
    parser.add_argument("--append", action="store_true", help="Ajoute les bases validées au catalogue final existant (pour un Batch de complément).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.source_examples < 1:
        raise ValueError("--source-examples doit être positif.")
    if args.input is None:
        state = json.loads((HERE / "02_batch" / "batch_state.json").read_text(encoding="utf-8"))
        args.input = HERE / "03_responses" / state["raw_output_file"]
    if CATALOG.exists() and not args.overwrite and not args.append:
        raise FileExistsError(f"{CATALOG} existe déjà ; utilisez --overwrite pour le remplacer.")
    accepted, rejected = [], []
    existing = read_jsonl(CATALOG) if args.append and CATALOG.is_file() else []
    existing_ids = {row["db_id"] for row in existing}
    DATABASES.mkdir(parents=True, exist_ok=True)
    for response in read_jsonl(args.input):
        identifier = str(response.get("custom_id", ""))
        match = re.fullmatch(r"new-db:\d{3}:(synthetic_\d{3})", identifier)
        try:
            if not match:
                raise ValueError("identifiant Batch invalide")
            payload = json.loads(output_text(response) or "")
            db_id = match.group(1); db_path = DATABASES / f"{db_id}.sqlite"
            if db_id in existing_ids:
                raise ValueError("db_id déjà présent dans le catalogue")
            if db_path.exists() and not args.overwrite:
                raise FileExistsError("SQLite existe déjà")
            if db_path.exists():
                db_path.unlink()
            schema, counts = build_database(db_path, payload["ddl"], payload["data"])
            accepted.append({"db_id": db_id, "schema": schema, "source_examples": args.source_examples, "database_path": str(db_path.resolve()), "description": payload["description"], "table_row_counts": counts})
        except Exception as error:
            rejected.append({"id": identifier, "reason": str(error)})
            if 'db_path' in locals() and db_path.exists():
                db_path.unlink()
    report = FINAL / "materialized_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"accepted": len(accepted), "rejected": len(rejected), "rejections": rejected}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not accepted:
        reasons = "; ".join(sorted({row["reason"] for row in rejected}))
        raise RuntimeError(f"Aucune base valide : le catalogue n'a pas été écrit. Motifs : {reasons}")
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    CATALOG.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [*existing, *accepted]), encoding="utf-8")
    print(f"{len(accepted)} bases validées, {len(rejected)} rejetées, {len(existing) + len(accepted)} au total : {CATALOG}")

if __name__ == "__main__":
    main()
