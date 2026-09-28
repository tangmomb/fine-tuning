"""Convertit les exemples validés au contrat messages JSONL du fine-tuning."""

from __future__ import annotations
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = """Tu génères une requête SQL SQLite à partir d'une question en français et du schéma fourni.
Retourne uniquement la requête SQL valide, sans explication ni balise Markdown."""

def ask_experiment(value: str | None) -> str:
    choice = value or input("Expérience à exporter [A/B/C/D] : ").strip().upper()
    if choice not in "ABCD": raise ValueError("Choix attendu : A, B, C ou D.")
    return choice

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--experiment", choices="ABCD"); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args(); args.experiment = ask_experiment(args.experiment)
    source = HERE / "03_quality" / "01_deterministic_checks" / args.experiment / "selected.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = HERE / "04_export" / f"synthetic_data_{args.experiment}.jsonl"
    if output.exists() and not args.overwrite:
        if input(f"{output.name} existe déjà. Le remplacer ? [o/N] ").strip().lower() not in {"o", "oui"}: return
    output.parent.mkdir(parents=True, exist_ok=True)
    examples = []
    for row in rows:
        user = json.dumps({"question": row["question"], "schema": row["schema"]}, ensure_ascii=False)
        examples.append({"id": row["id"], "source_id": row["id"], "db_id": row["db_id"], "hardness": row.get("hardness"), "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}, {"role": "assistant", "content": row["sql"].strip()}]})
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8")
    (output.parent / f"synthetic_data_{args.experiment}_manifest.json").write_text(json.dumps({"experiment": args.experiment, "examples": len(examples), "format": "chat_messages_jsonl_text_to_sql"}, indent=2) + "\n", encoding="utf-8")
    print(f"{len(examples)} exemples SFT écrits dans {output}")

if __name__ == "__main__": main()
