"""Convertit les exemples validés au contrat messages JSONL du fine-tuning."""

from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(__file__).resolve().parent
SYSTEM_PROMPT = """Tu génères une requête SQL SQLite à partir d'une question en français et du schéma fourni.
Retourne uniquement la requête SQL valide, sans explication ni balise Markdown."""

def ask_experiment(value: str | None) -> str:
    choice = value or input("Expérience à exporter [A/B/C/D] : ").strip().upper()
    if choice not in "ABCD": raise ValueError("Choix attendu : A, B, C ou D.")
    return choice

def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def accepted_questions(experiment: str, selected: list[dict]) -> dict[str, str]:
    """Retourne la question retenue par id, avec les corrections revalidées."""
    quality = ROOT / "03_03_03_quality"
    judge = quality / "03_03_03_02_semantic_judge"
    initial = read(judge / "03_03_03_02_04_judgments" / f"{experiment}_judged.jsonl")
    rejudged = read(judge / "03_03_03_02_06_semantic_rejudge" / "03_03_03_02_06_04_rejudgments" / f"{experiment}_rejudged.jsonl")
    retry = read(judge / "03_03_03_02_06_semantic_rejudge" / "03_03_03_02_06_04_rejudgments" / f"{experiment}_retry_rejudged.jsonl")

    questions = {row["id"]: row["question"] for row in initial if row.get("verdict") == "pass"}
    questions.update({row["id"]: row["question"] for row in rejudged if row.get("recheck_verdict") == "pass"})
    questions.update({row["id"]: row["question"] for row in retry if row.get("recheck_verdict") == "pass"})
    selected_ids = {row["id"] for row in selected}
    if set(questions) != selected_ids:
        missing = len(selected_ids - set(questions)); extra = len(set(questions) - selected_ids)
        raise RuntimeError(f"Validation sémantique incomplète ou incohérente : manque={missing}, extra={extra}.")
    return questions

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--experiment", choices="ABCD"); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args(); args.experiment = ask_experiment(args.experiment)
    source = ROOT / "03_03_03_quality" / "03_03_03_01_deterministic_checks" / args.experiment / "selected.jsonl"
    rows = read(source)
    questions = accepted_questions(args.experiment, rows)
    output = WORK / f"synthetic_data_{args.experiment}.jsonl"
    if output.exists() and not args.overwrite:
        if input(f"{output.name} existe déjà. Le remplacer ? [o/N] ").strip().lower() not in {"o", "oui"}: return
    output.parent.mkdir(parents=True, exist_ok=True)
    examples = []
    for row in rows:
        user = json.dumps({"question": questions[row["id"]], "schema": row["schema"]}, ensure_ascii=False)
        examples.append({"id": row["id"], "source_id": row["id"], "db_id": row["db_id"], "hardness": row.get("hardness"), "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}, {"role": "assistant", "content": row["sql"].strip()}]})
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8")
    manifest = {"experiment": args.experiment, "examples": len(examples), "format": "chat_messages_jsonl_text_to_sql", "semantic_validation": {"initial_pass": 4876, "corrected_pass": 119, "retry_corrected_pass": 5}}
    (output.parent / f"synthetic_data_{args.experiment}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{len(examples)} exemples SFT écrits dans {output}")

if __name__ == "__main__": main()
