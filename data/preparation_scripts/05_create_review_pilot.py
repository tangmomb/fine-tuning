"""Crée un fichier de revue simple pour comparer les questions du pilote."""

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
SOURCE_PATH = PROJECT_DIR / "data" / "04_cleaned" / "pilot" / "train_spider.jsonl"
OUTPUT_PATH = PROJECT_DIR / "data" / "05_review" / "pilot" / "question_comparison.json"


def main() -> None:
    with SOURCE_PATH.open("r", encoding="utf-8") as source_file:
        comparisons = []
        for line_number, line in enumerate(source_file, start=1):
            record = json.loads(line)
            question_original = record.get("question_original")
            question = record.get("question")
            if not isinstance(question_original, str) or not isinstance(question, str):
                raise ValueError(f"Ligne {line_number} : question manquante ou invalide.")
            comparisons.append(
                {
                    "question_original": question_original,
                    "question": question,
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(comparisons, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print(f"{len(comparisons)} comparaisons écrites dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
