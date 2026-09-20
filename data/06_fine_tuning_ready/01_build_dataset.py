"""Construit le JSONL text-to-SQL de fine-tuning depuis pilot ou production."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sql_utils import normalize_sql

ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT = """Tu génères une requête SQL SQLite à partir d'une question en français et du schéma fourni.
Retourne uniquement la requête SQL valide, sans explication ni balise Markdown."""


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_examples(examples, accepted_translations):
    """Garantit le contrat SFT avant toute écriture du fichier final."""
    if len(examples) != len(accepted_translations):
        raise ValueError("Le nombre d'exemples SFT ne correspond pas aux traductions retenues.")
    for index, (example, translation) in enumerate(zip(examples, accepted_translations)):
        messages = example.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"Exemple {index} : structure messages invalide.")
        system, user, assistant = messages
        if system != {"role": "system", "content": SYSTEM_PROMPT}:
            raise ValueError(f"Exemple {index} : message system non conforme.")
        if not isinstance(user, dict) or user.get("role") != "user" or not isinstance(user.get("content"), str):
            raise ValueError(f"Exemple {index} : message user invalide.")
        try:
            user_data = json.loads(user["content"])
        except json.JSONDecodeError as error:
            raise ValueError(f"Exemple {index} : user n'est pas un JSON valide.") from error
        if set(user_data) != {"question", "schema"}:
            raise ValueError(f"Exemple {index} : clés user invalides.")
        if user_data != {"question": translation["question"], "schema": translation["schema"]}:
            raise ValueError(f"Exemple {index} : contenu user non conforme.")
        if assistant != {"role": "assistant", "content": normalize_sql(translation["sql"])}:
            raise ValueError(f"Exemple {index} : assistant ne contient pas uniquement le SQL gold.")


def correction_map(environment):
    path = ROOT / "data" / "05_checks" / environment / "manual_corrections.jsonl"
    if not path.is_file():
        return {}
    return {
        row["id"]: row["question"]
        for row in load(path)
        if isinstance(row.get("id"), str) and isinstance(row.get("question"), str) and row["question"].strip()
    }


def example_from_translation(translation):
    user = json.dumps({"question": translation["question"], "schema": translation["schema"]}, ensure_ascii=False)
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": normalize_sql(translation["sql"])},
    ]}


def main():
    choice = input("Dossier à traiter — 1) pilot  2) production [1/2] : ").strip()
    environments = {"1": "pilot", "2": "production"}
    if choice not in environments:
        raise ValueError("Choix attendu : 1 (pilot) ou 2 (production).")
    environment = environments[choice]
    translations_path = ROOT / "data" / "04_translated_fr" / environment / "train_spider.jsonl"
    judgments_path = ROOT / "data" / "05_checks" / environment / "sol_judgments.jsonl"
    output = ROOT / "data" / "06_fine_tuning_ready" / environment / "train.jsonl"
    manifest = ROOT / "data" / "06_fine_tuning_ready" / environment / "manifest.json"
    if environment == "production":
        corrections = correction_map(environment)
        judgments = {row["id"]: row for row in load(judgments_path)}
        accepted_by_split, rejected = {"train_spider": [], "train_others": [], "dev": []}, []
        for split in ("train_spider", "train_others", "dev"):
            translations = load(ROOT / "data" / "04_translated_fr" / environment / f"{split}.jsonl")
            checks = {row["id"]: row for row in load(ROOT / "data" / "05_checks" / environment / f"{split}_deterministic_checks.jsonl")}
            for index, translation in enumerate(translations):
                identifier = f"{split}:{index}"
                if identifier in corrections:
                    translation = {**translation, "question": corrections[identifier]}
                judgment = judgments.get(identifier)
                accepted_by_judge = judgment and judgment.get("verdict") == "pass"
                corrected = identifier in corrections
                if not corrected and judgment and judgment.get("verdict") != "pass":
                    rejected.append({"id": identifier, "reason": f"judge_{judgment.get('verdict')}"})
                    continue
                if not corrected and checks[identifier].get("status") != "pass" and not accepted_by_judge:
                    rejected.append({"id": identifier, "reason": "deterministic_fail"})
                    continue
                accepted_by_split[split].append(translation)
        train_translations = [*accepted_by_split["train_spider"], *accepted_by_split["train_others"]]
        validation_translations = accepted_by_split["dev"]
        examples = [example_from_translation(translation) for translation in train_translations]
        validation_examples = [example_from_translation(translation) for translation in validation_translations]
        validate_examples(examples, train_translations)
        validate_examples(validation_examples, validation_translations)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples), encoding="utf-8")
        validation_output = output.parent / "validation.jsonl"
        validation_output.write_text("".join(json.dumps(example, ensure_ascii=False) + "\n" for example in validation_examples), encoding="utf-8")
        manifest.write_text(json.dumps({"source_translations": "data/04_translated_fr/production/*.jsonl", "source_judgments": str(judgments_path.relative_to(ROOT)), "total_candidates": len(train_translations) + len(validation_translations) + len(rejected), "train_examples": len(train_translations), "validation_examples": len(validation_translations), "rejected": rejected, "manual_corrections": sorted(corrections), "format": "chat_messages_jsonl_text_to_sql"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{len(train_translations)} exemples d'entraînement écrits dans {output}")
        print(f"{len(validation_translations)} exemples de validation écrits dans {validation_output}")
        return
    translations, judgments = load(translations_path), load(judgments_path)
    if len(translations) != len(judgments):
        raise ValueError(f"Traductions ({len(translations)}) et jugements ({len(judgments)}) ne correspondent pas.")
    examples, accepted, rejected = [], [], []
    for index, (translation, judgment) in enumerate(zip(translations, judgments)):
        fields = ("question_original_en", "question", "sql", "schema")
        if not all(isinstance(translation.get(field), str) and translation[field].strip() for field in fields):
            raise ValueError(f"Traduction invalide à l'index {index}.")
        identifier = f"train_spider:{index}"
        if judgment.get("verdict") != "pass":
            rejected.append({"id": identifier, "verdict": judgment.get("verdict")})
            continue
        examples.append(example_from_translation(translation))
        accepted.append(translation)
    validate_examples(examples, accepted)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples), encoding="utf-8")
    manifest.write_text(json.dumps({
        "source_translations": str(translations_path.relative_to(ROOT)),
        "source_judgments": str(judgments_path.relative_to(ROOT)),
        "total_candidates": len(translations), "accepted_pass": len(examples),
        "rejected": rejected, "format": "chat_messages_jsonl_text_to_sql",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(examples)} exemples validés écrits dans {output}")


if __name__ == "__main__":
    main()
