"""Construit le JSONL text-to-SQL de fine-tuning à partir du pilote validé."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRANSLATIONS = ROOT / "data" / "04_translated_fr" / "pilot" / "train_spider.jsonl"
JUDGMENTS = ROOT / "data" / "05_checks" / "pilot" / "sol_judgments.jsonl"
OUTPUT = ROOT / "data" / "06_fine_tuning_ready" / "pilot" / "train.jsonl"
MANIFEST = ROOT / "data" / "06_fine_tuning_ready" / "pilot" / "manifest.json"
SYSTEM_PROMPT = """Tu génères une requête SQL SQLite à partir d'une question en français et du schéma fourni.
Retourne uniquement la requête SQL valide, sans explication ni balise Markdown."""


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalize_sql(sql):
    """Compacte les blancs hors chaînes et identifiants SQL quotés."""
    output = []
    quote = None
    pending_space = False
    index = 0
    while index < len(sql):
        character = sql[index]
        if quote:
            output.append(character)
            if character == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    output.append(sql[index + 1])
                    index += 1
                else:
                    quote = None
        elif character in {"'", '"', chr(96)}:
            if pending_space and output:
                output.append(" ")
            pending_space = False
            output.append(character)
            quote = character
        elif character == ",":
            if output and output[-1] == " ":
                output.pop()
            pending_space = False
            output.append(character)
        elif character.isspace():
            pending_space = True
        else:
            if pending_space and output:
                output.append(" ")
            pending_space = False
            output.append(character)
        index += 1
    return "".join(output).strip()


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


def main():
    translations, judgments = load(TRANSLATIONS), load(JUDGMENTS)
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
        user = json.dumps(
            {"question": translation["question"], "schema": translation["schema"]},
            ensure_ascii=False,
        )
        assistant = normalize_sql(translation["sql"])
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]})
        accepted.append(translation)
    validate_examples(examples, accepted)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples), encoding="utf-8")
    MANIFEST.write_text(json.dumps({
        "source_translations": str(TRANSLATIONS.relative_to(ROOT)),
        "source_judgments": str(JUDGMENTS.relative_to(ROOT)),
        "total_candidates": len(translations), "accepted_pass": len(examples),
        "rejected": rejected, "format": "chat_messages_jsonl_text_to_sql",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(examples)} exemples validés écrits dans {OUTPUT}")


if __name__ == "__main__":
    main()
