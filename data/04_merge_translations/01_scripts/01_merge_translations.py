"""Fusionne les réponses Mistral avec les exemples Spider enrichis."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "00_shared" / "01_python"))
from translation_common import ROOT, choose_environment, config, environment_dir, extract_question, load_jsonl, load_state, normalize_identifier, write_jsonl


def main() -> None:
    environment = choose_environment()
    questions, invalid_responses = {}, []
    for number, item in enumerate(load_state(environment)["batches"], 1):
        path = environment_dir("03_batch_responses", environment) / f"output_{number:02d}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"Réponse brute absente : {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            response = json.loads(line)
            try:
                identifier, question = extract_question(response)
            except (json.JSONDecodeError, RuntimeError) as error:
                raw_identifier = response.get("custom_id")
                identifier = normalize_identifier(raw_identifier, item["split"]) if isinstance(raw_identifier, str) else raw_identifier
                invalid_responses.append((identifier, str(error)))
                continue
            identifier = normalize_identifier(identifier, item["split"])
            split, _ = identifier.split(":", 1)
            questions.setdefault(split, {})[identifier] = question
    overrides_path = environment_dir("02_translation_batches", environment) / "03_manual_translations" / "manual_translations.jsonl"
    if overrides_path.is_file():
        for row in load_jsonl(overrides_path):
            identifier, question = row.get("id"), row.get("question")
            if not isinstance(identifier, str) or not isinstance(question, str) or not question.strip() or ":" not in identifier:
                raise ValueError(f"Exception manuelle invalide : {row!r}")
            split, _ = identifier.split(":", 1)
            questions.setdefault(split, {})[identifier] = question.strip()
    unresolved = [f"{identifier or '<id inconnu>'} ({error})" for identifier, error in invalid_responses if not isinstance(identifier, str) or ":" not in identifier or identifier not in questions.get(identifier.split(":", 1)[0], {})]
    if unresolved:
        details = "; ".join(unresolved[:20])
        suffix = "" if len(unresolved) <= 20 else f" ; … ({len(unresolved)} au total)"
        raise RuntimeError(f"{len(unresolved)} réponses Mistral invalides : {details}{suffix}")
    splits, limit, _ = config(environment)
    for split in splits:
        records = load_jsonl(ROOT / "data" / "01_prepare" / "02_output" / f"{split}.jsonl")[:limit]
        missing = [f"{split}:{index}" for index in range(len(records)) if f"{split}:{index}" not in questions.get(split, {})]
        if missing:
            raise RuntimeError(f"Traductions absentes pour {split} : {', '.join(missing[:20])}")
        translated = [{**record, "question": questions[split][f"{split}:{index}"]} for index, record in enumerate(records)]
        output = environment_dir("04_merge_translations", environment) / f"{split}.jsonl"
        write_jsonl(output, translated)
        print(f"{split} : {len(translated)} traductions écrites dans {output}")


if __name__ == "__main__":
    main()
