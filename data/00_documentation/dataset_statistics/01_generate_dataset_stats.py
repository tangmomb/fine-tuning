"""Génère les données affichées par l'interface de statistiques des datasets finaux."""

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "stats-data.js"
TOKENIZER_PATH = ROOT / "models" / "Qwen3.5-0.8B"
DATASETS = (
    ("Apprentissage", "train", ROOT / "data" / "06_training_dataset" / "03_production" / "train.jsonl", "training"),
    ("Apprentissage", "validation", ROOT / "data" / "06_training_dataset" / "03_production" / "validation.jsonl", "training"),
    ("Évaluation", "test_inputs", ROOT / "data" / "07_evaluation_dataset" / "03_production" / "test_inputs.jsonl", "evaluation_input"),
    ("Évaluation", "test_gold", ROOT / "data" / "07_evaluation_dataset" / "03_production" / "test_gold.jsonl", "evaluation_gold"),
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def distribution(values: list[int]) -> dict:
    if not values:
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "total": 0}
    return {
        "mean": round(mean(values), 1),
        "median": round(median(values), 1),
        "min": min(values),
        "max": max(values),
        "total": sum(values),
    }


def extract_record(row: dict, kind: str) -> tuple[list[dict], str | None, str | None, str | None]:
    if kind == "evaluation_gold":
        return [], None, None, row["sql"]
    messages = row["messages"]
    user = next(message["content"] for message in messages if message["role"] == "user")
    user_json = json.loads(user)
    response = next((message["content"] for message in messages if message["role"] == "assistant"), None)
    return messages, user_json["question"], user_json["schema"], response


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    files = []
    for group, split, path, kind in DATASETS:
        rows = load_jsonl(path)
        prompts, questions, schemas, responses = [], [], [], []
        for row in rows:
            messages, question, schema, response = extract_record(row, kind)
            if messages:
                # La conversation est encodée exactement avec le template Qwen.
                prompt_messages = [message for message in messages if message["role"] != "assistant"]
                rendered = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
                prompts.append(len(rendered["input_ids"]))
            if question is not None:
                questions.append(len(tokenizer.encode(question, add_special_tokens=False)))
            if schema is not None:
                schemas.append(len(tokenizer.encode(schema, add_special_tokens=False)))
            if response is not None:
                responses.append(len(tokenizer.encode(response, add_special_tokens=False)))
        files.append({
            "group": group,
            "split": split,
            "path": path.relative_to(ROOT).as_posix(),
            "records": len(rows),
            "bytes": path.stat().st_size,
            "prompt": distribution(prompts),
            "question": distribution(questions),
            "schema": distribution(schemas),
            "response": distribution(responses),
        })
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tokenizer": "Qwen3.5-0.8B (tokenizer local)",
        "files": files,
    }
    OUTPUT.write_text("window.DATASET_STATS = " + json.dumps(payload, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"Statistiques écrites dans {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
