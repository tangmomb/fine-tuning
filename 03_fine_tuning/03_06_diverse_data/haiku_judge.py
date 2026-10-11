"""Juge sémantique avec Claude Haiku 5.5, mêmes consignes et même format que semantic_judge.py.

  python haiku_judge.py --input pilot_v2 --run-sync --compare judged.jsonl   # test sur le pilote, comparé à Terra
  python haiku_judge.py --input A_v2 --submit                                  # Message Batches API
  python haiku_judge.py --input A_v2 --collect                                 # statut, puis judged.jsonl

Les requêtes sont construites à partir de judge_requests.jsonl (sortie de semantic_judge.py --prepare),
pour que Haiku et Terra voient exactement le même prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from semantic_judge import SCHEMA, read_jsonl, write_jsonl  # noqa: E402

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 16000


def client() -> anthropic.Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.partition("=")[2].strip().strip('"')
    return anthropic.Anthropic(max_retries=5)


def params(openai_line: dict, effort: str) -> dict:
    """Convertit une requête du juge OpenAI (system + user) en paramètres Messages API."""
    messages = openai_line["body"]["input"]
    system = next(message["content"] for message in messages if message["role"] == "system")
    user = next(message["content"] for message in messages if message["role"] == "user")
    return {"model": MODEL, "max_tokens": MAX_TOKENS, "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}}}


def parse_verdicts(message) -> list[dict]:
    if message.stop_reason != "end_turn":
        return []
    text = next((block.text for block in message.content if block.type == "text"), "")
    try:
        return json.loads(text)["verdicts"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def consolidate(folder: Path, verdict_list: list[dict], output_name: str) -> list[dict]:
    verdicts = {answer["id"]: answer for answer in verdict_list
                if answer.get("verdict") in {"pass", "fail", "uncertain"} and isinstance(answer.get("id"), str)}
    judged = [{**row, "verdict": verdicts.get(row["id"], {}).get("verdict", "missing"),
               "issues": verdicts.get(row["id"], {}).get("issues", ["Verdict absent."])}
              for row in read_jsonl(folder / "validated.jsonl")]
    write_jsonl(folder / output_name, judged)
    print("Verdicts Haiku :", dict(Counter(row["verdict"] for row in judged)))
    return judged


def run_sync(folder: Path, effort: str, output_name: str) -> list[dict]:
    api = client()
    lines = read_jsonl(folder / "judge_requests.jsonl")
    usage = Counter()

    def judge(line: dict) -> list[dict]:
        message = api.messages.create(**params(line, effort))
        usage["input_tokens"] += message.usage.input_tokens
        usage["output_tokens"] += message.usage.output_tokens
        return parse_verdicts(message)

    with ThreadPoolExecutor(max_workers=6) as pool:
        verdict_list = [answer for answers in pool.map(judge, lines) for answer in answers]
    print("Usage :", dict(usage))
    return consolidate(folder, verdict_list, output_name)


def compare(haiku: list[dict], reference_path: Path) -> None:
    reference = {row["id"]: row for row in read_jsonl(reference_path)}
    rejected = {row_id for row_id, row in reference.items() if row["verdict"] != "pass"}
    haiku_rejected = {row["id"] for row in haiku if row["verdict"] != "pass"}
    accepted = set(reference) - rejected
    print(f"\nRéférence ({reference_path.name}) : {len(rejected)} rejets sur {len(reference)}")
    print(f"Rejets de la référence aussi rejetés par Haiku : {len(rejected & haiku_rejected)} / {len(rejected)}")
    print(f"Paires acceptées par la référence mais rejetées par Haiku : {len(accepted & haiku_rejected)} / {len(accepted)}")
    by_id = {row["id"]: row for row in haiku}
    for row_id in sorted(rejected - haiku_rejected):
        print(f"\n  Laissé passer : {reference[row_id]['question']}\n    SQL : {reference[row_id]['sql']}\n    Référence : {' / '.join(reference[row_id]['issues'])}")
    for row_id in sorted(accepted & haiku_rejected):
        print(f"\n  Rejet en plus : {by_id[row_id]['question']}\n    SQL : {by_id[row_id]['sql']}\n    Haiku : {' / '.join(by_id[row_id]['issues'])}")


def submit(folder: Path, effort: str) -> None:
    api = client()
    lines = read_jsonl(folder / "judge_requests.jsonl")
    requests = [Request(custom_id=f"r{index}", params=MessageCreateParamsNonStreaming(**params(line, effort)))
                for index, line in enumerate(lines)]
    batch = api.messages.batches.create(requests=requests)
    state = {"batch_id": batch.id, "model": MODEL, "effort": effort, "requests": len(requests),
             "custom_ids": {f"r{index}": line["custom_id"] for index, line in enumerate(lines)}}
    (folder / "haiku_judge_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Batch Haiku soumis : {batch.id} ({batch.processing_status})")


def collect(folder: Path) -> None:
    api = client()
    state = json.loads((folder / "haiku_judge_state.json").read_text(encoding="utf-8"))
    batch = api.messages.batches.retrieve(state["batch_id"])
    print(f"Statut : {batch.processing_status} · {batch.request_counts}")
    if batch.processing_status != "ended":
        return
    verdict_list, failures = [], Counter()
    for result in api.messages.batches.results(state["batch_id"]):
        if result.result.type == "succeeded":
            answers = parse_verdicts(result.result.message)
            if not answers:
                failures[result.result.message.stop_reason or "invalid_json"] += 1
            verdict_list.extend(answers)
        else:
            failures[result.result.type] += 1
    print("Requêtes sans verdict :", dict(failures))
    consolidate(folder, verdict_list, "judged.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--compare", help="Fichier de verdicts de référence dans --input (par exemple judged.jsonl de Terra).")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run-sync", action="store_true")
    action.add_argument("--submit", action="store_true")
    action.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.run_sync:
        judged = run_sync(args.input, args.effort, f"judged_haiku_{args.effort}.jsonl")
        if args.compare:
            compare(judged, args.input / args.compare)
    elif args.submit:
        submit(args.input, args.effort)
    else:
        collect(args.input)


if __name__ == "__main__":
    main()
