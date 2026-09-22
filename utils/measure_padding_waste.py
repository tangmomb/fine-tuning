"""Mesure le gaspillage de padding des batches réels du benchmark text-to-SQL.

Le script reprend le même ordre d'exemples et le même chat template que
``evaluation_brut_models/assets/run_evaluation.py``. Il ne charge pas le modèle
en VRAM : seul le processor/tokenizer est nécessaire.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation_brut_models.assets.run_evaluation import (  # noqa: E402
    FEW_SHOT_SOURCE,
    INPUTS,
    load_jsonl,
    make_messages,
    stable_few_shot_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint dont le tokenizer doit être utilisé.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                        help="Tailles de batch à simuler (défaut : 1 2 4 8 16 32).")
    parser.add_argument("--mode", choices=("zero-shot", "few-shot"), default="zero-shot")
    parser.add_argument("--few-shot-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, help="Limite le nombre d'exemples, utile pour un smoke test.")
    parser.add_argument("--show-batches", action="store_true", help="Affiche le détail de chaque batch.")
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return sorted(values)[max(0, math.ceil(len(values) * quantile) - 1)]


def measure_batches(processor: Any, messages: list[list[dict[str, str]]], batch_size: int, thinking: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    total_real_tokens = 0
    total_padded_tokens = 0
    for offset in range(0, len(messages), batch_size):
        batch = messages[offset:offset + batch_size]
        inputs = processor.apply_chat_template(
            batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=thinking,
            processor_kwargs={"padding": True},
        )
        attention_mask = inputs["attention_mask"]
        token_counts = attention_mask.sum(dim=1).tolist()
        padded_tokens = len(batch) * inputs["input_ids"].shape[1]
        real_tokens = sum(token_counts)
        total_real_tokens += real_tokens
        total_padded_tokens += padded_tokens
        details.append({
            "batch_index": len(details),
            "examples": len(batch),
            "real_tokens": real_tokens,
            "padded_tokens": padded_tokens,
            "padding_waste": 1 - real_tokens / padded_tokens,
        })
    waste_values = [item["padding_waste"] for item in details]
    return {
        "batch_size": batch_size,
        "batches": len(details),
        "real_tokens": total_real_tokens,
        "padded_tokens": total_padded_tokens,
        "padding_tokens": total_padded_tokens - total_real_tokens,
        "padding_waste": 1 - total_real_tokens / total_padded_tokens if total_padded_tokens else 0,
        "mean_batch_padding_waste": sum(waste_values) / len(waste_values) if waste_values else 0,
        "p95_batch_padding_waste": percentile(waste_values, 0.95),
    }, details


def main() -> None:
    args = parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(f"Checkpoint introuvable : {args.model}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit doit être au moins 1.")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise ValueError("Toutes les tailles de batch doivent être au moins 1.")

    from transformers import AutoProcessor

    inputs = load_jsonl(INPUTS)
    if args.limit:
        inputs = inputs[:args.limit]
    demonstrations = (
        stable_few_shot_examples(load_jsonl(FEW_SHOT_SOURCE), args.few_shot_k, args.seed)
        if args.mode == "few-shot" else []
    )
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    messages = [make_messages(row, demonstrations) for row in inputs]

    print(f"{len(messages)} exemples ; mode={args.mode} ; thinking={args.thinking}")
    print("batch  batches  tokens réels  tokens padded  padding  waste global  waste moyen  p95 waste")
    for batch_size in dict.fromkeys(args.batch_sizes):
        summary, details = measure_batches(processor, messages, batch_size, args.thinking)
        print(
            f"{summary['batch_size']:>5}  {summary['batches']:>7}  {summary['real_tokens']:>13}"
            f"  {summary['padded_tokens']:>13}  {summary['padding_tokens']:>7}"
            f"  {summary['padding_waste']:>11.2%}  {summary['mean_batch_padding_waste']:>11.2%}"
            f"  {summary['p95_batch_padding_waste']:>8.2%}"
        )
        if args.show_batches:
            print(json.dumps(details, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
