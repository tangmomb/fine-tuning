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
DEFAULT_MODEL = ROOT / "models" / "Qwen3.5-0.8B"
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
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Dossier du tokenizer (défaut : models/Qwen3.5-0.8B).",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                        help="Tailles de batch à simuler (défaut : 1 2 4 8 16 32 64 128 256).")
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


def token_lengths(processor: Any, messages: list[list[dict[str, str]]], thinking: bool) -> list[int]:
    """Retourne la longueur tokenisée de chaque prompt, sans padding."""
    lengths = []
    for message in messages:
        encoded = processor.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=thinking,
        )
        lengths.append(int(encoded["attention_mask"].sum().item()))
    return lengths


def make_batches(
    messages: list[list[dict[str, str]]],
    batch_size: int,
    lengths: list[int] | None = None,
) -> list[list[tuple[int, list[dict[str, str]]]]]:
    """Crée des batches sans modifier l'ordre ni le contenu de ``messages``.

    Avec ``lengths``, une copie ordonnée par longueur est utilisée pour rapprocher
    les prompts de taille semblable dans un même batch.
    """
    indexed_messages = list(enumerate(messages))
    if lengths is not None:
        indexed_messages.sort(key=lambda item: lengths[item[0]])
    return [indexed_messages[offset:offset + batch_size] for offset in range(0, len(indexed_messages), batch_size)]


def measure_batches(
    processor: Any,
    batches: list[list[tuple[int, list[dict[str, str]]]]],
    batch_size: int,
    thinking: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    total_real_tokens = 0
    total_padded_tokens = 0
    for indexed_batch in batches:
        batch = [message for _, message in indexed_batch]
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
            "dataset_indices": [index for index, _ in indexed_batch],
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
    lengths = token_lengths(processor, messages, args.thinking)

    print(f"Tokenizer : {args.model}")
    print(f"{len(messages)} exemples ; mode={args.mode} ; thinking={args.thinking}")
    print("batch  stratégie         batches  tokens réels  tokens padded  padding  waste global  waste moyen  p95 waste")
    for batch_size in dict.fromkeys(args.batch_sizes):
        strategies = (
            ("ordre dataset", make_batches(messages, batch_size)),
            ("tailles proches", make_batches(messages, batch_size, lengths)),
        )
        for strategy, batches in strategies:
            summary, details = measure_batches(processor, batches, batch_size, args.thinking)
            print(
                f"{summary['batch_size']:>5}  {strategy:<16}  {summary['batches']:>7}  {summary['real_tokens']:>13}"
                f"  {summary['padded_tokens']:>13}  {summary['padding_tokens']:>7}"
                f"  {summary['padding_waste']:>11.2%}  {summary['mean_batch_padding_waste']:>11.2%}"
                f"  {summary['p95_batch_padding_waste']:>8.2%}"
            )
            if args.show_batches:
                print(json.dumps({"strategy": strategy, "batches": details}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
