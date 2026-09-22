"""Évalue un checkpoint text-to-SQL en zero-shot ou few-shot sur le split test.

Les prédictions et les métriques sont écrites séparément afin que chaque run soit
reproductible et vérifiable sans réexécuter le modèle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "00_shared" / "01_python"))
from sql_utils import normalize_sql

INPUTS = ROOT / "data" / "07_evaluation_dataset" / "03_production" / "test_inputs.jsonl"
GOLD = ROOT / "data" / "07_evaluation_dataset" / "03_production" / "test_gold.jsonl"
FEW_SHOT_SOURCE = ROOT / "data" / "06_training_dataset" / "03_production" / "train.jsonl"
DATABASES = ROOT / "BRUT_spider-original" / "data" / "spider_data" / "test_database"
OUTPUT_ROOT = ROOT / "evaluation_brut_models" / "runs"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_sql(text: str) -> str:
    """Extrait une requête SQL d'une sortie modèle sans masquer les erreurs."""
    text = text.strip()
    text = re.sub(r"^```(?:sql|sqlite)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"^SQL(?:Query)?\s*:\s*", "", text, flags=re.IGNORECASE)
    return normalize_sql(text.rstrip(";"))


def is_read_only_sql(sql: str) -> bool:
    """Accepte uniquement SELECT ou WITH ... SELECT, pour protéger les bases gold."""
    compact = sql.lstrip().upper()
    return compact.startswith("SELECT ") or compact.startswith("WITH ") or compact == "SELECT" or compact == "WITH"


def execute_sql(database: Path, sql: str) -> tuple[bool, list[list[Any]] | None, str | None]:
    if not sql or not is_read_only_sql(sql):
        return False, None, "requête non lecture seule"
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        cursor = connection.execute(sql)
        # json permet une comparaison stable même pour les valeurs NULL et numériques.
        return True, [list(row) for row in cursor.fetchall()], None
    except sqlite3.Error as error:
        return False, None, str(error)
    finally:
        connection.close()


def is_syntax_error(error: str | None) -> bool:
    """Classe les diagnostics SQLite de parsing, sans confondre schéma et runtime."""
    if error is None:
        return False
    lowered = error.lower()
    return any(marker in lowered for marker in (
        "syntax error", "incomplete input", "unrecognized token", "near ", "incomplete statement",
    ))


def database_path(db_id: str) -> Path:
    path = DATABASES / db_id / f"{db_id}.sqlite"
    if not path.is_file():
        raise FileNotFoundError(f"Base SQLite introuvable pour {db_id} : {path}")
    return path


def stable_few_shot_examples(source_rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, str]]:
    """Prélève des démonstrations few-shot, de façon déterministe et en lecture seule."""
    candidates = []
    for row in source_rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            continue
        user, assistant = messages[1], messages[2]
        if user.get("role") == "user" and assistant.get("role") == "assistant":
            candidates.append({"id": f"demonstration:{len(candidates)}", "user": user["content"], "sql": assistant["content"]})
    if count > len(candidates):
        raise ValueError(f"few-shot-k={count}, mais seulement {len(candidates)} démonstrations valides.")
    ranked = sorted(candidates, key=lambda item: hashlib.sha256(f"{seed}:{item['user']}".encode()).hexdigest())
    return ranked[:count]


def make_messages(input_row: dict[str, Any], demonstrations: list[dict[str, str]]) -> list[dict[str, str]]:
    messages = list(input_row["messages"])
    if demonstrations:
        messages = [messages[0]] + [
            message
            for demo in demonstrations
            for message in (
                {"role": "user", "content": demo["user"]},
                {"role": "assistant", "content": demo["sql"]},
            )
        ] + [messages[1]]
    return messages


def generate_batch(model_path: Path, message_batches: list[list[dict[str, str]]], max_new_tokens: int, device: str, thinking: bool) -> tuple[list[str], float, float | None]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as error:
        raise RuntimeError("Installez torch et transformers pour lancer une génération locale.") from error

    # Le chargement est mémorisé sur la fonction : un seul checkpoint par processus.
    cache_key = (str(model_path), device)
    if getattr(generate_batch, "cache_key", None) != cache_key:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        # Les RTX Turing (dont la RTX 2080) n'ont pas de calcul BF16 natif,
        # alors que les checkpoints Qwen sont déclarés en BF16.
        if device != "cpu" and torch.cuda.is_available():
            capability_major = torch.cuda.get_device_capability()[0]
            model_dtype = torch.bfloat16 if capability_major >= 8 else torch.float16
        else:
            model_dtype = torch.float32
        kwargs: dict[str, Any] = {"local_files_only": True, "dtype": model_dtype}
        if device == "auto":
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if device != "auto":
            model.to(device)
        model.eval()
        generate_batch.cache_key, generate_batch.processor, generate_batch.model = cache_key, processor, model
    processor, model = generate_batch.processor, generate_batch.model
    processor.tokenizer.padding_side = "left"
    inputs = processor.apply_chat_template(
        message_batches, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", enable_thinking=thinking, processor_kwargs={"padding": True},
    )
    # Le processeur multimodal Qwen ajoute ce champ même pour un prompt texte,
    # mais le modèle text-to-SQL ne l'accepte pas dans generate().
    inputs.pop("mm_token_type_ids", None)
    # Avec device_map=auto, le modèle est placé sur CUDA mais les entrées restent
    # sur CPU ; les déplacer vers le premier device du modèle évite un fallback.
    input_device = next(model.parameters()).device if device == "auto" else torch.device(device)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    cuda_active = torch.cuda.is_available()
    if cuda_active:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if cuda_active:
        torch.cuda.synchronize()
        peak_vram_mib: float | None = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
    else:
        peak_vram_mib = None
    generated = output[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True), round((time.perf_counter() - started) * 1000, 2), peak_vram_mib


def score(predictions: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_by_id = {row["id"]: row for row in gold_rows}
    if set(row["id"] for row in predictions) != set(gold_by_id):
        raise ValueError("Les identifiants de prédictions ne correspondent pas exactement au fichier gold.")
    results = []
    for prediction in predictions:
        gold = gold_by_id[prediction["id"]]
        predicted_sql = clean_sql(prediction.get("sql", prediction.get("raw_output", "")))
        gold_sql = normalize_sql(gold["sql"])
        database = database_path(gold["db_id"])
        pred_ok, pred_rows, error = execute_sql(database, predicted_sql)
        gold_ok, gold_rows_result, gold_error = execute_sql(database, gold_sql)
        if not gold_ok:
            raise RuntimeError(f"Gold SQL invalide pour {gold['id']} : {gold_error}")
        results.append({
            "id": gold["id"], "db_id": gold["db_id"], "predicted_sql": predicted_sql,
            "gold_sql": gold_sql, "exact_match": predicted_sql.lower() == gold_sql.lower(),
            "execution_match": pred_ok and pred_rows == gold_rows_result,
            "syntax_valid": not is_syntax_error(error),
            "execution_success": pred_ok,
            "execution_error": error,
        })
    total = len(results)
    metrics = {
        "examples": total,
        "exact_match": sum(row["exact_match"] for row in results) / total if total else 0,
        "execution_accuracy": sum(row["execution_match"] for row in results) / total if total else 0,
        "syntactically_valid_sql_rate": sum(row["syntax_valid"] for row in results) / total if total else 0,
        "execution_success_rate": sum(row["execution_success"] for row in results) / total if total else 0,
    }
    return results, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Checkpoint local Qwen ; requis sans --predictions.")
    parser.add_argument("--mode", choices=("zero-shot", "few-shot"), default="zero-shot")
    parser.add_argument("--few-shot-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cuda ou cpu")
    parser.add_argument("--max-new-tokens", type=int,
                        help="Plafond de sortie ; défaut : 1024 sans thinking, 2048 avec thinking.")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False,
                        help="Active le reasoning du modèle pendant la génération.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 impose une génération déterministe.")
    parser.add_argument("--environment", choices=("auto", "local", "scaleway"), default="auto")
    parser.add_argument("--limit", type=int, help="Limite utile pour un smoke test.")
    parser.add_argument("--predictions", type=Path, help="JSONL existant (id, sql ou raw_output) à scorer sans modèle.")
    parser.add_argument("--run-name", help="Nom de dossier ; défaut : modèle-mode-date.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    if not args.predictions and not args.model:
        raise ValueError("--model est requis pour générer des prédictions.")
    if args.temperature != 0:
        raise ValueError("Ce protocole de baseline impose --temperature 0.")
    if args.max_new_tokens is None:
        args.max_new_tokens = 2048 if args.thinking else 1024
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens doit être au moins 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size doit être au moins 1.")
    inputs, gold = load_jsonl(INPUTS), load_jsonl(GOLD)
    if args.limit:
        inputs, gold = inputs[:args.limit], gold[:args.limit]
    # UTC, lisible dans les noms de dossiers et sans caractères interdits sous Windows.
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")
    model_name = args.model.name if args.model else args.predictions.stem
    run_name = args.run_name or f"{model_name}-{args.mode}-{timestamp}"
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    batch_latencies: list[float] = []
    if args.predictions:
        predictions = load_jsonl(args.predictions)
    else:
        demonstrations = stable_few_shot_examples(load_jsonl(FEW_SHOT_SOURCE), args.few_shot_k, args.seed) if args.mode == "few-shot" else []
        predictions = []
        for offset in range(0, len(inputs), args.batch_size):
            batch = inputs[offset:offset + args.batch_size]
            raw_outputs, latency_ms, peak_vram_mib = generate_batch(args.model, [make_messages(row, demonstrations) for row in batch], args.max_new_tokens, args.device, args.thinking)
            batch_latencies.append(latency_ms)
            for row, raw_output in zip(batch, raw_outputs):
                predictions.append({"id": row["id"], "db_id": row["db_id"], "raw_output": raw_output, "sql": clean_sql(raw_output), "batch_latency_ms": latency_ms, "peak_vram_mib": peak_vram_mib, "batch_size": len(batch)})
            print(f"[{min(offset + len(batch), len(inputs))}/{len(inputs)}]", flush=True)
        write_jsonl(run_dir / "few_shot_examples.jsonl", demonstrations)
    results, metrics = score(predictions, gold)
    batch_latency_values = [row["batch_latency_ms"] for row in predictions if isinstance(row.get("batch_latency_ms"), (int, float))]
    vram_values = [row["peak_vram_mib"] for row in predictions if isinstance(row.get("peak_vram_mib"), (int, float))]
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    write_jsonl(run_dir / "results.jsonl", results)
    manifest = {
        "created_at_utc": timestamp, "mode": args.mode, "model": str(args.model) if args.model else None,
        "predictions_source": str(args.predictions) if args.predictions else None,
        "few_shot_k": args.few_shot_k if args.mode == "few-shot" else 0, "seed": args.seed,
        "temperature": args.temperature, "max_new_tokens": args.max_new_tokens, "thinking_enabled": args.thinking,
        "batch_size": args.batch_size,
        "environment": args.environment,
        "execution_hostname": socket.gethostname(),
        "execution_platform": platform.platform(),
        "inputs": str(INPUTS.relative_to(ROOT)), "inputs_sha256": file_hash(INPUTS),
        "gold": str(GOLD.relative_to(ROOT)),
        "few_shot_source": str(FEW_SHOT_SOURCE.relative_to(ROOT)),
        "few_shot_source_sha256": file_hash(FEW_SHOT_SOURCE), "examples": len(inputs),
        "total_execution_seconds": round(time.perf_counter() - run_started, 2),
        "mean_batch_latency_ms": round(sum(batch_latency_values) / len(batch_latency_values), 2) if batch_latency_values else None,
        "p95_batch_latency_ms": sorted(batch_latency_values)[max(0, int(len(batch_latency_values) * .95) - 1)] if batch_latency_values else None,
        "throughput_examples_per_second": round(len(predictions) / sum(batch_latencies) * 1000, 3) if batch_latencies else None,
        "peak_vram_mib": max(vram_values) if vram_values else None,
        **metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
