"""Génère et/ou score un checkpoint text-to-SQL sur le split test.

Les phases peuvent être séparées : ``generate`` ne requiert ni gold SQL ni
bases SQLite ; ``evaluate`` ne charge aucun modèle. Le manifeste placé à côté
des prédictions permet de déplacer un run GPU puis de le scorer localement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import re
import sqlite3
import subprocess
import sys
import threading
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
SQL_EXECUTION_TIMEOUT_SECONDS = 5.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def optional_file_hash(path: Path) -> str | None:
    return file_hash(path) if path.is_file() else None


def clean_sql(text: str) -> str:
    """Extrait une requête SQL, en retirant un éventuel raisonnement Qwen."""
    text = text.strip()
    # Avec enable_thinking=True, Qwen renvoie généralement le raisonnement dans
    # ce bloc avant sa réponse. Il est conservé dans raw_output pour audit, mais
    # ne doit jamais parvenir à SQLite. Un bloc incomplet est volontairement
    # conservé : le scorer le signalera alors comme une sortie invalide.
    thinking_block = re.match(r"^\s*<think>.*?</think>\s*", text, flags=re.IGNORECASE | re.DOTALL)
    if thinking_block:
        text = text[thinking_block.end():]
    text = re.sub(r"^```(?:sql|sqlite)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"^SQL(?:Query)?\s*:\s*", "", text, flags=re.IGNORECASE)
    return normalize_sql(text.rstrip(";"))


def is_read_only_sql(sql: str) -> bool:
    """Accepte uniquement SELECT ou WITH ... SELECT, pour protéger les bases gold."""
    compact = sql.lstrip().upper()
    return compact.startswith("SELECT ") or compact.startswith("WITH ") or compact == "SELECT" or compact == "WITH"


def execute_sql(database: Path, sql: str) -> tuple[bool, list[list[Any]] | None, str | None, float]:
    """Exécute une requête lecture seule, avec un délai maximal explicite."""
    started = time.perf_counter()
    if not sql or not is_read_only_sql(sql):
        return False, None, "requête non lecture seule", round((time.perf_counter() - started) * 1000, 2)
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        deadline = time.perf_counter() + SQL_EXECUTION_TIMEOUT_SECONDS
        connection.set_progress_handler(lambda: int(time.perf_counter() >= deadline), 1_000)
        cursor = connection.execute(sql)
        # json permet une comparaison stable même pour les valeurs NULL et numériques.
        return True, [list(row) for row in cursor.fetchall()], None, round((time.perf_counter() - started) * 1000, 2)
    except sqlite3.Error as error:
        message = (f"timeout après {SQL_EXECUTION_TIMEOUT_SECONDS:g}s" if "interrupted" in str(error).lower()
                   else str(error))
        return False, None, message, round((time.perf_counter() - started) * 1000, 2)
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


def sql_token_count(sql: str) -> int:
    """Compte les tokens lexicaux SQL, indépendamment du tokenizer du modèle."""
    return len(re.findall(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`|\b\w+\b|<=|>=|<>|!=|==|[-+*/(),.;=<>]", sql))


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


def group_inputs_by_prompt_length(
    model_path: Path,
    inputs: list[dict[str, Any]],
    demonstrations: list[dict[str, str]],
    thinking: bool,
) -> list[tuple[int, dict[str, Any]]]:
    """Trie une copie des entrées par longueur réelle du prompt tokenisé."""
    try:
        from transformers import AutoProcessor
    except ImportError as error:
        raise RuntimeError("Installez transformers pour regrouper les batches par taille.") from error

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    lengths: list[tuple[int, int, dict[str, Any]]] = []
    for index, row in enumerate(inputs):
        encoded = processor.apply_chat_template(
            make_messages(row, demonstrations),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=thinking,
        )
        lengths.append((int(encoded["attention_mask"].sum().item()), index, row))
    # L'index sert de second critère pour garder un ordre déterministe en cas
    # de prompts de même longueur.
    return [(index, row) for _, index, row in sorted(lengths, key=lambda item: (item[0], item[1]))]


def nvidia_smi_stats(device: Any) -> dict[str, float]:
    """Lit les compteurs GPU disponibles via nvidia-smi."""
    device_index = device.index if getattr(device, "index", None) is not None else 0
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={device_index}",
             "--query-gpu=utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        values = result.stdout.strip().splitlines()[0].split(",")
        names = ("gpu_utilization_percent", "gpu_power_w")
        return {
            name: float(value.strip())
            for name, value in zip(names, values)
            if value.strip().lower() not in {"", "n/a", "[not supported]"}
        }
    except (FileNotFoundError, IndexError, OSError, subprocess.SubprocessError, ValueError):
        return {}


def collect_gpu_stats(device: Any, stop_event: threading.Event, samples: list[dict[str, float]]) -> None:
    """Échantillonne les compteurs GPU pendant une génération."""
    while not stop_event.is_set():
        stats = nvidia_smi_stats(device)
        if stats:
            samples.append(stats)
        # nvidia-smi fournit des compteurs sur une fenêtre d'environ une seconde.
        stop_event.wait(1.0)


def percentile(values: list[float | int], quantile: float) -> float | int | None:
    """Percentile au rang le plus proche, stable aussi pour les petits runs."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def without_none(values: dict[str, Any]) -> dict[str, Any]:
    """Évite les rubriques de télémétrie vides dans les manifests."""
    return {key: value for key, value in values.items() if value is not None}


def organize_generation_manifest(flat: dict[str, Any]) -> dict[str, Any]:
    """Structure le manifeste de génération par domaine fonctionnel."""
    return {
        "artifact": {"type": flat["artifact_type"], "created_at_utc": flat["created_at_utc"], "phase": flat["phase"]},
        "model": without_none({"path": flat["model"], "config_sha256": flat["model_config_sha256"]}),
        "configuration": {"mode": flat["mode"], "few_shot_k": flat["few_shot_k"], "seed": flat["seed"],
                            "temperature": flat["temperature"], "max_new_tokens": flat["max_new_tokens"],
                            "thinking_enabled": flat["thinking_enabled"], "batch_size": flat["batch_size"],
                            "group_batches_by_length": flat["group_batches_by_length"],
                            "save_prompts": flat["save_prompts"]},
        "environment": {"target": flat["environment"], "hostname": flat["generation_hostname"], "platform": flat["generation_platform"]},
        "dataset": {"inputs": flat["inputs"], "inputs_sha256": flat["inputs_sha256"], "examples": flat["examples"]},
        "generation": {
            "duration_seconds": flat["generation_seconds"],
            "throughput": without_none({"examples_per_second": flat["generation_examples_per_second"], "active_generation_output_tokens_per_second": flat["generation_output_tokens_per_second"], "active_generation_total_tokens_per_second": flat["total_tokens_per_second"]}),
            "tokens": {"input": without_none({key: flat.get(f"{key}_input_tokens") for key in ("total", "mean", "p95")}),
                       "output": without_none({"total": flat["generated_output_tokens"], **{key: flat.get(f"{key}_output_tokens") for key in ("mean", "p95", "max")}}),
                       "padding": {"tokens": flat["padding_tokens"], "rate": flat["padding_rate"]}},
            "latency_ms": without_none({key: flat.get(f"{key}_example_latency_ms") for key in ("mean", "p95")}),
            "reliability": {key: flat[key] for key in ("truncated_generation_count", "truncation_rate", "failed_generations", "oom_count")},
        },
        "gpu": without_none({
            "utilization_percent": without_none({"mean": flat.get("mean_gpu_utilization_percent"), "peak": flat.get("peak_gpu_utilization_percent")}),
            "energy_wh": without_none({"total": flat.get("gpu_energy_wh"), "per_example": flat.get("energy_per_example_wh")}),
            "vram_mib": without_none({"mean": flat.get("mean_vram_mib"), "peak": flat.get("peak_vram_mib")}),
        }),
    }


def organize_evaluation_manifest(flat: dict[str, Any]) -> dict[str, Any]:
    """Structure les métriques SQL pour un usage analytique direct."""
    return {
        "artifact": {"created_at_utc": flat["created_at_utc"], "phase": flat["phase"]},
        "source": {
            "predictions": flat["predictions_source"],
            "generation": without_none({"manifest_sha256": flat.get("source_generation_manifest_sha256"),
                                           "model_path": flat.get("source_model_path"),
                                           "model_config_sha256": flat.get("source_model_config_sha256")}),
        },
        "configuration": {key: flat[key] for key in ("mode", "few_shot_k", "seed", "temperature", "max_new_tokens", "thinking_enabled", "batch_size", "group_batches_by_length", "environment")},
        "environment": {"hostname": flat["execution_hostname"], "platform": flat["execution_platform"]},
        "dataset": {key: flat[key] for key in ("inputs", "inputs_sha256", "gold", "few_shot_source", "few_shot_source_sha256")},
        "evaluation": {
            "examples": flat["examples"], "duration_seconds": flat["evaluation_seconds"],
            "examples_per_second": flat["evaluation_examples_per_second"], "total_execution_seconds": flat["total_execution_seconds"],
            "accuracy": {"exact_match": {"rate": flat["exact_match"], "count": flat["exact_match_count"]},
                         "execution": {"rate": flat["execution_accuracy"], "correct_count": flat["execution_correct_count"], "success_rate": flat["execution_success_rate"], "success_count": flat["execution_success_count"]},
                         "syntax": {"valid_rate": flat["syntactically_valid_sql_rate"], "valid_count": flat["syntactically_valid_sql_count"], "invalid_count": flat["invalid_sql_count"]}},
            "errors": {key: flat[key] for key in ("execution_error_count", "timeout_count", "empty_prediction_count")},
            "predicted_sql": {"mean_length_chars": flat["mean_sql_length_chars"], "tokens": {"mean": flat["mean_sql_tokens"], "p95": flat["p95_sql_tokens"]}},
            "execution_time_ms": {"median": flat["median_execution_time_ms"], "p95": flat["p95_execution_time_ms"], "max": flat["max_execution_time_ms"]},
        },
    }


def generate_batch(model_path: Path, message_batches: list[list[dict[str, str]]], max_new_tokens: int, device: str, thinking: bool) -> tuple[list[str], list[int], list[int], int, float, float | None, list[dict[str, float]]]:
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
    attention_mask = inputs.get("attention_mask")
    input_token_counts = (attention_mask.sum(dim=1).tolist() if attention_mask is not None
                          else [int(inputs["input_ids"].shape[1])] * len(message_batches))
    padding_tokens = int(inputs["input_ids"].numel() - sum(input_token_counts))
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
    gpu_samples: list[dict[str, float]] = []
    sampling_stop = threading.Event()
    sampler: threading.Thread | None = None
    if cuda_active:
        sampler = threading.Thread(
            target=collect_gpu_stats,
            args=(input_device, sampling_stop, gpu_samples),
            daemon=True,
        )
        sampler.start()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if cuda_active:
        torch.cuda.synchronize()
        sampling_stop.set()
        if sampler is not None:
            sampler.join()
        peak_vram_mib: float | None = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
    else:
        peak_vram_mib = None
    generated = output[:, inputs["input_ids"].shape[1]:]
    pad_token_id = processor.tokenizer.pad_token_id
    output_token_counts = [int(generated.shape[1])] * len(generated) if pad_token_id is None else generated.ne(pad_token_id).sum(dim=1).tolist()
    return processor.batch_decode(generated, skip_special_tokens=True), output_token_counts, input_token_counts, padding_tokens, round((time.perf_counter() - started) * 1000, 2), peak_vram_mib, gpu_samples


def score(predictions: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_by_id = {row["id"]: row for row in gold_rows}
    if set(row["id"] for row in predictions) != set(gold_by_id):
        raise ValueError("Les identifiants de prédictions ne correspondent pas exactement au fichier gold.")
    results = []
    for prediction in predictions:
        gold = gold_by_id[prediction["id"]]
        # Repartir de la sortie brute applique les correctifs d'extraction aux
        # anciens runs (notamment ceux générés avec le thinking). Les clés
        # nettoyées restent acceptées si raw_output n'est pas disponible.
        predicted_sql = clean_sql(prediction.get("raw_output", prediction.get("clean_sql", prediction.get("sql_normalized", prediction.get("sql", "")))))
        gold_sql = normalize_sql(gold["sql"])
        database = database_path(gold["db_id"])
        pred_ok, pred_rows, error, execution_time_ms = execute_sql(database, predicted_sql)
        gold_ok, gold_rows_result, gold_error, _gold_execution_time_ms = execute_sql(database, gold_sql)
        if not gold_ok:
            raise RuntimeError(f"Gold SQL invalide pour {gold['id']} : {gold_error}")
        results.append({
            "id": gold["id"], "db_id": gold["db_id"], "predicted_sql": predicted_sql,
            "gold_sql": gold_sql, "exact_match": predicted_sql.lower() == gold_sql.lower(),
            "execution_match": pred_ok and pred_rows == gold_rows_result,
            "syntax_valid": bool(predicted_sql) and not is_syntax_error(error),
            "execution_success": pred_ok,
            "execution_time_ms": execution_time_ms,
            "execution_error": error,
        })
    total = len(results)
    exact_match_count = sum(row["exact_match"] for row in results)
    execution_correct_count = sum(row["execution_match"] for row in results)
    syntactically_valid_sql_count = sum(row["syntax_valid"] for row in results)
    execution_success_count = sum(row["execution_success"] for row in results)
    invalid_sql_count = total - syntactically_valid_sql_count
    execution_error_count = sum(row["execution_error"] is not None for row in results)
    timeout_count = sum("timeout" in (row["execution_error"] or "").lower() for row in results)
    empty_prediction_count = sum(not row["predicted_sql"].strip() for row in results)
    sql_lengths = [len(row["predicted_sql"]) for row in results]
    sql_token_counts = [sql_token_count(row["predicted_sql"]) for row in results]
    execution_times = [row["execution_time_ms"] for row in results]
    metrics = {
        "examples": total,
        "exact_match": exact_match_count / total if total else 0,
        "execution_accuracy": execution_correct_count / total if total else 0,
        "syntactically_valid_sql_rate": syntactically_valid_sql_count / total if total else 0,
        "execution_success_rate": execution_success_count / total if total else 0,
        "exact_match_count": exact_match_count,
        "execution_correct_count": execution_correct_count,
        "syntactically_valid_sql_count": syntactically_valid_sql_count,
        "execution_success_count": execution_success_count,
        "invalid_sql_count": invalid_sql_count,
        "execution_error_count": execution_error_count,
        "timeout_count": timeout_count,
        "empty_prediction_count": empty_prediction_count,
        "evaluation_failed_count": 0,
        "mean_sql_length_chars": round(sum(sql_lengths) / len(sql_lengths), 2) if sql_lengths else None,
        "mean_sql_tokens": round(sum(sql_token_counts) / len(sql_token_counts), 2) if sql_token_counts else None,
        "p95_sql_tokens": percentile(sql_token_counts, .95),
        "median_execution_time_ms": percentile(execution_times, .50),
        "p95_execution_time_ms": percentile(execution_times, .95),
        "max_execution_time_ms": max(execution_times, default=None),
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
    parser.add_argument("--group-batches-by-length", action=argparse.BooleanOptionalAction, default=False,
                        help="Regroupe les prompts de longueur proche pour réduire le padding.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 impose une génération déterministe.")
    parser.add_argument("--environment", choices=("auto", "local", "scaleway"), default="auto")
    parser.add_argument("--limit", type=int, help="Limite utile pour un smoke test.")
    parser.add_argument("--phase", choices=("generate", "evaluate", "both"), default="both",
                        help="generate (GPU), evaluate (local) ou both (compatibilité ; défaut).")
    parser.add_argument("--predictions", type=Path,
                        help="JSONL existant (id, clean_sql ou raw_output) à scorer ; requis avec --phase evaluate.")
    parser.add_argument("--save-prompts", action="store_true",
                        help="Inclut les messages effectivement envoyés au modèle dans predictions.jsonl.")
    parser.add_argument("--output-dir", type=Path,
                        help="Dossier de sortie explicite. Utile pour écrire les métriques dans un run copié.")
    parser.add_argument("--run-name", help="Nom de dossier ; défaut : modèle-mode-date.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    if args.phase == "generate" and not args.model:
        raise ValueError("--model est requis avec --phase generate.")
    if args.phase == "evaluate" and not args.predictions:
        raise ValueError("--predictions est requis avec --phase evaluate.")
    if args.phase == "both" and not args.predictions and not args.model:
        raise ValueError("--model est requis pour générer des prédictions.")
    if args.phase == "both" and args.predictions:
        args.phase = "evaluate"
    if args.temperature != 0:
        raise ValueError("Ce protocole de baseline impose --temperature 0.")
    if args.max_new_tokens is None:
        args.max_new_tokens = 2048 if args.thinking else 1024
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens doit être au moins 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size doit être au moins 1.")
    inputs = load_jsonl(INPUTS) if args.phase in ("generate", "both") else []
    gold = load_jsonl(GOLD) if args.phase in ("evaluate", "both") else []
    if args.limit:
        inputs, gold = inputs[:args.limit], gold[:args.limit]
    # UTC : lisible, sans ambiguïté et disponible sur toutes les plateformes.
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")
    model_name = args.model.name if args.model else args.predictions.parent.name
    run_name = args.run_name or f"{model_name}-{args.mode}-{timestamp}"
    run_dir = args.output_dir or (OUTPUT_ROOT / run_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    batch_latencies: list[float] = []
    batch_peak_vram_mib: list[float] = []
    gpu_samples: list[dict[str, float]] = []
    total_padding_tokens = 0
    failed_generations = 0
    oom_count = 0
    retry_count = 0
    source_generation_manifest: dict[str, Any] | None = None
    source_generation_manifest_sha256: str | None = None
    source_model_path: str | None = None
    source_model_config_sha256: str | None = None
    if args.phase == "evaluate":
        predictions = load_jsonl(args.predictions)
        source_manifest_path = args.predictions.parent / "generation_manifest.json"
        if source_manifest_path.is_file():
            source_generation_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            source_generation_manifest_sha256 = file_hash(source_manifest_path)
            expected_hash = source_generation_manifest.get("dataset", {}).get("inputs_sha256", source_generation_manifest.get("inputs_sha256"))
            source_model = source_generation_manifest.get("model")
            if isinstance(source_model, dict):
                source_model_path = source_model.get("path")
                source_model_config_sha256 = source_model.get("config_sha256")
            elif isinstance(source_model, str):  # Compatibilité avec les manifests antérieurs.
                source_model_path = source_model
                source_model_config_sha256 = source_generation_manifest.get("model_config_sha256")
            if expected_hash and expected_hash != file_hash(INPUTS):
                raise ValueError("Le hash de test_inputs.jsonl ne correspond pas au manifeste de génération.")
        if args.limit:
            allowed_ids = {row["id"] for row in gold}
            predictions = [row for row in predictions if row.get("id") in allowed_ids]
        # Le fichier de prédictions est la source de vérité pour la portée
        # d'une évaluation séparée. Cela couvre aussi les runs partiels qui ne
        # correspondent pas nécessairement aux N premiers exemples du test.
        prediction_ids = {row.get("id") for row in predictions}
        gold = [row for row in gold if row["id"] in prediction_ids]
        print(f"Évaluation de {len(predictions)} prédictions.")
    else:
        demonstrations = stable_few_shot_examples(load_jsonl(FEW_SHOT_SOURCE), args.few_shot_k, args.seed) if args.mode == "few-shot" else []
        indexed_inputs = (group_inputs_by_prompt_length(args.model, inputs, demonstrations, args.thinking)
                          if args.group_batches_by_length else list(enumerate(inputs)))
        if args.group_batches_by_length:
            print("Batches regroupés par longueur de prompt pour réduire le padding.")
        predictions_by_index: list[dict[str, Any] | None] = [None] * len(inputs)
        for offset in range(0, len(indexed_inputs), args.batch_size):
            indexed_batch = indexed_inputs[offset:offset + args.batch_size]
            batch = [row for _, row in indexed_batch]
            try:
                raw_outputs, output_token_counts, input_token_counts, padding_tokens, latency_ms, peak_vram_mib, batch_gpu_samples = generate_batch(args.model, [make_messages(row, demonstrations) for row in batch], args.max_new_tokens, args.device, args.thinking)
            except RuntimeError as error:
                failed_generations += len(batch)
                oom_count += int("out of memory" in str(error).lower())
                raise
            batch_latencies.append(latency_ms)
            gpu_samples.extend(batch_gpu_samples)
            total_padding_tokens += padding_tokens
            if peak_vram_mib is not None:
                batch_peak_vram_mib.append(peak_vram_mib)
            # Tous les exemples d'un même batch terminent ensemble : leur
            # latence observée est donc celle du batch, pas une fraction de
            # celle-ci (qui représenterait un coût de débit, pas une latence).
            example_latency_ms = latency_ms
            for (input_index, row), raw_output, output_tokens, input_tokens in zip(indexed_batch, raw_outputs, output_token_counts, input_token_counts):
                prediction = {"id": row["id"], "db_id": row["db_id"], "raw_output": raw_output,
                              "clean_sql": clean_sql(raw_output), "input_tokens": input_tokens,
                              "output_tokens": output_tokens, "example_latency_ms": example_latency_ms,
                              "batch_latency_ms": latency_ms, "peak_vram_mib": peak_vram_mib,
                              "batch_size": len(batch)}
                if args.save_prompts:
                    prediction["messages"] = make_messages(row, demonstrations)
                predictions_by_index[input_index] = prediction
            print(f"[{min(offset + len(batch), len(indexed_inputs))}/{len(indexed_inputs)}]", flush=True)
        predictions = [prediction for prediction in predictions_by_index if prediction is not None]
        write_jsonl(run_dir / "few_shot_examples.jsonl", demonstrations)

    if args.phase != "evaluate":
        # Cet artefact est volontairement écrit avant tout scoring : c'est
        # celui à rapatrier depuis la machine GPU avec predictions.jsonl.
        write_jsonl(run_dir / "predictions.jsonl", predictions)
        generation_seconds = time.perf_counter() - run_started
        model_config = args.model / "config.json" if args.model else None
        input_token_values = [row["input_tokens"] for row in predictions if isinstance(row.get("input_tokens"), int)]
        output_token_values = [row["output_tokens"] for row in predictions if isinstance(row.get("output_tokens"), int)]
        example_latency_values = [row["example_latency_ms"] for row in predictions if isinstance(row.get("example_latency_ms"), (int, float))]
        def mean_gpu_stat(name: str) -> float | None:
            values = [sample[name] for sample in gpu_samples if name in sample]
            return round(sum(values) / len(values), 2) if values else None
        def peak_gpu_stat(name: str) -> float | None:
            values = [sample[name] for sample in gpu_samples if name in sample]
            return max(values) if values else None
        generation_manifest = {
            "artifact_type": "text_to_sql_predictions",
            "created_at_utc": timestamp,
            "phase": "generate",
            "model": str(args.model),
            "model_config_sha256": optional_file_hash(model_config),
            "mode": args.mode, "few_shot_k": args.few_shot_k if args.mode == "few-shot" else 0,
            "seed": args.seed, "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens, "thinking_enabled": args.thinking,
            "batch_size": args.batch_size, "group_batches_by_length": args.group_batches_by_length,
            "save_prompts": args.save_prompts,
            "environment": args.environment, "generation_hostname": socket.gethostname(),
            "generation_platform": platform.platform(),
            "inputs": str(INPUTS.relative_to(ROOT)), "inputs_sha256": file_hash(INPUTS),
            "examples": len(predictions),
            "generation_seconds": round(generation_seconds, 2),
            "generation_examples_per_second": round(len(predictions) / sum(batch_latencies) * 1000, 3) if batch_latencies else None,
            "total_input_tokens": sum(input_token_values),
            "padding_tokens": total_padding_tokens,
            "padding_rate": round(total_padding_tokens / (sum(input_token_values) + total_padding_tokens), 4) if input_token_values or total_padding_tokens else 0,
            "mean_input_tokens": round(sum(input_token_values) / len(input_token_values), 2) if input_token_values else None,
            "p50_input_tokens": percentile(input_token_values, .50),
            "p95_input_tokens": percentile(input_token_values, .95),
            "max_input_tokens": max(input_token_values, default=None),
            "generated_output_tokens": sum(output_token_values),
            "mean_output_tokens": round(sum(output_token_values) / len(output_token_values), 2) if output_token_values else None,
            "p50_output_tokens": percentile(output_token_values, .50),
            "p95_output_tokens": percentile(output_token_values, .95),
            "max_output_tokens": max(output_token_values, default=None),
            "generation_output_tokens_per_second": round(sum(output_token_values) / sum(batch_latencies) * 1000, 3) if batch_latencies else None,
            "total_tokens_per_second": round((sum(input_token_values) + sum(output_token_values)) / sum(batch_latencies) * 1000, 3) if batch_latencies else None,
            "mean_batch_latency_ms": round(sum(batch_latencies) / len(batch_latencies), 2) if batch_latencies else None,
            "p95_batch_latency_ms": percentile(batch_latencies, .95),
            "mean_example_latency_ms": round(sum(example_latency_values) / len(example_latency_values), 2) if example_latency_values else None,
            "p50_example_latency_ms": percentile(example_latency_values, .50),
            "p95_example_latency_ms": percentile(example_latency_values, .95),
            "p99_example_latency_ms": percentile(example_latency_values, .99),
            "failed_generations": failed_generations,
            "oom_count": oom_count,
            "retry_count": retry_count,
            "truncated_generation_count": sum(value >= args.max_new_tokens for value in output_token_values),
            "truncation_rate": round(sum(value >= args.max_new_tokens for value in output_token_values) / len(output_token_values), 4) if output_token_values else 0,
        }
        for output_name, source_name, aggregate in (
            ("mean_gpu_utilization_percent", "gpu_utilization_percent", mean_gpu_stat),
            ("peak_gpu_utilization_percent", "gpu_utilization_percent", peak_gpu_stat),
        ):
            value = aggregate(source_name)
            if value is not None:
                generation_manifest[output_name] = value
        if batch_peak_vram_mib:
            generation_manifest["mean_vram_mib"] = round(sum(batch_peak_vram_mib) / len(batch_peak_vram_mib), 2)
            generation_manifest["peak_vram_mib"] = max(batch_peak_vram_mib)
        mean_power_w = mean_gpu_stat("gpu_power_w")
        if mean_power_w is not None and batch_latencies:
            energy_wh = mean_power_w * sum(batch_latencies) / 3_600_000
            generation_manifest["gpu_energy_wh"] = round(energy_wh, 6)
            generation_manifest["energy_per_example_wh"] = round(energy_wh / len(predictions), 6) if predictions else None
        generation_manifest = organize_generation_manifest(generation_manifest)
        (run_dir / "generation_manifest.json").write_text(json.dumps(generation_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.phase == "generate":
            print(json.dumps(generation_manifest, ensure_ascii=False, indent=2))
            return

    results, metrics = score(predictions, gold)
    batch_latency_values = [row["batch_latency_ms"] for row in predictions if isinstance(row.get("batch_latency_ms"), (int, float))]
    output_token_values = [row["output_tokens"] for row in predictions if isinstance(row.get("output_tokens"), int)]
    vram_values = [row["peak_vram_mib"] for row in predictions if isinstance(row.get("peak_vram_mib"), (int, float))]
    write_jsonl(run_dir / "results.jsonl", results)
    total_execution_seconds = time.perf_counter() - run_started
    manifest = {
        "created_at_utc": timestamp, "phase": "evaluate", "mode": args.mode, "model": str(args.model) if args.model else None,
        "predictions_source": str(args.predictions) if args.predictions else None,
        "source_generation_manifest_sha256": source_generation_manifest_sha256,
        "source_model_path": source_model_path,
        "source_model_config_sha256": source_model_config_sha256,
        "few_shot_k": args.few_shot_k if args.mode == "few-shot" else 0, "seed": args.seed,
        "temperature": args.temperature, "max_new_tokens": args.max_new_tokens, "thinking_enabled": args.thinking,
        "batch_size": args.batch_size, "group_batches_by_length": args.group_batches_by_length,
        "environment": args.environment,
        "execution_hostname": socket.gethostname(),
        "execution_platform": platform.platform(),
        "inputs": str(INPUTS.relative_to(ROOT)), "inputs_sha256": file_hash(INPUTS),
        "gold": str(GOLD.relative_to(ROOT)),
        "few_shot_source": str(FEW_SHOT_SOURCE.relative_to(ROOT)),
        "few_shot_source_sha256": file_hash(FEW_SHOT_SOURCE), "examples": len(inputs),
        "evaluation_seconds": round(total_execution_seconds, 2),
        "evaluation_examples_per_second": round(len(predictions) / total_execution_seconds, 3) if predictions else None,
        "total_execution_seconds": round(total_execution_seconds, 2),
        **metrics,
    }
    if args.phase == "both":
        manifest.update({
            "mean_batch_latency_ms": round(sum(batch_latency_values) / len(batch_latency_values), 2) if batch_latency_values else None,
            "p95_batch_latency_ms": sorted(batch_latency_values)[max(0, int(len(batch_latency_values) * .95) - 1)] if batch_latency_values else None,
            "generation_examples_per_second": round(len(predictions) / sum(batch_latencies) * 1000, 3) if batch_latencies else None,
            "generated_output_tokens": sum(output_token_values) if output_token_values else None,
            "generation_output_tokens_per_second": round(sum(output_token_values) / sum(batch_latencies) * 1000, 3) if output_token_values and batch_latencies else None,
            "end_to_end_examples_per_second": round(len(predictions) / total_execution_seconds, 3) if predictions else None,
            "end_to_end_output_tokens_per_second": round(sum(output_token_values) / total_execution_seconds, 3) if output_token_values else None,
            "mean_gpu_utilization_percent": mean_gpu_stat("gpu_utilization_percent"),
            "peak_gpu_utilization_percent": peak_gpu_stat("gpu_utilization_percent"),
            "peak_vram_mib": max(vram_values) if vram_values else None,
        })
    manifest = organize_evaluation_manifest(manifest)
    (run_dir / "metrics.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
