"""Évalue un modèle OpenAI sur le protocole text-to-SQL via la Batch API.

Un batch OpenAI est asynchrone : lancez d'abord ``submit`` puis, lorsque le
statut est ``completed``, relancez avec ``retrieve`` et le dossier du run.
Les prompts, démonstrations et le scoring SQLite sont ceux du runner local.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_evaluation_brut_models"))

from assets.run_evaluation import (  # noqa: E402
    FEW_SHOT_SOURCE,
    GOLD,
    INPUTS,
    clean_sql,
    file_hash,
    load_jsonl,
    make_messages,
    organize_evaluation_manifest,
    organize_generation_manifest,
    score,
    stable_few_shot_examples,
    write_jsonl,
)

OUTPUT_ROOT = ROOT / "02_evaluation_brut_models" / "runs_openai"
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def load_key_from_dotenv() -> None:
    """Charge seulement OPENAI_API_KEY depuis .env si le shell ne l'a pas déjà."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    dotenv = ROOT / ".env"
    if not dotenv.is_file():
        return
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "OPENAI_API_KEY" and value.strip():
            os.environ["OPENAI_API_KEY"] = value.strip().strip('"').strip("'")
            return


def openai_client() -> Any:
    load_key_from_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY est requis (variable d'environnement ou fichier .env).")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Installez le SDK : .\\.venv\\Scripts\\python.exe -m pip install openai") from error
    return OpenAI()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")


def choose_action() -> str:
    print("\nAction Batch OpenAI :\n  1) Soumettre un nouveau batch\n  2) Récupérer et scorer un batch existant")
    choice = input("Action [1] : ").strip() or "1"
    actions = {"1": "submit", "2": "retrieve"}
    if choice not in actions:
        raise ValueError("Choisissez 1 ou 2.")
    return actions[choice]


def choose_model() -> str:
    value = input("Modèle OpenAI (ex. gpt-5.6-luna) : ").strip()
    if not value:
        raise ValueError("Un modèle OpenAI est requis.")
    return value


def choose_mode() -> str:
    print("\nMode d'évaluation :\n  1) zero-shot\n  2) few-shot\n  3) Les deux")
    choice = input("Mode [1] : ").strip() or "1"
    modes = {"1": "zero-shot", "2": "few-shot", "3": "both"}
    if choice not in modes:
        raise ValueError("Choisissez 1, 2 ou 3.")
    return modes[choice]


def choose_limit() -> int | None:
    total = len(load_jsonl(INPUTS))
    print(f"\nJeu de test : {total} exemples")
    print(f"  1) Tous les tests — {total} exemples (défaut)")
    print("  2) Un nombre précis — smoke test ou diagnostic")
    choice = input("Portée [1] : ").strip() or "1"
    if choice == "1":
        return None
    if choice == "2":
        value = input(f"Nombre d'exemples [1-{total}] : ").strip()
        try:
            limit = int(value)
        except ValueError as error:
            raise ValueError("Le nombre d'exemples doit être un entier.") from error
        if not 1 <= limit <= total:
            raise ValueError(f"Le nombre d'exemples doit être compris entre 1 et {total}.")
        return limit
    raise ValueError("Choisissez 1 ou 2.")


def choose_few_shot_k() -> int:
    value = input("Nombre d'exemples few-shot [2] : ").strip() or "2"
    try:
        count = int(value)
    except ValueError as error:
        raise ValueError("Le nombre d'exemples few-shot doit être un entier positif.") from error
    if count < 1:
        raise ValueError("Le nombre d'exemples few-shot doit être au moins 1.")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("submit", "retrieve"), help="Soumet ou récupère un batch ; choix interactif par défaut.")
    parser.add_argument("--model", help="ID du modèle OpenAI ; demandé interactivement lors d'une soumission.")
    parser.add_argument("--mode", choices=("zero-shot", "few-shot", "both"), help="Mode de prompting ; choix interactif par défaut.")
    parser.add_argument("--few-shot-k", type=int, help="Nombre de démonstrations few-shot (défaut interactif : 2).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", default="none", help="Effort de raisonnement OpenAI (défaut : none, baseline directe).")
    parser.add_argument("--limit", type=int, help="Smoke test seulement ; à omettre pour la baseline complète.")
    parser.add_argument("--run-dir", type=Path, help="Dossier d'un run soumis ; requis pour --action retrieve si non saisi interactivement.")
    parser.add_argument("--run-name", help="Nom de run sous runs_openai/ lors de la soumission.")
    parser.add_argument("--save-prompts", action=argparse.BooleanOptionalAction, default=False,
                        help="Ajoute les messages aux prédictions récupérées (le JSONL soumis les contient déjà).")
    return parser.parse_args()


def request_body(model: str, messages: list[dict[str, str]], max_output_tokens: int, reasoning_effort: str) -> dict[str, Any]:
    # Chat Completions est choisi car chaque réponse conserve directement les
    # compteurs prompt/completion et le texte à comparer aux runs Qwen.
    return {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_output_tokens,
        "reasoning_effort": reasoning_effort,
    }


def run_dir_for(name: str | None, model: str, mode: str) -> Path:
    if name:
        return OUTPUT_ROOT / name
    safe_model = "".join(char if char.isalnum() or char in "._-" else "-" for char in model)
    return OUTPUT_ROOT / f"batch-{utc_stamp()}" / f"{safe_model}-{mode}"


def submit(args: argparse.Namespace, model: str, mode: str, few_shot_k: int, limit: int | None) -> None:
    if args.max_output_tokens < 1:
        raise ValueError("--max-output-tokens doit être au moins 1.")
    inputs = load_jsonl(INPUTS)
    if limit is not None:
        inputs = inputs[:limit]
    demonstrations = stable_few_shot_examples(load_jsonl(FEW_SHOT_SOURCE), few_shot_k, args.seed) if mode == "few-shot" else []
    run_dir = run_dir_for(args.run_name, model, mode)
    run_dir.mkdir(parents=True, exist_ok=False)
    requests_path = run_dir / "batch_input.jsonl"
    request_rows = []
    for row in inputs:
        messages = make_messages(row, demonstrations)
        request_rows.append({
            "custom_id": row["id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": request_body(model, messages, args.max_output_tokens, args.reasoning_effort),
        })
    write_jsonl(requests_path, request_rows)
    write_jsonl(run_dir / "few_shot_examples.jsonl", demonstrations)

    client = openai_client()
    with requests_path.open("rb") as file_handle:
        uploaded = client.files.create(file=file_handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"benchmark": "text-to-sql", "mode": mode},
    )
    manifest = organize_generation_manifest({
        "artifact_type": "openai_batch_text_to_sql",
        "created_at_utc": utc_stamp(),
        "phase": "submitted",
        "model": f"openai/{model}",
        "model_config_sha256": None,
        "mode": mode,
        "few_shot_k": few_shot_k if mode == "few-shot" else 0,
        "seed": args.seed,
        "temperature": None,
        "max_new_tokens": args.max_output_tokens,
        "thinking_enabled": args.reasoning_effort != "none",
        "batch_size": None,
        "group_batches_by_length": False,
        "save_prompts": args.save_prompts,
        "environment": "openai-batch",
        "inputs": str(INPUTS.relative_to(ROOT)),
        "inputs_sha256": file_hash(INPUTS),
        "examples": len(inputs),
        "generation_hostname": socket.gethostname(),
        "generation_platform": platform.platform(),
        # La Batch API ne fournit ni latence par requête ni télémétrie machine.
        "generation_seconds": None,
        "generation_examples_per_second": None,
        "generation_output_tokens_per_second": None,
        "total_tokens_per_second": None,
        "total_input_tokens": None,
        "generated_output_tokens": None,
        "padding_tokens": 0,
        "padding_rate": 0,
        "failed_generations": 0,
        "oom_count": 0,
        "truncated_generation_count": 0,
        "truncation_rate": 0,
    })
    manifest["openai_batch"] = {
        "id": batch.id,
        "status": batch.status,
        "input_file_id": uploaded.id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "reasoning_effort": args.reasoning_effort,
    }
    (run_dir / "generation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Batch soumis : {batch.id} ({batch.status})")
    print(f"Run : {run_dir}")
    print("Quand le batch est terminé, récupérez-le avec :")
    print(f'  python 02_evaluation_brut_models\\run_openai.py --action retrieve --run-dir "{run_dir}"')


def content_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def retrieve(args: argparse.Namespace, run_dir: Path) -> None:
    manifest_path = run_dir / "generation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifeste introuvable : {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batch_info_manifest = manifest.get("openai_batch", {})
    batch_id = batch_info_manifest.get("id") if isinstance(batch_info_manifest, dict) else None
    if not isinstance(batch_id, str):
        raise ValueError("Le manifeste ne contient pas de batch_id OpenAI.")
    client = openai_client()
    batch = client.batches.retrieve(batch_id)
    batch_info = batch.model_dump(mode="json")
    (run_dir / "batch_status.json").write_text(json.dumps(batch_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not isinstance(batch_info_manifest, dict):
        batch_info_manifest = {}
        manifest["openai_batch"] = batch_info_manifest
    batch_info_manifest.update({
        "status": batch.status,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
    })
    if batch.status not in TERMINAL_STATUSES:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Batch {batch_id} : {batch.status}. Réessayez plus tard.")
        return
    if not batch.output_file_id:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Batch {batch_id} terminé avec le statut {batch.status}, sans fichier de sortie.")

    output_path = run_dir / "batch_output.jsonl"
    output_path.write_text(client.files.content(batch.output_file_id).text, encoding="utf-8")
    if batch.error_file_id:
        (run_dir / "batch_errors.jsonl").write_text(client.files.content(batch.error_file_id).text, encoding="utf-8")
    outputs = load_jsonl(output_path)
    outputs_by_id = {row.get("custom_id"): row for row in outputs}
    inputs = load_jsonl(INPUTS)
    demonstrations = load_jsonl(run_dir / "few_shot_examples.jsonl")
    expected = manifest.get("dataset", {}).get("examples")
    if isinstance(expected, int):
        inputs = inputs[:expected]
    started = time.perf_counter()
    predictions = []
    failed_requests = 0
    for input_row in inputs:
        output = outputs_by_id.get(input_row["id"])
        response = output.get("response") if isinstance(output, dict) else None
        body = response.get("body") if isinstance(response, dict) else None
        choices = body.get("choices") if isinstance(body, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message") if isinstance(choice, dict) else {}
        usage = body.get("usage") if isinstance(body, dict) else {}
        api_error = None
        if not isinstance(body, dict) or response.get("status_code") != 200:
            failed_requests += 1
            api_error = (output or {}).get("error") or (body or {}).get("error") or {"message": "réponse Batch absente"}
        prediction = {
            "id": input_row["id"],
            "db_id": input_row["db_id"],
            "raw_output": content_text(message if isinstance(message, dict) else {}),
            "clean_sql": clean_sql(content_text(message if isinstance(message, dict) else {})),
            "input_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
            "output_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
            "api_request_id": response.get("request_id") if isinstance(response, dict) else None,
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        }
        if api_error:
            prediction["api_error"] = api_error
        if args.save_prompts:
            prediction["messages"] = make_messages(input_row, demonstrations)
        predictions.append(prediction)
    write_jsonl(run_dir / "predictions.jsonl", predictions)

    results, metrics = score(predictions, load_jsonl(GOLD)[:len(inputs)])
    write_jsonl(run_dir / "results.jsonl", results)
    input_tokens = [row["input_tokens"] for row in predictions if isinstance(row.get("input_tokens"), int)]
    output_tokens = [row["output_tokens"] for row in predictions if isinstance(row.get("output_tokens"), int)]
    manifest["artifact"]["phase"] = "completed"
    manifest["openai_batch"].update({
        "request_counts": batch_info.get("request_counts"),
        "usage": batch_info.get("usage"),
    })
    manifest["generation"].update({
        "tokens": {
            "input": {"total": sum(input_tokens)},
            "output": {"total": sum(output_tokens)},
            "padding": {"tokens": 0, "rate": 0},
        },
        "reliability": {"truncated_generation_count": 0, "truncation_rate": 0,
                        "failed_generations": failed_requests, "oom_count": 0},
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evaluation_seconds = time.perf_counter() - started
    metrics_manifest = organize_evaluation_manifest({
        "created_at_utc": utc_stamp(),
        "phase": "evaluate",
        "mode": manifest.get("configuration", {}).get("mode"),
        "model": manifest.get("model", {}).get("path"),
        "predictions_source": str((run_dir / "predictions.jsonl").relative_to(ROOT)),
        "source_generation_manifest_sha256": file_hash(manifest_path),
        "source_model_path": manifest.get("model", {}).get("path"),
        "source_model_config_sha256": None,
        "few_shot_k": manifest.get("configuration", {}).get("few_shot_k", 0),
        "seed": manifest.get("configuration", {}).get("seed"),
        "temperature": None,
        "max_new_tokens": manifest.get("configuration", {}).get("max_new_tokens"),
        "thinking_enabled": manifest.get("configuration", {}).get("thinking_enabled", False),
        "batch_size": None,
        "group_batches_by_length": False,
        "environment": "openai-batch",
        "execution_hostname": socket.gethostname(),
        "execution_platform": platform.platform(),
        "inputs": str(INPUTS.relative_to(ROOT)),
        "inputs_sha256": file_hash(INPUTS),
        "gold": str(GOLD.relative_to(ROOT)),
        "few_shot_source": str(FEW_SHOT_SOURCE.relative_to(ROOT)),
        "few_shot_source_sha256": file_hash(FEW_SHOT_SOURCE),
        "examples": len(inputs),
        "evaluation_seconds": round(evaluation_seconds, 2),
        "evaluation_examples_per_second": round(len(predictions) / evaluation_seconds, 3) if predictions else None,
        "total_execution_seconds": round(evaluation_seconds, 2),
        "openai_batch_id": batch_id,
        "openai_batch_status": batch.status,
        "openai_input_tokens": sum(input_tokens),
        "openai_output_tokens": sum(output_tokens),
        "openai_failed_requests": failed_requests,
        **metrics,
    })
    metrics_manifest["openai_batch"] = {
        "id": batch_id,
        "status": batch.status,
        "input_tokens": sum(input_tokens),
        "output_tokens": sum(output_tokens),
        "failed_requests": failed_requests,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics_manifest, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    action = args.action or choose_action()
    if action == "retrieve":
        run_dir = args.run_dir
        if run_dir is None:
            value = input("Dossier du run OpenAI : ").strip()
            if not value:
                raise ValueError("Un dossier de run est requis.")
            run_dir = Path(value)
        retrieve(args, run_dir.resolve())
        return

    model = args.model or choose_model()
    mode_choice = args.mode or choose_mode()
    limit = args.limit if args.limit is not None else choose_limit()
    if limit is not None and limit < 1:
        raise ValueError("--limit doit être au moins 1.")
    modes = ("zero-shot", "few-shot") if mode_choice == "both" else (mode_choice,)
    few_shot_k = args.few_shot_k
    if "few-shot" in modes and few_shot_k is None:
        few_shot_k = choose_few_shot_k()
    few_shot_k = few_shot_k or 0
    if "few-shot" in modes and few_shot_k < 1:
        raise ValueError("--few-shot-k doit être au moins 1 en mode few-shot.")
    for mode in modes:
        submit(args, model, mode, few_shot_k, limit)


if __name__ == "__main__":
    main()
