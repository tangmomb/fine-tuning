"""Exécute le protocole de baseline identique pour tous les checkpoints locaux."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "evaluation_brut_models" / "assets" / "run_evaluation.py"
OUTPUT_ROOT = ROOT / "evaluation_brut_models" / "runs"


def parse_args(default_device: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", type=Path, help="Checkpoints à évaluer ; sans cette option, un choix interactif est proposé.")
    parser.add_argument("--mode", choices=("zero-shot", "few-shot", "both"), help="Mode à lancer ; sans cette option, un choix interactif est proposé.")
    parser.add_argument("--few-shot-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, help="Nombre de prompts générés simultanément.")
    parser.add_argument("--limit", type=int, help="Smoke test seulement ; à omettre pour la baseline complète.")
    return parser.parse_args()


def choose_models(available: list[Path]) -> list[Path]:
    if not available:
        raise ValueError("Aucun checkpoint trouvé dans models/ ; utilisez --models.")
    print("\nModèles disponibles :")
    for index, model in enumerate(available, start=1):
        print(f"  {index}) {model.name}")
    print("  0) Tous les modèles")
    choice = input("Modèle(s) à évaluer [0] : ").strip() or "0"
    if choice == "0":
        return available
    try:
        selected_indexes = [int(value.strip()) for value in choice.split(",")]
        if not selected_indexes or any(index < 1 or index > len(available) for index in selected_indexes):
            raise ValueError
    except ValueError as error:
        raise ValueError("Choisissez 0 ou une liste d'indices séparés par des virgules, par exemple 1,3.") from error
    return [available[index - 1] for index in dict.fromkeys(selected_indexes)]


def choose_modes() -> tuple[str, ...]:
    print("\nMode d'évaluation :\n  1) zero-shot\n  2) few-shot\n  3) Les deux")
    choice = input("Mode [3] : ").strip() or "3"
    modes = {"1": ("zero-shot",), "2": ("few-shot",), "3": ("zero-shot", "few-shot")}
    if choice not in modes:
        raise ValueError("Choisissez 1, 2 ou 3.")
    return modes[choice]


def choose_batch_size(default: int) -> int:
    value = input(f"Taille de lot / batch size [{default}] : ").strip() or str(default)
    try:
        batch_size = int(value)
    except ValueError as error:
        raise ValueError("La taille de lot doit être un entier positif.") from error
    if batch_size < 1:
        raise ValueError("La taille de lot doit être au moins 1.")
    return batch_size


def describe_execution_device(requested_device: str) -> str:
    """Return a human-readable confirmation of the device used by a run."""
    if requested_device == "cpu":
        return "CPU (demandé explicitement)"
    try:
        import torch
    except ImportError:
        return f"{requested_device} (PyTorch indisponible : vérification GPU impossible)"
    if torch.cuda.is_available():
        return f"CUDA — {torch.cuda.get_device_name(0)}"
    if requested_device == "cuda":
        raise RuntimeError("--device cuda a été demandé, mais aucune GPU CUDA n'est détectée.")
    return "CPU (aucune GPU CUDA détectée)"


def main(default_model_names: tuple[str, ...], default_device: str, environment: str, default_batch_size: int) -> None:
    args = parse_args(default_device)
    execution_device = describe_execution_device(args.device)
    print(f"\nPériphérique détecté : {execution_device}")
    print(f"Les évaluations vont se lancer sur : {execution_device}")
    available = [ROOT / "models" / name for name in default_model_names if (ROOT / "models" / name).is_dir()]
    models = args.models or choose_models(available)
    modes = ("zero-shot", "few-shot") if args.mode == "both" else (args.mode,) if args.mode else choose_modes()
    batch_size = args.batch_size if args.batch_size is not None else choose_batch_size(default_batch_size)
    if batch_size < 1:
        raise ValueError("--batch-size doit être au moins 1.")
    # UTC, lisible dans les noms de dossiers et sans caractères interdits sous Windows.
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")
    common = ["--few-shot-k", str(args.few_shot_k), "--seed", str(args.seed), "--device", args.device,
              "--max-new-tokens", str(args.max_new_tokens), "--batch-size", str(batch_size),
              "--temperature", "0", "--environment", environment]
    if args.limit:
        common += ["--limit", str(args.limit)]
    for model in models:
        model = model.resolve()
        if not model.is_dir():
            raise FileNotFoundError(f"Checkpoint introuvable : {model}")
        for mode in modes:
            run_name = f"baseline-{stamp}/{model.name}-{mode}"
            command = [sys.executable, str(RUNNER), "--model", str(model), "--mode", mode, "--run-name", run_name, *common]
            print(f"\nLancement sur : {execution_device}")
            print("> " + subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, check=True, cwd=ROOT)
