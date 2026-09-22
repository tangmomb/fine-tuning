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
TEST_INPUTS = ROOT / "data" / "07_evaluation_dataset" / "03_production" / "test_inputs.jsonl"


def parse_args(default_device: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", type=Path, help="Checkpoints à évaluer ; sans cette option, un choix interactif est proposé.")
    parser.add_argument("--mode", choices=("zero-shot", "few-shot", "both"), help="Mode à lancer ; sans cette option, un choix interactif est proposé.")
    parser.add_argument("--phase", choices=("generate", "evaluate", "both"),
                        help="Phase à lancer ; sans cette option, un choix interactif est proposé.")
    parser.add_argument("--predictions", type=Path,
                        help="Prédictions JSONL à scorer ; requis avec --phase evaluate.")
    parser.add_argument("--few-shot-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--max-new-tokens", type=int,
                        help="Plafond de sortie ; défaut : 1024 sans thinking, 2048 avec thinking.")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                        help="Active le reasoning du modèle ; sans cette option, le choix est demandé.")
    parser.add_argument("--batch-size", type=int, help="Nombre de prompts générés simultanément.")
    parser.add_argument("--group-batches-by-length", action=argparse.BooleanOptionalAction, default=None,
                        help="Regroupe les prompts de longueur proche pour réduire le padding.")
    parser.add_argument("--save-prompts", action=argparse.BooleanOptionalAction, default=None,
                        help="Conserve les messages et le prompt final rendu dans predictions.jsonl.")
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


def choose_phase() -> str:
    print("\nQue voulez-vous lancer ?")
    print("  1) Génération uniquement — à exécuter sur la machine GPU")
    print("  2) Évaluation uniquement — depuis un fichier predictions.jsonl, sans modèle")
    print("  3) Génération puis évaluation — exécution complète")
    choice = input("Phase [3] : ").strip() or "3"
    phases = {"1": "generate", "2": "evaluate", "3": "both"}
    if choice not in phases:
        raise ValueError("Choisissez 1, 2 ou 3.")
    return phases[choice]


def choose_test_limit() -> int | None:
    """Demande explicitement la portée du run, sans masquer un smoke test."""
    total = sum(1 for line in TEST_INPUTS.read_text(encoding="utf-8").splitlines() if line.strip())
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


def choose_batch_size(default: int) -> int:
    value = input(f"Taille de lot / batch size [{default}] : ").strip() or str(default)
    try:
        batch_size = int(value)
    except ValueError as error:
        raise ValueError("La taille de lot doit être un entier positif.") from error
    if batch_size < 1:
        raise ValueError("La taille de lot doit être au moins 1.")
    return batch_size


def choose_group_batches_by_length() -> bool:
    print("\nRegrouper les prompts de taille proche dans les mêmes batches ?")
    print("  1) Non — conserve l'ordre du dataset (défaut)")
    print("  2) Oui — réduit le padding, sans modifier les prédictions finales")
    choice = input("Regroupement par taille [1] : ").strip() or "1"
    if choice == "1":
        return False
    if choice == "2":
        return True
    raise ValueError("Choisissez 1 ou 2.")


def choose_thinking() -> bool:
    print("\nAutoriser le thinking / reasoning du modèle ?")
    print("  1) Non — baseline directe question → SQL (défaut)")
    print("  2) Oui — plus lent, budget de tokens potentiellement plus élevé")
    choice = input("Thinking [1] : ").strip() or "1"
    if choice == "1":
        return False
    if choice == "2":
        return True
    raise ValueError("Choisissez 1 ou 2.")


def choose_save_prompts() -> bool:
    print("\nConserver les prompts envoyés au modèle dans predictions.jsonl ?")
    print("  1) Non — sorties et métriques seulement (défaut)")
    print("  2) Oui — messages et prompt final, utile pour l'audit")
    choice = input("Conserver les prompts [1] : ").strip() or "1"
    if choice == "1":
        return False
    if choice == "2":
        return True
    raise ValueError("Choisissez 1 ou 2.")


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
    phase = args.phase or choose_phase()
    # En évaluation seule, le fichier predictions.jsonl détermine déjà la
    # portée : ses IDs indiquent exactement quelles références scorer. Ne pas
    # demander une seconde fois une taille de jeu qui risquerait de contredire
    # le run choisi. --limit reste disponible pour un diagnostic explicite.
    if phase == "evaluate":
        limit = args.limit
    else:
        limit = args.limit if args.limit is not None else choose_test_limit()
    if limit is not None and limit < 1:
        raise ValueError("--limit doit être au moins 1.")
    execution_device = describe_execution_device(args.device)
    print(f"\nPériphérique détecté : {execution_device}")
    print(f"Le lancement va s'exécuter sur : {execution_device}")
    if phase == "evaluate":
        predictions = args.predictions
        if predictions is None:
            value = input("Chemin vers predictions.jsonl : ").strip()
            if not value:
                raise ValueError("Un fichier predictions.jsonl est requis pour l'évaluation seule.")
            predictions = Path(value)
        predictions = predictions.resolve()
        if not predictions.is_file():
            raise FileNotFoundError(f"Prédictions introuvables : {predictions}")
        output_dir = predictions.parent / "sql_execution"
        command = [sys.executable, str(RUNNER), "--phase", "evaluate", "--predictions", str(predictions),
                   "--output-dir", str(output_dir)]
        if limit is not None:
            command += ["--limit", str(limit)]
        print("> " + subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        return
    available = [ROOT / "models" / name for name in default_model_names if (ROOT / "models" / name).is_dir()]
    models = args.models or choose_models(available)
    modes = ("zero-shot", "few-shot") if args.mode == "both" else (args.mode,) if args.mode else choose_modes()
    batch_size = args.batch_size if args.batch_size is not None else choose_batch_size(default_batch_size)
    group_batches_by_length = (args.group_batches_by_length if args.group_batches_by_length is not None
                               else choose_group_batches_by_length())
    thinking = args.thinking if args.thinking is not None else choose_thinking()
    save_prompts = args.save_prompts if args.save_prompts is not None else choose_save_prompts()
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (2048 if thinking else 1024)
    if batch_size < 1:
        raise ValueError("--batch-size doit être au moins 1.")
    if max_new_tokens < 1:
        raise ValueError("--max-new-tokens doit être au moins 1.")
    # UTC : lisible, sans ambiguïté et disponible sur toutes les plateformes.
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")
    common = ["--phase", phase, "--few-shot-k", str(args.few_shot_k), "--seed", str(args.seed), "--device", args.device,
              "--max-new-tokens", str(max_new_tokens), "--batch-size", str(batch_size),
              "--temperature", "0", "--environment", environment]
    common += ["--thinking" if thinking else "--no-thinking"]
    common += ["--group-batches-by-length" if group_batches_by_length else "--no-group-batches-by-length"]
    if save_prompts:
        common += ["--save-prompts"]
    if limit is not None:
        common += ["--limit", str(limit)]
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
