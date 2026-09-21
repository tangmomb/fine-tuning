"""Point d'entrée interactif pour la baseline locale ou Scaleway."""

import argparse
import sys

from assets.run_benchmark import main


def choose_environment() -> tuple[tuple[str, ...], str, str, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--environment", choices=("local", "scaleway"))
    args, remaining = parser.parse_known_args()
    # Les autres options (--models, --mode, etc.) sont traitées par assets.
    sys.argv = [sys.argv[0], *remaining]
    choice = {"local": "1", "scaleway": "2"}.get(args.environment)
    if choice is None:
        print("Environnement d'exécution :\n  1) Machine locale\n  2) Instance GPU Scaleway")
        choice = input("Environnement [1] : ").strip() or "1"
    if choice == "1":
        return ("Qwen3.5-0.8B", "Qwen3.5-2B"), "auto", "local", 2
    if choice == "2":
        return ("Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B"), "cuda", "scaleway", 1
    raise ValueError("Choisissez 1 (local) ou 2 (Scaleway).")


if __name__ == "__main__":
    models, device, environment, default_batch_size = choose_environment()
    main(
        default_model_names=models,
        default_device=device,
        environment=environment,
        default_batch_size=default_batch_size,
    )
