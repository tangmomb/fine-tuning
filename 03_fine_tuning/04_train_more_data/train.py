"""Entraîne ``train`` enrichi par l'un des jeux synthétiques A à D.

Chaque lancement correspond à une expérience du chapitre 06 : le jeu Spider
initial est conservé intégralement et seul le dataset synthétique choisi est
ajouté. L'entraînement proprement dit est délégué à ``01_train_main/train.py``
afin de conserver exactement le même protocole LoRA, de validation et de test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE_TRAIN = ROOT / "01_data" / "06_training_dataset" / "03_production" / "train.jsonl"
SYNTHETIC_DIR = ROOT / "03_fine_tuning" / "03_more_data" / "04_export"
COMBINED_DIR = Path(__file__).resolve().parent / "01_datasets"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MAIN_TRAINER = ROOT / "03_fine_tuning" / "01_train_main" / "train.py"
EXPERIMENTS = ("A", "B", "C", "D")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise ValueError(f"JSONL invalide dans {path} : {error}") from error


def fingerprint(row: dict[str, Any]) -> str:
    """Identifie un exemple par son contenu SFT, indépendamment de ses métadonnées."""
    messages = row["messages"]
    return hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Dataset vide : {path}")
    for number, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"{path}, ligne {number} : messages system/user/assistant attendus.")
        if not all(isinstance(message.get("content"), str) and message["content"].strip() for message in messages):
            raise ValueError(f"{path}, ligne {number} : contenu de message vide ou invalide.")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_combined_dataset(experiment: str, rebuild: bool) -> Path:
    """Écrit un dataset reproductible sans jamais modifier les sources."""
    synthetic = SYNTHETIC_DIR / f"synthetic_data_{experiment}.jsonl"
    output = COMBINED_DIR / f"train_plus_{experiment}.jsonl"
    manifest = COMBINED_DIR / f"train_plus_{experiment}_manifest.json"
    if output.exists() and manifest.exists() and not rebuild:
        print(f"Dataset fusionné réutilisé : {output}")
        return output
    if not BASE_TRAIN.is_file():
        raise FileNotFoundError(f"Train Spider introuvable : {BASE_TRAIN}")
    if not synthetic.is_file():
        raise FileNotFoundError(
            f"Dataset synthétique {experiment} introuvable : {synthetic}. "
            f"Exécutez d'abord 03_more_data/04_export/build_sft.py --experiment {experiment}."
        )

    base_rows = read_jsonl(BASE_TRAIN)
    synthetic_rows = read_jsonl(synthetic)
    validate_rows(base_rows, BASE_TRAIN)
    validate_rows(synthetic_rows, synthetic)
    seen = {fingerprint(row) for row in base_rows}
    added_rows: list[dict[str, Any]] = []
    duplicates = 0
    for row in synthetic_rows:
        key = fingerprint(row)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        added_rows.append(row)
    if not added_rows:
        raise ValueError(f"Aucun exemple inédit à ajouter depuis {synthetic}.")

    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [*base_rows, *added_rows]),
        encoding="utf-8",
    )
    manifest.write_text(json.dumps({
        "experiment": experiment,
        "base_dataset": str(BASE_TRAIN),
        "base_dataset_sha256": sha256(BASE_TRAIN),
        "base_examples": len(base_rows),
        "synthetic_dataset": str(synthetic),
        "synthetic_dataset_sha256": sha256(synthetic),
        "synthetic_examples_before_deduplication": len(synthetic_rows),
        "synthetic_duplicates_already_in_train": duplicates,
        "synthetic_examples_added": len(added_rows),
        "combined_examples": len(base_rows) + len(added_rows),
        "combined_dataset_sha256": sha256(output),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset fusionné créé : {output} ({len(base_rows)} + {len(added_rows)} = {len(base_rows) + len(added_rows)} exemples)")
    return output


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Les autres options sont transmises à 01_train_main/train.py, par exemple --models Qwen3.5-2B --epochs 3 --lora-preset A --learning-rate 1e-4.",
    )
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS,
                        help="Expérience du chapitre 06 : A, B, C ou D.")
    parser.add_argument("--rebuild-dataset", action="store_true",
                        help="Reconstruit train_plus_<exp>.jsonl et son manifeste.")
    parser.add_argument("--gpus", type=int, choices=(1, 2), default=2,
                        help="Nombre de H100 utilisées (2 par défaut ; 1 pour un lancement mono-GPU).")
    return parser.parse_known_args()


def has_option(arguments: list[str], option: str) -> bool:
    return any(argument == option or argument.startswith(f"{option}=") for argument in arguments)


def main() -> None:
    args, trainer_args = parse_args()
    dataset = build_combined_dataset(args.experiment, args.rebuild_dataset)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.gpus == 2 and not has_option(trainer_args, "--evaluate-run"):
        interactive_options = ("--models", "--epochs", "--lora-preset", "--learning-rate")
        missing = [option for option in interactive_options if not has_option(trainer_args, option)]
        if missing:
            raise ValueError(
                "Un lancement sur deux GPU doit être non interactif pour garder les deux rangs synchronisés. "
                f"Ajoutez : {' '.join(missing)}"
            )
    # 2 GPU × batch local 4 × accumulation 8 = batch effectif 64, identique
    # au protocole mono-GPU historique (1 × 2 × 32), mais avec deux fois moins
    # de micro-batches. Les valeurs explicitement fournies sont respectées.
    if args.gpus == 2 and not has_option(trainer_args, "--per-device-batch-size"):
        trainer_args.extend(("--per-device-batch-size", "4"))
    if args.gpus == 2 and not has_option(trainer_args, "--gradient-accumulation-steps"):
        trainer_args.extend(("--gradient-accumulation-steps", "8"))
    if not has_option(trainer_args, "--run-name"):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
        trainer_args.extend(("--run-name", f"more-data-{args.experiment.lower()}-{timestamp}"))
    launch_gpus = 1 if has_option(trainer_args, "--evaluate-run") else args.gpus
    if launch_gpus != args.gpus:
        print("--evaluate-run : évaluation relancée sur une seule H100 (l'inférence de récupération n'est pas distribuée).")
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node", str(launch_gpus), str(MAIN_TRAINER),
        "--dataset", str(dataset),
        "--output-root", str(ARTIFACTS_DIR / args.experiment),
        *trainer_args,
    ]
    print(f"Lancement distribué sur {args.gpus} H100 avec le dataset enrichi…")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
