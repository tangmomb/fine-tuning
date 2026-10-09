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
BASE_TRAIN = ROOT / "01_data" / "01_06_training_dataset" / "01_06_03_production" / "train.jsonl"
SYNTHETIC_DIR = ROOT / "03_fine_tuning" / "03_03_more_data" / "03_03_04_export"
COMBINED_DIR = Path(__file__).resolve().parent / "01_datasets"
ARTIFACTS_ROOT = ROOT / "03_fine_tuning" / "artifacts"
MAIN_TRAINER = ROOT / "03_fine_tuning" / "03_01_train_main" / "train.py"
EXPERIMENTS = ("A", "B", "C", "D")
FIXED_MODEL = "Qwen3.5-2B"
FIXED_EPOCHS = "3"
FIXED_LORA_PRESET = "A"
FIXED_LEARNING_RATE = "1e-4"
FIXED_GPUS = 1
FIXED_PER_DEVICE_BATCH_SIZE = "8"
FIXED_GRADIENT_ACCUMULATION_STEPS = "8"


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


def campaign_artifacts_dir() -> Path:
    """Réunit les expériences A à D lancées le même jour dans une campagne."""
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    return ARTIFACTS_ROOT / f"data-scaling-new-data-{date_tag}"


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
            f"Exécutez d'abord 03_03_more_data/03_03_04_export/build_sft.py --experiment {experiment}."
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
    parser.add_argument("--experiment", choices=EXPERIMENTS,
                        help="Expérience du chapitre 06 : A, B, C ou D. Demandée au lancement si absente.")
    parser.add_argument("--rebuild-dataset", action="store_true",
                        help="Reconstruit train_plus_<exp>.jsonl et son manifeste.")
    parser.add_argument("--gpus", type=int, choices=(1,),
                        help="Compatibilité CLI : ce protocole utilise une seule H100.")
    return parser.parse_known_args()


def has_option(arguments: list[str], option: str) -> bool:
    return any(argument == option or argument.startswith(f"{option}=") for argument in arguments)


def prompt_choice(label: str, choices: tuple[str, ...], default: str) -> str:
    """Demande un choix simple avant de démarrer torchrun."""
    displayed = "/".join(choices)
    while True:
        choice = input(f"{label} [{default}] ({displayed}) : ").strip() or default
        matching_choice = next((item for item in choices if item.lower() == choice.lower()), None)
        if matching_choice is not None:
            return matching_choice
        print(f"Choix invalide : entrez l'une des valeurs suivantes : {displayed}.")


def prompt_training_options(args: argparse.Namespace, trainer_args: list[str]) -> None:
    """Demande le choix d'expérience puis applique la configuration retenue."""
    if args.experiment is None:
        args.experiment = prompt_choice("Expérience", EXPERIMENTS, "A")
    args.gpus = FIXED_GPUS

    # Ce protocole compare les expériences A à D avec une configuration strictement identique.
    if has_option(trainer_args, "--models"):
        model_index = next(index for index, value in enumerate(trainer_args) if value == "--models")
        supplied_models = []
        for value in trainer_args[model_index + 1:]:
            if value.startswith("--"):
                break
            supplied_models.append(value)
        if supplied_models != [FIXED_MODEL]:
            raise ValueError(f"Ce lanceur entraîne uniquement {FIXED_MODEL}. Retirez --models ou utilisez --models {FIXED_MODEL}.")
    else:
        trainer_args.extend(("--models", FIXED_MODEL))

    if has_option(trainer_args, "--evaluate-run"):
        return
    fixed_options = (
        ("--epochs", FIXED_EPOCHS),
        ("--lora-preset", FIXED_LORA_PRESET),
        ("--learning-rate", FIXED_LEARNING_RATE),
        ("--per-device-batch-size", FIXED_PER_DEVICE_BATCH_SIZE),
        ("--gradient-accumulation-steps", FIXED_GRADIENT_ACCUMULATION_STEPS),
    )
    for option, value in fixed_options:
        if has_option(trainer_args, option):
            raise ValueError(f"{option} est fixé à {value} pour ce protocole et ne peut pas être modifié.")
        trainer_args.extend((option, value))
    print(
        f"Configuration fixe : {FIXED_MODEL} · LoRA r=8, alpha=16 · "
        f"learning rate {FIXED_LEARNING_RATE} · {FIXED_EPOCHS} époques · "
        f"1 H100 · batch local {FIXED_PER_DEVICE_BATCH_SIZE} · "
        f"accumulation {FIXED_GRADIENT_ACCUMULATION_STEPS} · batch effectif 64"
    )


def main() -> None:
    args, trainer_args = parse_args()
    prompt_training_options(args, trainer_args)
    dataset = build_combined_dataset(args.experiment, args.rebuild_dataset)
    campaign_dir = campaign_artifacts_dir()
    campaign_dir.mkdir(parents=True, exist_ok=True)
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
        "--output-root", str(campaign_dir / args.experiment),
        *trainer_args,
    ]
    print("Lancement sur une H100 avec le dataset enrichi…")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
