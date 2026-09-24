"""Télécharge les artefacts de fine-tuning produits sur une H100.

Les dossiers de runs distants sous ``03_fine_tuning/artifacts/`` sont copiés
dans le même dossier du dépôt local. Les artefacts restent volontairement hors
de Git : ils incluent des poids LoRA, checkpoints, prédictions et logs.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "03_fine_tuning" / "artifacts"
REMOTE_ARTIFACTS = "/root/fine-tuning/03_fine_tuning/artifacts"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"


def ask_ssh_target() -> str:
    """Demande une cible SSH simple, avec ``root`` comme utilisateur par défaut."""
    while True:
        value = input("Adresse SSH de la H100 (ex. root@151.115.87.239) : ").strip()
        if not value:
            print("Adresse SSH requise.")
            continue
        target = value if "@" in value else f"root@{value}"
        if target.count("@") != 1 or any(character.isspace() for character in target):
            print("Adresse invalide. Exemple : root@151.115.87.239")
            continue
        user, host = target.split("@", maxsplit=1)
        if not user or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", host):
            print("Adresse invalide. Exemple : root@151.115.87.239")
            continue
        return target


def ssh_options() -> list[str]:
    """Utilise la clé habituelle si elle existe, sinon la configuration SSH locale."""
    return ["-i", str(SSH_KEY)] if SSH_KEY.is_file() else []


def list_remote_runs(target: str, options: list[str]) -> list[str]:
    command = [
        "ssh", *options, target,
        f"find {REMOTE_ARTIFACTS} -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' | sort",
    ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or f"code {error.returncode}"
        raise SystemExit(f"Impossible de lister les artefacts distants : {detail}") from error
    return [line for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    target = ask_ssh_target()
    options = ssh_options()
    runs = list_remote_runs(target, options)
    if not runs:
        raise SystemExit(f"Aucun sous-dossier trouvé dans {REMOTE_ARTIFACTS} sur la H100.")

    print("Runs distants trouvés :")
    for run in runs:
        print(f"  - {run}")

    DESTINATION.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement dans {DESTINATION}…")
    command = [
        "scp", *options, "-r",
        f"{target}:{REMOTE_ARTIFACTS}/.",
        str(DESTINATION),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Téléchargement échoué (code {error.returncode}).") from error
    print(f"Artefacts téléchargés dans : {DESTINATION}")


if __name__ == "__main__":
    main()
