"""Télécharge sans écrasement les nouveaux artefacts de fine-tuning d'une H100.

Seuls les dossiers de runs absents du dossier local sont copiés. Un dossier
ayant le même nom est ignoré, afin de ne jamais fusionner ou écraser des poids,
checkpoints, prédictions ou logs déjà téléchargés. Les artefacts restent
volontairement hors de Git.
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
    runs = [line for line in result.stdout.splitlines() if line.strip()]
    invalid = [run for run in runs if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run)]
    if invalid:
        raise SystemExit(f"Nom(s) de dossier distant invalide(s) : {', '.join(invalid)}")
    return runs


def new_runs(runs: list[str]) -> tuple[list[str], list[str]]:
    """Sépare les nouveaux dossiers de ceux déjà présents localement."""
    pending = [run for run in runs if not (DESTINATION / run).exists()]
    skipped = [run for run in runs if (DESTINATION / run).exists()]
    return pending, skipped


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
    pending, skipped = new_runs(runs)
    if skipped:
        print("Runs déjà présents, ignorés sans modification :")
        for run in skipped:
            print(f"  - {run}")
    if not pending:
        print("Aucun nouvel artefact à télécharger.")
        return

    print(f"Téléchargement sans écrasement dans {DESTINATION}…")
    for run in pending:
        # Vérification répétée juste avant la copie, pour éviter un conflit créé
        # entre l'inventaire initial et la commande scp.
        if (DESTINATION / run).exists():
            print(f"  - {run} ignoré : présent localement depuis l'inventaire.")
            continue
        command = [
            "scp", *options, "-r",
            f"{target}:{REMOTE_ARTIFACTS}/{run}",
            str(DESTINATION),
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(f"Téléchargement de {run} échoué (code {error.returncode}).") from error
        print(f"  - {run} téléchargé")
    print(f"Artefacts disponibles dans : {DESTINATION}")


if __name__ == "__main__":
    main()
