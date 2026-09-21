"""Remplace le contenu local de runs_scaleway par les runs d'une VM Scaleway.

Le script demande l'adresse IP publique de la VM à chaque exécution. Il conserve
le dossier ``evaluation_brut_models/runs_scaleway`` mais supprime son contenu
avant de télécharger les résultats distants.
"""

from __future__ import annotations

import ipaddress
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "evaluation_brut_models" / "runs_scaleway"
REMOTE_RUNS = "/root/fine-tuning/evaluation_brut_models/runs/."
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"


def ask_ip_address() -> str:
    """Return a validated IPv4 address entered by the user."""
    while True:
        value = input("Adresse IP publique de la VM Scaleway : ").strip()
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            print("Adresse IP invalide. Exemple : 151.115.87.239")
            continue
        if address.version != 4:
            print("Utilisez l'adresse IPv4 publique de la VM.")
            continue
        return str(address)


def empty_directory(directory: Path) -> None:
    """Delete all children while preserving the directory itself."""
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> None:
    if not SSH_KEY.is_file():
        raise SystemExit(f"Clé SSH introuvable : {SSH_KEY}")

    ip_address = ask_ip_address()
    print(f"Remplacement du contenu de {DESTINATION}…")
    empty_directory(DESTINATION)

    command = [
        "scp",
        "-i",
        str(SSH_KEY),
        "-r",
        f"root@{ip_address}:{REMOTE_RUNS}",
        str(DESTINATION),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Téléchargement échoué (code {error.returncode}).") from error

    print(f"Runs téléchargés dans : {DESTINATION}")


if __name__ == "__main__":
    main()
