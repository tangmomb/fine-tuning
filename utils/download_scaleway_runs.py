"""Télécharge les runs d'une VM Scaleway sans vider le dossier local.

Le script demande l'adresse IP publique de la VM à chaque exécution. Les fichiers
distants dont le chemin existe déjà localement sont remplacés ; les autres sont
ajoutés au dossier ``evaluation_brut_models/runs_scaleway``.
"""

from __future__ import annotations

import ipaddress
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


def main() -> None:
    if not SSH_KEY.is_file():
        raise SystemExit(f"Clé SSH introuvable : {SSH_KEY}")

    ip_address = ask_ip_address()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    print(f"Synchronisation des runs dans {DESTINATION}…")

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
