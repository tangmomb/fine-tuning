"""Affiche les instances GPU Scaleway disponibles dans une ou plusieurs zones.

Configurez SCW_SECRET_KEY avec une clé API Scaleway ayant le droit de lire les
types d'instances, puis lancez par exemple :

    python utils/check_scaleway_gpu_availability.py  # choisit le GPU dans un menu
    python utils/check_scaleway_gpu_availability.py --gpu H100 --zone fr-par-2
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_URL = "https://api.scaleway.com/instance/v2alpha1/zones/{zone}/server-types"
ALL_ZONES = (
    "fr-par-1", "fr-par-2", "fr-par-3",
    "nl-ams-1", "nl-ams-2", "nl-ams-3",
    "pl-waw-1", "pl-waw-2", "pl-waw-3", "it-mil-1",
)
DEFAULT_GPUS = ("L4", "L40S", "H100")
DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zone",
        action="append",
        choices=ALL_ZONES,
        help="Zone Scaleway à interroger. Répétez l'option pour plusieurs zones. Défaut : toutes les zones.",
    )
    parser.add_argument(
        "--gpu",
        action="append",
        choices=DEFAULT_GPUS,
        help="GPU à rechercher. Sans cette option, le script demande un choix interactif.",
    )
    parser.add_argument("--available-only", action="store_true", help="Masque les offres scarce et shortage.")
    return parser.parse_args()


def choose_gpu() -> tuple[str, ...]:
    """Demande le GPU cible pour une exécution interactive."""
    print("GPU à vérifier :\n  1) L4\n  2) L40S\n  3) H100")
    choice = input("Choix [1-3] : ").strip()
    choices = {"1": ("L4",), "2": ("L40S",), "3": ("H100",)}
    if choice not in choices:
        raise ValueError("Choisissez 1 (L4), 2 (L40S) ou 3 (H100).")
    return choices[choice]


def load_secret_key() -> str | None:
    """Lit SCW_SECRET_KEY dans l'environnement ou dans le .env du projet."""
    if secret_key := os.environ.get("SCW_SECRET_KEY"):
        return secret_key
    if not DOTENV_PATH.is_file():
        return None
    for line in DOTENV_PATH.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().partition("=")
        if key == "SCW_SECRET_KEY" and separator:
            return value.strip().strip('"').strip("'") or None
    return None


def request_server_types(zone: str, secret_key: str) -> list[dict[str, Any]]:
    """Récupère toutes les pages des types d'instances d'une zone."""
    server_types: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        parameters: dict[str, str | int] = {"page_size": 100}
        if page_token:
            parameters["page_token"] = page_token
        request = Request(
            f"{API_URL.format(zone=zone)}?{urlencode(parameters)}",
            headers={"X-Auth-Token": secret_key, "Accept": "application/json"},
        )
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
        server_types.extend(payload.get("server_types", []))
        page_token = payload.get("next_page_token")
        if not page_token:
            return server_types


def gibibytes(value: Any) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return "—"
    return f"{value / 1024**3:.0f} GiB"


def gpu_types(server_types: list[dict[str, Any]], wanted_gpus: tuple[str, ...], available_only: bool) -> list[dict[str, Any]]:
    result = []
    for item in server_types:
        gpu = item.get("gpu_info") or {}
        gpu_name = str(gpu.get("name", "")).upper()
        if item.get("gpu_count", 0) > 0 and any(re.search(rf"\b{re.escape(name)}\b", gpu_name) for name in wanted_gpus):
            result.append(item)
    if available_only:
        result = [item for item in result if item.get("availability") == "available"]
    return sorted(result, key=lambda item: (item.get("availability") != "available", item.get("name", "")))


def display(zone: str, server_types: list[dict[str, Any]], wanted_gpus: tuple[str, ...], available_only: bool) -> None:
    rows = gpu_types(server_types, wanted_gpus, available_only)
    print(f"\n{zone}")
    if not rows:
        print("  Aucune instance GPU correspondante disponible." if available_only else "  Aucun type d'instance GPU correspondant retourné.")
        return
    print(f"  {'Type':<18} {'Disponibilité':<13} {'GPU':<24} {'VRAM':>9} {'vCPU':>5} {'RAM':>9}")
    for item in rows:
        gpu = item.get("gpu_info") or {}
        gpu_name = gpu.get("name") or "non précisé"
        gpu_description = f"{item.get('gpu_count', 0)} × {gpu_name}"
        print(
            f"  {item.get('name', '—'):<18} {item.get('availability', '—'):<13} "
            f"{gpu_description:<24} {gibibytes(gpu.get('memory')):>9} "
            f"{item.get('vcpu_count', '—'):>5} {gibibytes(item.get('memory')):>9}"
        )


def main() -> None:
    args = parse_args()
    secret_key = load_secret_key()
    if not secret_key:
        raise SystemExit(f"Ajoutez SCW_SECRET_KEY à {DOTENV_PATH}.")
    wanted_gpus = tuple(name.upper() for name in args.gpu) if args.gpu else choose_gpu()
    for zone in args.zone or ALL_ZONES:
        try:
            display(zone, request_server_types(zone, secret_key), wanted_gpus, args.available_only)
        except HTTPError as error:
            raise SystemExit(f"{zone} : API Scaleway indisponible ({error.code} {error.reason}).") from error
        except URLError as error:
            raise SystemExit(f"{zone} : impossible de joindre l'API Scaleway ({error.reason}).") from error


if __name__ == "__main__":
    main()
