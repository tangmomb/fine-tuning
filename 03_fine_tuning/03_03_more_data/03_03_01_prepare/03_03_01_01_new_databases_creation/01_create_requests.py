"""Prépare et, sur demande explicite, soumet le Batch de création de bases SQLite."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEFAULT_MODEL = "gpt-5.6-luna"
MAX_OUTPUT_TOKENS = 48_000

DOMAINS = (
    "librairie indépendante", "location de vélos", "clinique vétérinaire", "cinéma municipal",
    "coopérative agricole", "réseau de bus", "atelier de réparation", "centre de formation",
    "agence immobilière", "festival de musique", "gestion de musée", "service de restauration",
    "club sportif", "laboratoire d'analyses", "port de plaisance", "magasin de bricolage",
    "association caritative", "plateforme de cours", "compagnie d'assurance", "entreprise de déménagement",
    "hôtel et réservations", "service de livraison", "pépinière horticole", "garage automobile",
    "bibliothèque de quartier", "cabinet d'architecture", "parc naturel", "caisse de crédit",
    "organisation de conférences", "marché alimentaire", "équipe e-sport", "studio de photographie",
    "compagnie ferroviaire", "pharmacie", "centre de recyclage", "école de danse",
    "pisciculture", "salle de spectacle", "service de paie", "gestion de copropriété",
    "tour-opérateur", "refuge animalier", "brasserie artisanale", "atelier de couture",
    "société de sécurité", "centre de congrès", "application de covoiturage", "site archéologique",
    "chaîne de cafés", "service funéraire",
)

DATABASE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "ddl": {"type": "string", "description": "Instructions CREATE TABLE et CREATE INDEX SQLite uniquement."},
        "data": {"type": "string", "description": "Instructions INSERT INTO SQLite uniquement."},
        "description": {"type": "string"},
    },
    "required": ["ddl", "data", "description"],
}

def api_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.partition("=")[2].strip().strip('"')
    raise RuntimeError("OPENAI_API_KEY est absent.")

def upload(path: Path, token: str) -> dict:
    boundary = f"----codex{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(path.name)[0] or "application/jsonl"
    body = bytearray()
    body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode())
    body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\nContent-Type: {mime}\r\n\r\n".encode())
    body.extend(path.read_bytes()); body.extend(f"\r\n--{boundary}--\r\n".encode())
    req = request.Request("https://api.openai.com/v1/files", data=bytes(body), method="POST", headers={"Authorization": f"Bearer {token}", "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with request.urlopen(req) as response:
        return json.load(response)

def post(url: str, payload: dict, token: str) -> dict:
    req = request.Request(url, data=json.dumps(payload).encode(), method="POST", headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with request.urlopen(req) as response:
        return json.load(response)

def prompt(db_id: str, domain: str) -> str:
    return f"""Conçois une base SQLite fictive et autonome pour le domaine « {domain} ».
Identifiant imposé : {db_id}. Elle ne doit pas reproduire un schéma Spider ni employer ses noms de tables emblématiques.

Produis deux chaînes SQL : `ddl` et `data`. Le DDL doit contenir uniquement CREATE TABLE et éventuellement CREATE INDEX ; les données uniquement INSERT INTO. Utilise 4 à 8 tables, des identifiants entiers, au moins trois relations par clés étrangères et des noms ASCII en snake_case. Ajoute 15 à 40 lignes plausibles par table, y compris les tables de relation. Les références doivent être valides et les valeurs doivent permettre filtres, jointures, agrégations, tris et sous-requêtes intéressants. N'utilise ni PRAGMA, ni ATTACH, ni trigger, ni view, ni DROP, ni transaction explicite, ni commentaires SQL. Les chaînes SQL doivent être directement exécutables par SQLite. La description résume le domaine en français, en une phrase."""

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="Nombre de bases à demander (100 par défaut).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--retry-rejected", action="store_true", help="Prépare seulement les bases rejetées par la dernière matérialisation.")
    parser.add_argument("--submit", action="store_true", help="Envoie le Batch à OpenAI (action externe facturée).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count doit être positif.")
    requests: list[tuple[int, str]]
    if args.retry_rejected:
        report = json.loads((HERE / "04_final" / "materialized_report.json").read_text(encoding="utf-8"))
        requests = []
        for row in report.get("rejections", []):
            match = re.fullmatch(r"new-db:(\d{3}):(synthetic_\d{3})", str(row.get("id", "")))
            if match:
                requests.append((int(match.group(1)), match.group(2)))
        if not requests:
            raise ValueError("Aucune base rejetée réutilisable dans materialized_report.json.")
        output = HERE / "01_requests" / "new_databases_complement_requests.jsonl"
    else:
        requests = [(number, f"synthetic_{number:03d}") for number in range(1, args.count + 1)]
        output = HERE / "01_requests" / "new_databases_requests.jsonl"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} existe déjà ; utilisez --overwrite pour le remplacer.")
    lines = []
    for number, db_id in requests:
        domain = DOMAINS[(number - 1) % len(DOMAINS)]
        body = {"model": args.model, "input": [{"role": "system", "content": "Tu génères des bases SQLite fictives cohérentes."}, {"role": "user", "content": prompt(db_id, domain)}], "max_output_tokens": MAX_OUTPUT_TOKENS, "text": {"format": {"type": "json_schema", "name": "sqlite_database", "strict": True, "schema": DATABASE_SCHEMA}}}
        lines.append(json.dumps({"custom_id": f"new-db:{number:03d}:{db_id}", "method": "POST", "url": "/v1/responses", "body": body}, ensure_ascii=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(requests)} demandes écrites dans {output}")
    if not args.submit:
        return
    token = api_key(); uploaded = upload(output, token)
    mode = "complement" if args.retry_rejected else "initial"
    batch = post("https://api.openai.com/v1/batches", {"input_file_id": uploaded["id"], "endpoint": "/v1/responses", "completion_window": "24h", "metadata": {"pipeline": "new-databases-creation", "mode": mode, "model": args.model, "count": str(len(requests))}}, token)
    state = {"mode": mode, "model": args.model, "requested_databases": len(requests), "request_file": output.name, "input_file_id": uploaded["id"], "batch_id": batch["id"], "status": batch.get("status"), "submitted_at": datetime.now(timezone.utc).isoformat()}
    state_name = "complement_batch_state.json" if args.retry_rejected else "batch_state.json"
    state_path = HERE / "02_batch" / state_name; state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Batch soumis : {batch['id']}")

if __name__ == "__main__":
    main()
