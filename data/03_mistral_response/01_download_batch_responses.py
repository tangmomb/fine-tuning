"""Télécharge les réponses brutes des lots Mistral terminés."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from translation_common import ROOT, api, choose_environment, download, load_state


def main() -> None:
    environment = choose_environment()
    jobs = []
    for item in load_state(environment)["batches"]:
        job = api(f"/batch/jobs/{item['batch_id']}")
        print(f"{item['split']} : {job['status']} — {job.get('succeeded_requests', 0)}/{job.get('total_requests', '?')}")
        jobs.append((item, job))
    if not all(job.get("status") == "SUCCESS" for _, job in jobs):
        print("Tous les lots ne sont pas encore terminés.")
        return
    if input("Télécharger les réponses brutes ? [o/N] ").strip().lower() not in {"o", "oui"}:
        return
    for number, (item, job) in enumerate(jobs, 1):
        output_file = job.get("output_file")
        if not isinstance(output_file, str):
            raise RuntimeError(f"{item['split']} : fichier de sortie absent.")
        path = ROOT / "data" / "03_mistral_response" / environment / f"output_{number:02d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(download(output_file), encoding="utf-8")
        print(f"{item['split']} : réponse brute écrite dans {path}")


if __name__ == "__main__":
    main()
