"""Résume les mesures GPU d'un run d'entraînement distribué."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Dossier du run, contenant logs/gpu_rank-*_telemetry.jsonl.")
    args = parser.parse_args()
    files = sorted((args.run_dir / "logs").glob("gpu_rank-*_telemetry.jsonl"))
    if not files:
        raise FileNotFoundError(f"Télémétrie introuvable dans {args.run_dir / 'logs'}.")
    for path in files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        logs = [row for row in rows if row["event"] == "log" and row.get("steps_per_second")]
        begin = next((row for row in rows if row["event"] == "train_begin"), {})
        end = next((row for row in reversed(rows) if row["event"] == "train_end"), {})
        if not logs:
            print(f"{path.name}: aucune fenêtre de mesure complète.")
            continue
        throughputs = [row["steps_per_second"] for row in logs]
        peak = max(row["max_memory_allocated_gib"] for row in logs)
        print(
            f"rang {logs[0]['rank']} · {begin.get('gpu_name', 'GPU')}\n"
            f"  débit médian : {statistics.median(throughputs):.3f} steps/s "
            f"(max {max(throughputs):.3f})\n"
            f"  VRAM de pic allouée : {max(peak, end.get('max_memory_allocated_gib', 0)):.2f} GiB\n"
            f"  durée train : {end.get('train_seconds', 'inconnue')} s"
        )


if __name__ == "__main__":
    main()
