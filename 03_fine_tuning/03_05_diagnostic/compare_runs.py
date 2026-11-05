"""Compare des groupes de runs (par exemple baseline vs A2) sur la validation, façon Spider.

Pré-requis : rescore_validation.py a été relancé après le téléchargement des runs.

  python compare_runs.py --baseline qwen3.5-2b-20260926-125948Z_r8 baseline-r8-seed7 --candidate A2/

Chaque motif sélectionne les runs dont le chemin contient ce texte. Pour chaque run, le
checkpoint retenu est le meilleur sur la validation (protocole du compte rendu), ou l'époque
fixée par --epoch. Le script affiche :
  - l'accuracy de chaque run et la moyenne par groupe ;
  - un test apparié (McNemar) et un IC95 bootstrap par base pour chaque paire baseline/candidat ;
  - la ventilation par type de requête gold et par difficulté Spider ;
  - le taux d'erreurs « no such column ».
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACTS = ROOT / "03_fine_tuning" / "artifacts"
RESCORE = HERE / "rescore_validation.json"
RAW_DEV = ROOT / "BRUT_spider-original" / "data" / "spider_data" / "dev.json"
CLASSIFIER = ROOT / "03_fine_tuning" / "03_02_train_data_scaling" / "build_train_with_hardness.py"
QUERY_TYPES = {
    "1 table": lambda s: not re.search(r"\bjoin\b", s),
    "JOIN": lambda s: bool(re.search(r"\bjoin\b", s)),
    "2+ JOIN": lambda s: len(re.findall(r"\bjoin\b", s)) >= 2,
    "ORDER BY / LIMIT": lambda s: bool(re.search(r"\border by\b|\blimit\b", s)),
    "GROUP BY": lambda s: bool(re.search(r"\bgroup by\b", s)),
    "sous-requête": lambda s: bool(re.search(r"\(\s*select\b", s)),
    "UNION/INTERSECT/EXCEPT": lambda s: bool(re.search(r"\b(union|intersect|except)\b", s)),
}


def hardness_by_id() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("spider_hardness_classifier", CLASSIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = json.loads(RAW_DEV.read_text(encoding="utf-8"))
    return {f"dev:{index}": module.spider_hardness(row["sql"]) for index, row in enumerate(rows)}


def pick_runs(report: dict, patterns: list[str], epoch: int | None) -> dict[str, str]:
    """Retourne {run: label du checkpoint retenu} pour les runs correspondant aux motifs."""
    by_run: dict[str, list[str]] = defaultdict(list)
    for label in report:
        run, _, checkpoint = label.rpartition("/eval/")
        if any(pattern in label for pattern in patterns):
            by_run[run].append(label)
    chosen = {}
    for run, labels in sorted(by_run.items()):
        if epoch is not None:
            labels = [label for label in labels if label.endswith(f"epoch-{epoch}")]
        if labels:
            chosen[run] = max(labels, key=lambda label: report[label]["spider"])
    return chosen


def mcnemar(gained: int, lost: int) -> float:
    n = gained + lost
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(min(gained, lost) + 1)) / 2 ** n)


def bootstrap(base: dict[str, bool], cand: dict[str, bool], db_of: dict[str, str], rng: random.Random) -> tuple[float, float]:
    by_db: dict[str, list[str]] = defaultdict(list)
    for example_id in base:
        by_db[db_of[example_id]].append(example_id)
    databases = sorted(by_db)
    diffs = []
    for _ in range(2000):
        sample = [example_id for db in rng.choices(databases, k=len(databases)) for example_id in by_db[db]]
        diffs.append(100 * sum(cand[i] - base[i] for i in sample) / len(sample))
    diffs.sort()
    return diffs[50], diffs[1949]


def predictions(label: str) -> dict[str, dict]:
    path = ARTIFACTS / label / "val_predictions.jsonl"
    return {row["id"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if row}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", nargs="+", required=True, help="Motifs des runs de référence.")
    parser.add_argument("--candidate", nargs="+", required=True, help="Motifs des runs à comparer.")
    parser.add_argument("--epoch", type=int, help="Époque fixe au lieu du meilleur checkpoint de validation.")
    args = parser.parse_args()

    report = json.loads(RESCORE.read_text(encoding="utf-8"))
    groups = {"baseline": pick_runs(report, args.baseline, args.epoch), "candidat": pick_runs(report, args.candidate, args.epoch)}
    for name, runs in groups.items():
        if not runs:
            raise SystemExit(f"Aucun run {name} trouvé : relancer rescore_validation.py après le téléchargement ?")

    dev = json.loads(RAW_DEV.read_text(encoding="utf-8"))
    db_of = {f"dev:{index}": row["db_id"] for index, row in enumerate(dev)}
    hardness = hardness_by_id()
    some_label = next(iter(groups["baseline"].values()))
    gold_sql = {example_id: row["gold_sql"].lower() for example_id, row in predictions(some_label).items()}

    print("Accuracy de validation (comparaison façon Spider) :")
    for name, runs in groups.items():
        scores = [report[label]["spider"] for label in runs.values()]
        for label in runs.values():
            print(f"  {name:9s} {label:85s} {100 * report[label]['spider']:6.2f} %")
        print(f"  {name:9s} moyenne sur {len(scores)} run(s) : {100 * sum(scores) / len(scores):.2f} %\n")

    rng = random.Random(0)
    print("Comparaisons appariées (candidat - baseline) :")
    for base_label in groups["baseline"].values():
        for cand_label in groups["candidat"].values():
            base, cand = report[base_label]["spider_correct"], report[cand_label]["spider_correct"]
            gained = sum(cand[i] and not base[i] for i in base)
            lost = sum(base[i] and not cand[i] for i in base)
            low, high = bootstrap(base, cand, db_of, rng)
            print(f"  {cand_label.split('/eval/')[0][-40:]:40s} vs {base_label.split('/eval/')[0][-40:]:40s} "
                  f"Δ {100 * (gained - lost) / len(base):+.2f} pts  +{gained}/-{lost}  p={mcnemar(gained, lost):.2f}  IC95 [{low:+.2f}, {high:+.2f}]")

    def mean_accuracy(runs: dict[str, str], ids: list[str]) -> float:
        return 100 * sum(report[label]["spider_correct"][i] for label in runs.values() for i in ids) / (len(ids) * len(runs))

    print("\nPar type de requête gold (moyenne des runs de chaque groupe) :")
    print(f"  {'':24s}{'n':>6s}{'baseline':>11s}{'candidat':>11s}{'Δ':>9s}")
    categories = [(name, [i for i, sql in gold_sql.items() if test(sql)]) for name, test in QUERY_TYPES.items()]
    categories += [(f"difficulté {level}", [i for i in gold_sql if hardness[i] == level]) for level in ("easy", "medium", "hard", "extra")]
    for name, ids in categories:
        base_acc, cand_acc = mean_accuracy(groups["baseline"], ids), mean_accuracy(groups["candidat"], ids)
        print(f"  {name:24s}{len(ids):6d}{base_acc:10.2f}%{cand_acc:10.2f}%{cand_acc - base_acc:+8.2f}")

    print("\nErreurs « no such column » (moyenne par run) :")
    for name, runs in groups.items():
        counts = [sum("no such column" in (row.get("execution_error") or "") for row in predictions(label).values()) for label in runs.values()]
        print(f"  {name:9s} {sum(counts) / len(counts):.1f} sur {len(gold_sql)}")


if __name__ == "__main__":
    main()
