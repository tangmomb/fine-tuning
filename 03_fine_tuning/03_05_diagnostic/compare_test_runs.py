"""Compare baseline et candidat sur le jeu de test, paire par paire.

À n'utiliser qu'après l'évaluation test finale (train.py --evaluate-run <run> --test-epoch N) :
ce script lit eval/epoch-N/final_test/test_predictions.jsonl et ne sert pas à choisir un modèle.

  python compare_test_runs.py --epoch 3 \
      --pair 42 qwen3.5-2b-20260926-125948Z_r8_alpha16_lr1e-4 data-scaling-new-data-20261011/A2/more-data-a2-20261011-205817Z \
      --pair 7 baseline-r8-seed7 data-scaling-new-data-20261104/A2/more-data-a2-seed7

Chaque --pair donne un nom, le run baseline et le run candidat (chemins relatifs à artifacts/).
Les runs ne sont comparés qu'à l'intérieur d'une paire. Le script affiche, en comparaison
stricte (évaluateur du projet) et façon Spider (rescore_validation.py) :
  - l'accuracy de chaque run ;
  - le test de McNemar apparié et un IC95 bootstrap par base de test ;
  - la ventilation par difficulté Spider et par type de requête gold ;
  - le nombre d'erreurs « no such column ».
Le détail chiffré est aussi écrit dans compare_test_runs.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import compare_runs
import rescore_validation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACTS = ROOT / "03_fine_tuning" / "artifacts"
RAW_TEST = ROOT / "BRUT_spider-original" / "data" / "spider_data" / "test.json"
TEST_DATABASES = ROOT / "BRUT_spider-original" / "data" / "spider_data" / "test_database"
OUTPUT = HERE / "compare_test_runs.json"
LEVELS = ("easy", "medium", "hard", "extra")


def load_run(run: str, epoch: int, cache: dict) -> dict:
    """Lit les prédictions test d'un run et calcule les deux comparaisons par exemple."""
    path = ARTIFACTS / run / "eval" / f"epoch-{epoch}" / "final_test" / "test_predictions.jsonl"
    if not path.is_file():
        raise SystemExit(f"Prédictions test introuvables : {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    strict, spider = {}, {}
    for row in rows:
        gold = rescore_validation.execute(row["db_id"], row["gold_sql"], cache)
        pred = rescore_validation.execute(row["db_id"], row["predicted_sql"], cache)
        ordered = rescore_validation.has_order_by(row["gold_sql"])
        strict[row["id"]] = bool(row["execution_match"])
        spider[row["id"]] = strict[row["id"]] or (
            pred is not None and gold is not None and rescore_validation.spider_match(pred, gold, ordered)
        )
    return {
        "run": run, "rows": {row["id"]: row for row in rows}, "strict": strict, "spider": spider,
        "no_such_column": sum("no such column" in (row.get("execution_error") or "") for row in rows),
    }


def paired(base: dict[str, bool], cand: dict[str, bool], db_of: dict[str, str]) -> dict:
    gained = sum(cand[i] and not base[i] for i in base)
    lost = sum(base[i] and not cand[i] for i in base)
    low, high = compare_runs.bootstrap(base, cand, db_of, random.Random(0))
    return {
        "baseline": 100 * sum(base.values()) / len(base), "candidate": 100 * sum(cand.values()) / len(cand),
        "delta": 100 * (gained - lost) / len(base), "gained": gained, "lost": lost,
        "mcnemar_p": compare_runs.mcnemar(gained, lost), "ci95": [low, high],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=3, action="append", required=True, metavar=("NOM", "BASELINE", "CANDIDAT"))
    parser.add_argument("--epoch", type=int, required=True, help="Époque évaluée sur le test, fixée à l'avance.")
    args = parser.parse_args()

    rescore_validation.DATABASES = TEST_DATABASES
    hardness_module = compare_runs.importlib.util.spec_from_file_location("spider_hardness_classifier", compare_runs.CLASSIFIER)
    classifier = compare_runs.importlib.util.module_from_spec(hardness_module)
    hardness_module.loader.exec_module(classifier)
    raw = json.loads(RAW_TEST.read_text(encoding="utf-8"))
    hardness = {f"test:{index}": classifier.spider_hardness(row["sql"]) for index, row in enumerate(raw)}
    raw_db = {f"test:{index}": row["db_id"] for index, row in enumerate(raw)}

    cache: dict = {}
    report: dict = {"epoch": args.epoch, "pairs": {}}
    pairs = []
    for name, base_run, cand_run in args.pair:
        base, cand = load_run(base_run, args.epoch, cache), load_run(cand_run, args.epoch, cache)
        if set(base["rows"]) != set(cand["rows"]) or set(base["rows"]) != set(hardness):
            raise SystemExit(f"Paire {name} : les IDs de test ne correspondent pas.")
        if any(row["db_id"] != raw_db[i] for i, row in base["rows"].items()):
            raise SystemExit("Les IDs test:N ne sont pas alignés sur test.json : difficulté Spider indisponible.")
        pairs.append((name, base, cand))

    ids = list(pairs[0][1]["rows"])
    db_of = {i: pairs[0][1]["rows"][i]["db_id"] for i in ids}
    gold_sql = {i: pairs[0][1]["rows"][i]["gold_sql"].lower() for i in ids}
    print(f"Jeu de test : {len(ids)} exemples, {len(set(db_of.values()))} bases, époque {args.epoch}.\n")

    for name, base, cand in pairs:
        entry = {"baseline_run": base["run"], "candidate_run": cand["run"]}
        print(f"Paire {name} : {cand['run']}\n   contre {base['run']}")
        for metric in ("strict", "spider"):
            stats = entry[metric] = paired(base[metric], cand[metric], db_of)
            print(f"  {metric:7s} baseline {stats['baseline']:.2f} %  candidat {stats['candidate']:.2f} %  "
                  f"Δ {stats['delta']:+.2f} pts  +{stats['gained']}/-{stats['lost']}  "
                  f"p={stats['mcnemar_p']:.2g}  IC95 [{stats['ci95'][0]:+.2f}, {stats['ci95'][1]:+.2f}]")
        entry["hardness_strict"] = {}
        for level in (*LEVELS, "total"):
            level_ids = [i for i in ids if level == "total" or hardness[i] == level]
            entry["hardness_strict"][level] = {
                "n": len(level_ids),
                "baseline": sum(base["strict"][i] for i in level_ids),
                "candidate": sum(cand["strict"][i] for i in level_ids),
            }
        print("  par difficulté (strict) : " + "  ".join(
            f"{level} {v['baseline']}/{v['n']} → {v['candidate']}/{v['n']}" for level, v in entry["hardness_strict"].items()))
        entry["no_such_column"] = {"baseline": base["no_such_column"], "candidate": cand["no_such_column"]}
        print(f"  no such column : {base['no_such_column']} → {cand['no_such_column']}\n")
        report["pairs"][name] = entry

    def mean_accuracy(side: int, type_ids: list[str]) -> float:
        return 100 * sum(pair[side]["spider"][i] for pair in pairs for i in type_ids) / (len(type_ids) * len(pairs))

    print(f"Par type de requête gold (façon Spider, moyenne des {len(pairs)} paires) :")
    print(f"  {'':24s}{'n':>6s}{'baseline':>11s}{'candidat':>11s}{'Δ':>9s}")
    report["query_types_spider"] = {}
    for name, test in compare_runs.QUERY_TYPES.items():
        type_ids = [i for i, sql in gold_sql.items() if test(sql)]
        base_acc, cand_acc = mean_accuracy(1, type_ids), mean_accuracy(2, type_ids)
        report["query_types_spider"][name] = {"n": len(type_ids), "baseline": base_acc, "candidate": cand_acc}
        print(f"  {name:24s}{len(type_ids):6d}{base_acc:10.2f}%{cand_acc:10.2f}%{cand_acc - base_acc:+8.2f}")

    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nDétail écrit dans {OUTPUT}")


if __name__ == "__main__":
    main()
