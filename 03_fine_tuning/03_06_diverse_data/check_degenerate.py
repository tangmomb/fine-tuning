"""Vérifie qu'un lot « hard / extra » n'est pas dégénéré comme les anciens C et D.

Les anciens C et D étaient classés hard / extra par le classifieur Spider, mais reposaient sur
un seul gabarit par niveau (sous-requête tautologique sur la même table, GROUP BY sans jointure).
Ce contrôle compare les paires valides d'un lot (validated.jsonl, produit par check_pilot.py)
aux exemples hard / extra du train Spider, à l'ancien D et à A2, à taille égale :

  python check_degenerate.py --input D_v2

Il affiche la diversité des squelettes SQL, la concentration sur les squelettes les plus
fréquents, les opérations réellement présentes, les sous-requêtes sur la même table sans filtre
et les jointures dont la table n'est jamais utilisée. `--drop-superfluous-joins` écarte ces
dernières de validated.jsonl avant le jugement. Le verdict final échoue si le lot s'écarte
trop des exemples hard / extra de Spider.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINE_TUNING = HERE.parent
sys.path.insert(0, str(HERE))
from create_requests import SPIDER_TRAIN, read_jsonl, skeleton  # noqa: E402

EXPORT = FINE_TUNING / "03_03_more_data" / "03_03_04_export"
FEATURES = {"JOIN": r"\bjoin\b", "2+ JOIN": r"\bjoin\b.*\bjoin\b", "3+ JOIN": r"\bjoin\b.*\bjoin\b.*\bjoin\b",
            "ORDER BY": r"\border by\b", "LIMIT": r"\blimit\b", "GROUP BY": r"\bgroup by\b", "HAVING": r"\bhaving\b",
            "sous-requête": r"\(\s*select\b", "UNION/INTERSECT/EXCEPT": r"\b(union|intersect|except)\b"}
# Seuils du verdict, relatifs aux exemples hard / extra de Spider tirés à taille égale.
MIN_SKELETON_RATIO = 0.6
MAX_TOP10_SHARE = 35.0
MIN_FEATURE_RATIO = 0.5
MAX_SAME_TABLE_SUBQUERY = 3.0


def assistant_sql(row: dict) -> str:
    messages = row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"])
    return messages[2]["content"]


def same_table_unfiltered_subquery(sql: str) -> bool:
    """Gabarit de l'ancien C/D : col IN (SELECT col FROM même_table) sans aucun filtre dans la sous-requête."""
    outer = re.search(r"\bfrom\s+(\w+)", sql, re.I)
    for inner in re.finditer(r"\bin\s*\(\s*select\s+[\w.]+\s+from\s+(\w+)\s*\)", sql, re.I):
        if outer and inner.group(1).lower() == outer.group(1).lower():
            return True
    return False


def unused_join(sql: str) -> bool:
    """Une table jointe dont l'alias n'apparaît que dans sa condition ON."""
    aliases = re.findall(r"\bjoin\s+\w+\s+as\s+(t\d)\b", sql, re.I)
    return any(len(re.findall(rf"\b{alias}\.", sql, re.I)) <= 1 for alias in aliases)


def superfluous_join(sql: str) -> bool:
    """Jointure à table inutilisée, hors comptage par GROUP BY : la jointure n'apporte ni colonne ni agrégat."""
    return unused_join(sql) and not re.search(r"\bgroup by\b", sql, re.I)


def describe(sqls: list[str]) -> dict[str, float]:
    n = len(sqls)
    skeletons = Counter(skeleton(sql) for sql in sqls)
    stats = {"squelettes / 100 ex.": 100 * len(skeletons) / n,
             "part du 1er squelette": 100 * skeletons.most_common(1)[0][1] / n,
             "part des 10 premiers": 100 * sum(count for _, count in skeletons.most_common(10)) / n}
    for label, pattern in FEATURES.items():
        stats[label] = 100 * sum(bool(re.search(pattern, sql, re.I | re.S)) for sql in sqls) / n
    stats["sous-requête même table sans filtre"] = 100 * sum(map(same_table_unfiltered_subquery, sqls)) / n
    stats["jointure à table inutilisée"] = 100 * sum(map(unused_join, sqls)) / n
    stats["jointure superflue (sans GROUP BY)"] = 100 * sum(map(superfluous_join, sqls)) / n
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Dossier contenant validated.jsonl.")
    parser.add_argument("--drop-superfluous-joins", action="store_true",
                        help="Écarte de validated.jsonl les paires à jointure superflue (copie complète dans validated_all.jsonl).")
    args = parser.parse_args()

    validated = read_jsonl(args.input / "validated.jsonl")
    if args.drop_superfluous_joins:
        backup = args.input / "validated_all.jsonl"
        if backup.exists():
            raise SystemExit(f"{backup} existe déjà : le filtre a déjà été appliqué.")
        backup.write_text((args.input / "validated.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        kept = [row for row in validated if not superfluous_join(row["sql"])]
        (args.input / "validated.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
        print(f"Jointures superflues écartées : {len(validated) - len(kept)} · {len(kept)} paires conservées\n")
        validated = kept
    lot = [row["sql"] for row in validated]
    spider_hard = [assistant_sql(row) for row in read_jsonl(SPIDER_TRAIN) if row["hardness"] in ("hard", "extra")]
    old_d = [assistant_sql(row) for row in read_jsonl(EXPORT / "synthetic_data_D.jsonl")]
    a2 = [assistant_sql(row) for row in read_jsonl(EXPORT / "synthetic_data_A2.jsonl")]
    rng = random.Random(0)
    n = min(len(lot), len(spider_hard), len(old_d), len(a2))
    columns = {"Spider hard/extra": rng.sample(spider_hard, n), "Ancien D": rng.sample(old_d, n),
               "A2": rng.sample(a2, n), "Lot contrôlé": rng.sample(lot, n)}
    table = {name: describe(sqls) for name, sqls in columns.items()}

    print(f"Paires valides du lot : {len(lot)} · comparaison à taille égale (n = {n})")
    print("Difficulté Spider du lot :", dict(Counter(row["hardness"] for row in validated)))
    print("\n" + f"{'':38s}" + "".join(f"{name:>19s}" for name in table))
    for key in next(iter(table.values())):
        print(f"{key:38s}" + "".join(f"{table[name][key]:>17.1f} %" if key != "squelettes / 100 ex." else f"{table[name][key]:>19.1f}"
                                     for name in table))

    reference, checked = table["Spider hard/extra"], table["Lot contrôlé"]
    failures = []
    if checked["squelettes / 100 ex."] < MIN_SKELETON_RATIO * reference["squelettes / 100 ex."]:
        failures.append("trop peu de squelettes SQL distincts")
    if checked["part des 10 premiers"] > MAX_TOP10_SHARE:
        failures.append("les 10 premiers squelettes concentrent trop de paires")
    for feature in ("JOIN", "2+ JOIN", "sous-requête", "UNION/INTERSECT/EXCEPT", "ORDER BY"):
        if checked[feature] < MIN_FEATURE_RATIO * reference[feature]:
            failures.append(f"{feature} sous-représenté")
    if checked["sous-requête même table sans filtre"] > MAX_SAME_TABLE_SUBQUERY:
        failures.append("sous-requêtes tautologiques sur la même table")
    if checked["jointure à table inutilisée"] > 2 * reference["jointure à table inutilisée"] + 2:
        failures.append("jointures inutiles plus fréquentes que dans Spider")
    print("\nVERDICT :", "OK, le lot n'est pas dégénéré." if not failures else "ÉCHEC — " + " ; ".join(failures))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
