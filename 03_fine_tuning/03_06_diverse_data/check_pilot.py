"""Contrôle un lot généré par create_requests.py et compare sa diversité à Spider et à l'ancien dataset A."""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINE_TUNING = HERE.parent
sys.path.insert(0, str(FINE_TUNING / "03_03_more_data" / "03_03_03_quality" / "03_03_03_01_deterministic_checks"))
sys.path.insert(0, str(HERE))
from deterministic_check import check_sql, clean_sql, output_text  # noqa: E402
from create_requests import SPIDER_TRAIN, read_jsonl, skeleton, structure_class  # noqa: E402

OLD_A = FINE_TUNING / "03_03_more_data" / "03_03_04_export" / "synthetic_data_A.jsonl"
FEATURES = {"JOIN": r"\bjoin\b", "2+ JOIN": r"\bjoin\b.*\bjoin\b", "ORDER BY": r"\border by\b", "LIMIT": r"\blimit\b",
            "GROUP BY": r"\bgroup by\b", "HAVING": r"\bhaving\b", "sous-requête": r"\(\s*select\b",
            "UNION/INTERSECT/EXCEPT": r"\b(union|intersect|except)\b", "DISTINCT": r"\bdistinct\b"}


def is_tautology(sql: str) -> bool:
    match = re.search(r"from (\w+)\s+where (\w+) in \(\s*select (\w+) from (\w+)\s*\)", sql, re.I)
    return bool(match and match.group(1).lower() == match.group(4).lower() and match.group(2).lower() == match.group(3).lower())


def row_count(sql: str, database_path: str) -> int | None:
    try:
        with sqlite3.connect(f"file:{Path(database_path).as_posix()}?mode=ro", uri=True) as connection:
            return len(connection.execute(sql).fetchmany(1000))
    except sqlite3.Error:
        return None


@lru_cache(maxsize=None)
def database_is_empty(database_path: str) -> bool:
    with sqlite3.connect(f"file:{Path(database_path).as_posix()}?mode=ro", uri=True) as connection:
        tables = [name for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        return all(connection.execute(f'SELECT 1 FROM "{table}" LIMIT 1').fetchone() is None for table in tables)


def describe(name: str, sqls: list[str]) -> dict:
    n = len(sqls)
    stats = {"n": n, "squelettes distincts": len({skeleton(s) for s in sqls}),
             "squelettes / 100 ex.": round(100 * len({skeleton(s) for s in sqls}) / n, 1)}
    for label, rx in FEATURES.items():
        stats[label] = f"{100 * sum(bool(re.search(rx, s, re.I | re.S)) for s in sqls) / n:.1f} %"
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "pilot")
    args = parser.parse_args()

    meta = {row["custom_id"]: row for row in read_jsonl(args.input / "request_meta.jsonl")}
    candidates, reasons, fidelity, seen = [], Counter(), Counter(), set()
    for response in read_jsonl(args.input / "responses.jsonl"):
        info = meta[response["custom_id"]]
        text = output_text(response)
        examples = json.loads(text)["examples"] if text else []
        per_exemplar = max(1, len(examples) // max(1, len(info["exemplars"])))
        for index, example in enumerate(examples):
            sql = clean_sql(example.get("sql"))
            question = (example.get("question") or "").strip()
            target = info["exemplars"][min(index // per_exemplar, len(info["exemplars"]) - 1)]
            rejection, hardness = check_sql(sql, info["database_path"])
            db_id = response["custom_id"].split(":")[2]
            if not rejection and ((db_id, question.casefold()) in seen or (db_id, sql.casefold()) in seen):
                rejection = "duplicate"
            if not rejection and is_tautology(sql):
                rejection = "tautology"
            rows = None if rejection else row_count(sql, info["database_path"])
            # Certaines bases Spider (geo, scholar, academic...) sont livrées vides : tout résultat y est vide.
            if not rejection and rows == 0 and not database_is_empty(info["database_path"]):
                rejection = "empty_result"
            reasons[rejection or "ok"] += 1
            if rejection:
                continue
            seen.update({(db_id, question.casefold()), (db_id, sql.casefold())})
            fidelity["structure identique à l'exemple" if structure_class(sql) == target["structure"] else "structure différente"] += 1
            candidates.append({"id": f"{response['custom_id']}:{index}", "custom_id": response["custom_id"], "db_id": db_id,
                               "database_path": info["database_path"], "question": question, "sql": sql, "hardness": hardness,
                               "exemplar_hardness": target["hardness"], "exemplar_sql": target["sql"]})

    (args.input / "validated.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in candidates) + "\n", encoding="utf-8")
    total = sum(reasons.values())
    print(f"\nPaires reçues : {total}")
    for reason, count in reasons.most_common():
        print(f"  {reason:22s} {count:4d}  {100 * count / total:5.1f} %")
    print("\nFidélité structurelle des paires valides :", dict(fidelity))
    print("Difficulté Spider des paires valides :", dict(Counter(row["hardness"] for row in candidates)))
    print("Difficulté des exemples Spider ciblés :", dict(Counter(row["exemplar_hardness"] for row in candidates)))

    spider = [(row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"]))[2]["content"]
              for row in read_jsonl(SPIDER_TRAIN)]
    old_a = [json.loads(line)["messages"][2]["content"] for line in OLD_A.read_text(encoding="utf-8").splitlines() if line.strip()]
    pilot = [row["sql"] for row in candidates]
    rng = random.Random(0)
    # Le nombre de squelettes dépend de n : on compare à taille égale.
    n = min(len(pilot), len(spider), len(old_a))
    columns = {"Spider train (même n)": rng.sample(spider, n), "Ancien A (même n)": rng.sample(old_a, n), "Lot généré": rng.sample(pilot, n)}
    table = {name: describe(name, sqls) for name, sqls in columns.items()}
    keys = list(next(iter(table.values())))
    print("\n" + f"{'':26s}" + "".join(f"{name:>24s}" for name in table))
    for key_ in keys:
        print(f"{key_:26s}" + "".join(f"{str(table[name][key_]):>24s}" for name in table))


if __name__ == "__main__":
    main()
