"""Re-score les prédictions de validation archivées avec une comparaison façon Spider.

L'évaluateur du projet compare les résultats SQLite en liste stricte : l'ordre
des lignes compte même sans ORDER BY, et l'ordre des colonnes compte aussi.
Ce script ré-exécute gold et prédiction, puis compare :
  - strict      : comparaison actuelle (liste ordonnée) ;
  - multiset    : lignes comparées sans ordre si le gold n'a pas d'ORDER BY ;
  - spider      : multiset + permutation des colonnes autorisée (test-suite Spider).
"""

from __future__ import annotations

import itertools
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "03_fine_tuning" / "artifacts"
DATABASES = ROOT / "BRUT_spider-original" / "data" / "spider_data" / "database"
OUTPUT = Path(__file__).resolve().parent / "rescore_validation.json"
TIMEOUT_SECONDS = 5.0


def execute(db_id: str, sql: str, cache: dict) -> list[tuple] | None:
    key = (db_id, sql)
    if key in cache:
        return cache[key]
    result = None
    if sql.lstrip().upper().startswith(("SELECT", "WITH")):
        path = DATABASES / db_id / f"{db_id}.sqlite"
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        connection.text_factory = lambda value: value.decode("utf-8", errors="replace")
        deadline = time.perf_counter() + TIMEOUT_SECONDS
        connection.set_progress_handler(lambda: int(time.perf_counter() >= deadline), 1_000)
        try:
            result = [tuple(row) for row in connection.execute(sql).fetchall()]
        except sqlite3.Error:
            result = None
        finally:
            connection.close()
    cache[key] = result
    return result


def has_order_by(sql: str) -> bool:
    return re.search(r"\border\s+by\b", sql, flags=re.IGNORECASE) is not None


def rows_match(pred: list[tuple], gold: list[tuple], ordered: bool) -> bool:
    if len(pred) != len(gold):
        return False
    if ordered:
        return pred == gold
    return Counter(map(repr, pred)) == Counter(map(repr, gold))


def spider_match(pred: list[tuple], gold: list[tuple], ordered: bool) -> bool:
    if rows_match(pred, gold, ordered):
        return True
    if not pred or not gold or len(pred[0]) != len(gold[0]) or len(pred[0]) > 6:
        return False
    width = len(gold[0])
    for permutation in itertools.permutations(range(width)):
        permuted = [tuple(row[i] for i in permutation) for row in pred]
        if rows_match(permuted, gold, ordered):
            return True
    return False


def main() -> None:
    cache: dict = {}
    report: dict[str, Any] = {}
    files = sorted(p for p in ARTIFACTS.rglob("val_predictions.jsonl") if "_discarded" not in p.parts)
    for path in files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts = Counter()
        per_example = {}
        for row in rows:
            gold = execute(row["db_id"], row["gold_sql"], cache)
            pred = execute(row["db_id"], row["predicted_sql"], cache)
            ordered = has_order_by(row["gold_sql"])
            strict = bool(row["execution_match"])
            multiset = pred is not None and gold is not None and rows_match(pred, gold, ordered)
            spider = pred is not None and gold is not None and spider_match(pred, gold, ordered)
            counts["strict"] += strict
            counts["multiset"] += multiset or strict
            counts["spider"] += spider or strict
            per_example[row["id"]] = bool(spider or strict)
        label = str(path.parent.relative_to(ARTIFACTS)).replace("\\", "/")
        report[label] = {"n": len(rows), **{k: counts[k] / len(rows) for k in ("strict", "multiset", "spider")},
                         "spider_correct": per_example}
        print(f"{label:80s} strict {counts['strict'] / len(rows):.2%}  multiset {counts['multiset'] / len(rows):.2%}  spider {counts['spider'] / len(rows):.2%}")
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
