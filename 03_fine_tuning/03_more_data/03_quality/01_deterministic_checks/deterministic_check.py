"""Valide les paires question/SQL générées par Luna, base par base."""
from __future__ import annotations
import argparse, importlib, importlib.util, json, re, sqlite3, sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import urlopen

HERE = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "01_data" / "00_shared" / "01_python"))
from sql_utils import normalize_sql
SPIDER_VENDOR = WORK
SPIDER_URLS = {"process_sql.py": "https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py"}
SPIDER_EVALUATOR: Any = None
SPIDER_SCHEMA_CACHE: dict[Path, Any] = {}
def read(path: Path) -> list[dict[str, Any]]: return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def output_text(response: dict[str, Any]) -> str | None:
    for item in response.get("response", {}).get("body", {}).get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str): return content["text"]
    return None
def normal(text: str) -> str: return re.sub(r"\s+", " ", text.casefold()).strip(" .?!…")
def clean_sql(sql: Any) -> str | None:
    """Même nettoyage que 02_evaluation_brut_models avant SQLite."""
    if not isinstance(sql, str): return None
    thinking_block = re.match(r"^\s*<think>.*?</think>\s*", sql, flags=re.IGNORECASE | re.DOTALL)
    if thinking_block: sql = sql[thinking_block.end():]
    sql = re.sub(r"^```(?:sql|sqlite)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql).strip()
    sql = re.sub(r"^SQL(?:Query)?\s*:\s*", "", sql, flags=re.IGNORECASE)
    return normalize_sql(sql.rstrip(";"))
def load_spider_evaluator() -> Any:
    """Charge le parseur officiel et la même règle que train_with_hardness."""
    global SPIDER_EVALUATOR
    if SPIDER_EVALUATOR is not None: return SPIDER_EVALUATOR
    SPIDER_VENDOR.mkdir(parents=True, exist_ok=True)
    for name, url in SPIDER_URLS.items():
        path = SPIDER_VENDOR / name
        if not path.is_file(): path.write_bytes(urlopen(url).read())
    if str(SPIDER_VENDOR) not in sys.path: sys.path.insert(0, str(SPIDER_VENDOR))
    process_sql = importlib.import_module("process_sql")
    classifier_path = HERE.parent / "02_train_data_scaling" / "build_train_with_hardness.py"
    spec = importlib.util.spec_from_file_location("train_hardness", classifier_path)
    if spec is None or spec.loader is None: raise RuntimeError("Impossible de charger la règle de difficulté du train.")
    classifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(classifier)
    SPIDER_EVALUATOR = (process_sql.get_schema, process_sql.Schema, process_sql.get_sql, classifier.spider_hardness)
    return SPIDER_EVALUATOR

def check_sql(sql: Any, database_path: Any) -> tuple[str | None, str | None]:
    if not isinstance(sql, str) or not re.match(r"^\s*(select|with)\b", sql, re.I): return "sql_not_read_only", None
    path = Path(database_path)
    if not path.is_file(): return "database_missing", None
    try:
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection: connection.execute(sql).fetchmany(1)
    except sqlite3.Error: return "sql_execution_error", None
    try:
        get_schema, schema_type, get_sql, spider_hardness = load_spider_evaluator()
        schema = SPIDER_SCHEMA_CACHE.setdefault(path, schema_type(get_schema(str(path))))
        return None, spider_hardness(get_sql(schema, sql))
    except Exception: return "spider_parse_error", None
def ask_experiment(value: str | None) -> str:
    choice = value or input("Expérience à contrôler [A/B/C/D] : ").strip().upper()
    if choice not in "ABCD": raise ValueError("Choix attendu : A, B, C ou D.")
    return choice
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--experiment", choices="ABCD"); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args(); args.experiment = ask_experiment(args.experiment)
    seed_source = "existing" if args.experiment in {"A", "C"} else "new"
    seeds = {f"{args.experiment}:{row['db_id']}": row for row in read(HERE / "01_prepare" / f"{seed_source}_sql_bases.jsonl")}
    generation_mode = "hard" if args.experiment in {"C", "D"} else "random"
    state_path = HERE / "02_batch" / "02_submissions" / f"{args.experiment}_batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    raw_name = state.get("raw_output_file", f"{args.experiment}.jsonl")
    accepted, rejected, returned, questions, pairs, count_by_db = [], [], set(), set(), set(), Counter()
    for response in read(HERE / "02_batch" / "03_responses" / raw_name):
        identifier = response.get("custom_id"); seed = seeds.get(identifier); returned.add(identifier); text = output_text(response)
        if not seed: rejected.append({"id": identifier, "reason": "unknown_database"}); continue
        try: generated = json.loads(text or "")["examples"]
        except (json.JSONDecodeError, KeyError, TypeError): rejected.append({"id": identifier, "reason": "invalid_json", "request_quota": seed["request_quota"], "target_quota": seed["target_quota"]}); continue
        for index, example in enumerate(generated):
            question, raw_sql = example.get("question"), example.get("sql"); sql = clean_sql(raw_sql); question_key = normal(question) if isinstance(question, str) else ""; sql_key = normal(sql) if isinstance(sql, str) else ""; record_id = f"{identifier}:{index}"
            if len(question_key) < 8: rejected.append({"id": record_id, "reason": "question_empty_or_short"}); continue
            if question_key in questions or (question_key, sql_key) in pairs: rejected.append({"id": record_id, "reason": "duplicate"}); continue
            reason, hardness = check_sql(sql, seed.get("database_path"))
            if reason: rejected.append({"id": record_id, "reason": reason}); continue
            questions.add(question_key); pairs.add((question_key, sql_key)); count_by_db[seed["db_id"]] += 1; accepted.append({"id": record_id, "db_id": seed["db_id"], "schema": seed["schema"], "hardness": hardness, "generation_mode": generation_mode, "question": question.strip(), "sql": sql.strip()})
    for identifier in sorted(set(seeds) - returned): rejected.append({"id": identifier, "reason": "missing_batch_response"})
    output_dir = WORK / args.experiment
    output = output_dir / "validated.jsonl"
    if output.exists() and not args.overwrite:
        if input(f"{output.name} existe déjà. Le remplacer ? [o/N] ").strip().lower() not in {"o", "oui"}: return
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted), encoding="utf-8")
    selected, selected_by_db = [], Counter()
    for row in accepted:
        if selected_by_db[row["db_id"]] < seeds[f"{args.experiment}:{row['db_id']}"]["target_quota"]:
            selected.append(row); selected_by_db[row["db_id"]] += 1
    shortfalls = {seed["db_id"]: seed["target_quota"] - selected_by_db[seed["db_id"]] for seed in seeds.values() if seed["target_quota"] > selected_by_db[seed["db_id"]]}
    write = lambda path, rows: path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    write(output, accepted); write(output_dir / "selected.jsonl", selected)
    manifest = {"experiment": args.experiment, "requested": sum(row["request_quota"] for row in seeds.values()), "target": sum(row["target_quota"] for row in seeds.values()), "valid_candidates": len(accepted), "selected": len(selected), "hardness_candidates": dict(Counter(row["hardness"] for row in accepted)), "hardness_selected": dict(Counter(row["hardness"] for row in selected)), "target_shortfall": sum(shortfalls.values()), "shortfall_by_database": shortfalls, "rejected": len(rejected), "reasons": dict(Counter(row["reason"] for row in rejected)), "rejections": rejected}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(accepted)} candidats valides, {len(selected)} sélectionnés, manque={sum(shortfalls.values())} : {output}")
if __name__ == "__main__": main()
