# Données Spider-FR préparées

Ces fichiers JSONL sont générés par `prepare_dataset.py` à partir des sources
brutes de `BRUT_spider-fr/` :

- `train_spider.json` → `train_spider.jsonl`
- `train_others.json` → `train_others.jsonl`
- `dev.json` → `dev.jsonl`

Pour chaque exemple, le script conserve la question française source et la
requête SQL, puis lit le schéma de la base SQLite correspondant au `db_id` dans
`BRUT_spider-original/data/spider_data/database/<db_id>/<db_id>.sqlite`.
Le schéma est construit depuis `sqlite_master` avec les instructions
`CREATE TABLE`, en ignorant les tables internes dont le nom commence par
`sqlite_`.

Chaque ligne JSONL contient exactement ces champs :

```json
{
  "db_id": "department_management",
  "question_original": "Combien de chefs des départements sont plus âgés que 56?",
  "schema": "CREATE TABLE ...",
  "sql": "SELECT count(*) FROM head WHERE age > 56"
}
```

Les fichiers source ne sont jamais modifiés. Avant écriture, le script vérifie
la présence de `db_id`, `question`, `query` et de la base SQLite associée. Les
doublons exacts selon `db_id + question_original + sql` sont retirés dans chaque
split, en conservant la première occurrence.

Pour les régénérer :

```powershell
python data/preparation_scripts/02_prepare_dataset.py
```
