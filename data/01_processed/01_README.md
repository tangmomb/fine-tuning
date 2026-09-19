# Étape 01 — données Spider anglaises préparées

Ces fichiers JSONL sont produits depuis les trois splits anglais originaux :

```text
BRUT_spider-original/data/spider_data/
├── train_spider.json
├── train_others.json
└── dev.json
```

La commande de génération est :

```powershell
python data/01_processed/02_prepare_english_dataset.py
```

Pour chaque exemple, le script vérifie que `db_id`, `question` et `query` sont présents et non vides, vérifie la présence de la base SQLite correspondante, puis extrait les instructions `CREATE TABLE` depuis SQLite. Les tables internes `sqlite_%` sont exclues.

Chaque ligne contient exactement :

```json
{
  "db_id": "department_management",
  "question_original_en": "How many heads of departments are older than 56 ?",
  "schema": "CREATE TABLE ...",
  "sql": "SELECT count(*) FROM head WHERE age > 56"
}
```

Les doublons exacts `(db_id, question_original_en, sql)` sont éliminés à l'intérieur de chaque split. Les fichiers sources ne sont jamais modifiés. Cette étape constitue l'entrée de la traduction directe anglais → français de l'étape 02.
