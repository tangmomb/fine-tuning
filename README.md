# Fine-tuning text-to-SQL

## Contenu brut de Spider original

Le dataset original est conservé dans `BRUT_spider-original/data/spider_data/` :

```text
spider_data/
├── database/            # bases SQLite d’entraînement/dev
├── test_database/       # bases SQLite réservées au test
├── tables.json          # métadonnées détaillées de tous les schémas
│
├── train_spider.json    # questions + SQL du train principal
├── train_others.json    # questions + SQL des autres jeux d’entraînement
├── dev.json             # questions + SQL de validation
├── test.json            # questions de test — sans SQL public utilisable
│
├── train_gold.sql       # SQL du train, format évaluation officielle
├── dev_gold.sql         # SQL du dev, format évaluation officielle
├── test_gold.sql        # SQL du test, format évaluation officielle
├── test_tables.json     # schémas des bases de test
│
├── README.txt           # documentation originale Spider
└── .DS_Store            # fichier macOS inutile
```

Pour le pipeline actuel, les ressources utilisées sont `database/`, `tables.json`, `train_spider.json`, `train_others.json` et `dev.json`. Les autres éléments sont conservés afin de préserver une copie complète du dataset brut.
