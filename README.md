# Fine-tuning text-to-SQL

## Contenu brut de Spider original

Le dataset original est conservé dans `BRUT_spider-original/data/spider_data/` :

```text
spider_data/
├── database/            # 166 bases SQLite utilisées par train et dev
├── test_database/       # 206 bases : les 166 ci-dessus + les 40 bases du test
├── tables.json          # schémas des 166 bases train/dev
│
├── train_spider.json    # questions + SQL du train principal
├── train_others.json    # questions + SQL des autres jeux d’entraînement
├── dev.json             # questions + SQL de validation
├── test.json            # questions + SQL de test
│
├── train_gold.sql       # 8 659 SQL : train_spider puis train_others, avec db_id
├── dev_gold.sql         # SQL du dev, format évaluation officielle
├── test_gold.sql        # SQL du test, format évaluation officielle
├── test_tables.json     # schémas des 40 bases de test
│
├── README.txt           # documentation originale Spider
└── .DS_Store            # fichier macOS inutile
```

Pour le pipeline actuel, les ressources utilisées sont `database/`, `tables.json`, `train_spider.json`, `train_others.json` et `dev.json`. Le split `test` est réservé à l'évaluation finale et n'est pas utilisé pour le fine-tuning ni pour les décisions de développement. Les autres éléments sont conservés afin de préserver une copie complète du dataset brut.

`test_database/` duplique exactement les 166 bases présentes dans `database/` et ajoute les 40 bases du test. Cette redondance vient de l'organisation du package Spider : elle permet d'exécuter toutes les évaluations depuis un seul répertoire. Pour le pipeline, utiliser `database/` suffit ; utiliser `test_database/` n'est nécessaire que pour exécuter les requêtes du split `test`.

En résumé :

- `train_spider.json` et `train_others.json` servent à l'apprentissage.
- `dev.json` sert à contrôler le modèle pendant le développement.
- `test.json` sert uniquement à l'évaluation finale.
- `database/` contient les bases train/dev.
- `test_database/` contient ces mêmes bases et les 40 bases du test.
