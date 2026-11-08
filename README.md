# Fine-tuning text-to-SQL

Le dossier `01_data/` est réservé à la préparation et aux jeux de données. Les
scripts, sorties et synthèses de benchmark des modèles bruts se trouvent dans
`02_evaluation_brut_models/`.

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

## Publication du rapport

Le rapport `00_compte_rendu/` est publié sur https://fine-tuning.tanguym.fr, servi par Caddy sur le VPS
Contabo (95.111.224.166). Le VPS garde dans `/srv/fine-tuning` une copie partielle de ce dépôt, limitée à
`00_compte_rendu/`, mise à jour depuis `master` toutes les 10 minutes par le timer systemd
`fine-tuning-pull.timer`. Il suffit donc de pousser sur `master` ; pour publier tout de suite :
`sudo systemctl start fine-tuning-pull.service`.

La configuration du serveur (bloc Caddy et timer) se trouve dans `deploy/` du dépôt
`tangmomb/agent_video_edit_detection`.
