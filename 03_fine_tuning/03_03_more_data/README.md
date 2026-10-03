# Pipeline de données synthétiques supplémentaires

Ce dossier prépare les quatre expériences du compte rendu.

| Expérience | Sélection SQL | Bases |
| --- | --- | --- |
| A | distribution historique des difficultés | déjà présentes dans le train |
| B | aléatoire | nouvelles |
| C | 50&nbsp;% hard / 50&nbsp;% extra | déjà présentes dans le train |
| D | 50&nbsp;% hard / 50&nbsp;% extra | nouvelles |

Un seul job OpenAI Batch peut contenir plusieurs requêtes par base. Pour A, le comportement standard sépare automatiquement les difficultés : 566 prompts couvrent les 146 schémas Spider et demandent exactement 6 250 paires (25 % de marge). Chaque prompt cible une seule difficulté et les demandes supérieures à 80 exemples sont découpées. Les contrôles conservent ensuite au plus 5 000 exemples selon les quotas par schéma et par difficulté. Pour C et D, le quota de chaque base est partagé à 50/50 entre `hard` et `extra` ; les arrondis des quotas impairs sont distribués entre les bases afin de garder un équilibre global exact. Chaque SQL généré est exécuté sur SQLite avant export.

## Contrat d'entrée

Les expériences A/C lisent les données Spider enrichies. Pour B/D, le catalogue est produit dans `03_03_01_prepare/03_03_01_01_new_databases_creation/03_03_01_01_04_final/new_databases.jsonl`, une ligne par base :

```json
{"db_id":"ma_base","schema":"CREATE TABLE ...;","source_examples":120,"hardness":"mixed","database_path":"C:/.../ma_base.sqlite"}
```

`database_path` est obligatoire : le pipeline y exécute chaque SQL généré. `source_examples` sert à répartir les 5 000 quotas proportionnellement au volume de chaque base.

## Exécution

Depuis la racine du dépôt :

```powershell
python 03_fine_tuning/03_03_more_data/03_03_01_prepare/prepare_seeds.py --source existing
python 03_fine_tuning/03_03_more_data/03_03_01_prepare/prepare_seeds.py --source new
python 03_fine_tuning/03_03_more_data/03_03_02_batch/create_batch.py --experiment A
python 03_fine_tuning/03_03_more_data/03_03_02_batch/download_batch.py
python 03_fine_tuning/03_03_more_data/03_03_03_quality/03_03_03_01_deterministic_checks/deterministic_check.py
python 03_fine_tuning/03_03_more_data/03_03_03_quality/03_03_03_02_semantic_judge/judge_semantics.py
python 03_fine_tuning/03_03_more_data/03_03_04_export/build_sft.py
```

La préparation écrit deux plans réutilisables : `03_03_01_prepare/existing_sql_bases.jsonl` (A/C) et `03_03_01_prepare/new_sql_bases.jsonl` (B/D). Les étapes suivantes demandent l’expérience ; la soumission à OpenAI nécessite une confirmation explicite. Chaque étape écrit ses résultats dans son propre dossier : `03_03_02_batch/03_03_02_01_requests/A_requests.jsonl`, puis les états et réponses brutes du Batch → `03_03_03_quality/03_03_03_01_deterministic_checks/A/selected.jsonl` → `03_03_04_export/synthetic_data_A.jsonl`. Le juge sémantique produit séparément ses artefacts dans `03_03_03_quality/03_03_03_02_semantic_judge/`.

Dans `03_03_02_batch`, `03_03_02_01_requests` contient le JSONL exact envoyé, `03_03_02_02_submissions` l’état et les identifiants associés, et `03_03_02_03_responses` les réponses et erreurs brutes téléchargées. Ces artefacts bruts ne sont jamais remplacés sans `--overwrite`.

## Garde-fous

- Les fichiers existants ne sont pas remplacés sans `--overwrite`.
- Le modèle est `gpt-5.6-luna`, configurable avec `--model`.
- La génération autorise jusqu’à 128&nbsp;000 tokens de sortie par prompt, pour que les bases aux quotas élevés puissent fermer leur JSON structuré.
- Le contrôle déterministe exclut les questions vides ou dupliquées et les SQL qui ne sont pas des lectures SQLite exécutables, mais conserve les surplus valides comme réserve.
- Un second Batch, confié à GPT-5.6 Terra, juge 200 paires question/SQL via un échantillon stratifié par difficulté et réparti entre les bases, afin de mesurer la fidélité sémantique sans facturer un jugement exhaustif.
- Une revue humaine/LLM est recommandée avant un entraînement définitif.
