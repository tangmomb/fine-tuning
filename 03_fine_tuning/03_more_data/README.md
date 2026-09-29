# Pipeline de données synthétiques supplémentaires

Ce dossier prépare les quatre expériences du compte rendu.

| Expérience | Sélection SQL | Bases |
| --- | --- | --- |
| A | distribution historique des difficultés | déjà présentes dans le train |
| B | aléatoire | nouvelles |
| C | difficile | déjà présentes dans le train |
| D | difficile | nouvelles |

Un seul job OpenAI Batch contient une requête par base : pour A, les 146 schémas Spider sont donc envoyés une seule fois chacun. GPT-5.6 Luna produit 6 250 paires (25 % de marge), puis les contrôles conservent au plus 5 000 exemples, selon les quotas par schéma et par difficulté. Chaque SQL généré est exécuté sur SQLite avant export.

## Contrat d'entrée

Les expériences A/C lisent les données Spider enrichies. Pour B/D, déposez un catalogue externe dans `01_prepare/00_input/new_databases.jsonl` (non versionné), une ligne par base :

```json
{"db_id":"ma_base","schema":"CREATE TABLE ...;","source_examples":120,"hardness":"mixed","database_path":"C:/.../ma_base.sqlite"}
```

`database_path` est obligatoire : le pipeline y exécute chaque SQL généré. `source_examples` sert à répartir les 5 000 quotas proportionnellement au volume de chaque base.

## Exécution

Depuis la racine du dépôt :

```powershell
python 03_fine_tuning/03_more_data/01_prepare/prepare_seeds.py --source existing
python 03_fine_tuning/03_more_data/01_prepare/prepare_seeds.py --source new
python 03_fine_tuning/03_more_data/02_batch/create_batch.py
python 03_fine_tuning/03_more_data/02_batch/download_batch.py
python 03_fine_tuning/03_more_data/03_quality/01_deterministic_checks/deterministic_check.py
python 03_fine_tuning/03_more_data/03_quality/02_semantic_judge/judge_semantics.py
python 03_fine_tuning/03_more_data/04_export/build_sft.py
```

La préparation écrit deux plans réutilisables : `01_prepare/existing_sql_bases.jsonl` (A/C) et `01_prepare/new_sql_bases.jsonl` (B/D). Les étapes suivantes demandent l’expérience ; la soumission à OpenAI nécessite une confirmation explicite. Chaque étape écrit ses résultats dans son propre dossier : `02_batch/01_requests/A_requests.jsonl`, puis les états et réponses brutes du Batch → `03_quality/01_deterministic_checks/A/selected.jsonl` → `04_export/synthetic_data_A.jsonl`. Le juge sémantique produit séparément un audit de 200 exemples dans `03_quality/02_semantic_judge/04_judgments/`.

Dans `02_batch`, `01_requests` contient le JSONL exact envoyé, `02_submissions` l’état et les identifiants associés, et `03_responses` les réponses et erreurs brutes téléchargées. Ces artefacts bruts ne sont jamais remplacés sans `--overwrite`.

## Garde-fous

- Les fichiers existants ne sont pas remplacés sans `--overwrite`.
- Le modèle est `gpt-5.6-luna`, configurable avec `--model`.
- La génération autorise jusqu’à 128&nbsp;000 tokens de sortie par prompt, pour que les bases aux quotas élevés puissent fermer leur JSON structuré.
- Le contrôle déterministe exclut les questions vides ou dupliquées et les SQL qui ne sont pas des lectures SQLite exécutables, mais conserve les surplus valides comme réserve.
- Un second Batch, confié à GPT-5.6 Terra, juge 200 paires question/SQL via un échantillon stratifié par difficulté et réparti entre les bases, afin de mesurer la fidélité sémantique sans facturer un jugement exhaustif.
- Une revue humaine/LLM est recommandée avant un entraînement définitif.
