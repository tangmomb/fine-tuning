# Mémo : terminer l'expérience A2 dans une nouvelle session

*Rédigé le 2026-11-14. À coller ou à référencer au début d'une nouvelle session Claude Code ouverte dans `E:\cooking\fine-tuning`.*

## Prompt de démarrage à coller

> Lis `MEMO_suite_experience_A2.md` à la racine du dépôt. Il décrit l'état de l'expérience A2 et ce qu'il reste à faire. Commence par l'étape 1 (option `--test-epoch`), puis guide-moi pour les évaluations sur le H100. Ne consulte pas le jeu de test en dehors de ce protocole.

## Où en est-on

- **Projet** : fine-tuning LoRA de Qwen 3.5 2B en text-to-SQL français (Spider traduit). Le compte rendu se trouve dans `00_compte_rendu/index.html` et est publié sur https://fine-tuning.tanguym.fr ; il suffit de pousser sur `master` pour le mettre à jour (voir `README.md`).
- **Diagnostic** (chapitre 06.4) : les anciens datasets synthétiques A à D n'apportaient rien. Les prompts de génération imposaient un seul gabarit SQL par difficulté, d'où des données sans JOIN, sans ORDER BY et sans LIMIT. Les scripts sont dans `03_fine_tuning/03_05_diagnostic/`.
- **Dataset A2** (chapitre 06.5), construit dans `03_fine_tuning/03_06_diverse_data/` :
  - 5 000 paires générées à partir d'exemples Spider, sur les 146 bases du train ;
  - toutes validées par un juge Claude Haiku 5.5 ;
  - export dans `03_fine_tuning/03_03_more_data/03_03_04_export/synthetic_data_A2.jsonl`.
- **Résultats sur la validation** (chapitre 06.6, déjà rédigé) : A2 bat le baseline (train Spider seul) sur les deux graines, avec un test apparié à p < 0,01 dans les deux cas.

  | Graine | Baseline | A2 | Gain |
  |---|---|---|---|
  | 42 | 66,44 % | 70,41 % | +3,97 points |
  | 7 | 65,67 % | 69,34 % | +3,67 points |

  Les erreurs « no such column » baissent d'environ 25 %.
- **Le jeu de test n'a jamais été consulté pour ces runs.**

### Les 4 runs à évaluer sur le test (époque 3)

| Graine | Baseline | A2 |
|---|---|---|
| 42 | `03_fine_tuning/artifacts/qwen3.5-2b-20260926-125948Z_r8_alpha16_lr1e-4` | `03_fine_tuning/artifacts/data-scaling-new-data-20261011/A2/more-data-a2-20261011-205817Z` |
| 7 | `03_fine_tuning/artifacts/baseline-r8-seed7` | `03_fine_tuning/artifacts/data-scaling-new-data-20261104/A2/more-data-a2-seed7` |

Configuration commune : Qwen3.5-2B, LoRA preset A (r=8, α=16), learning rate 1e-4, 3 époques, batch 2 × accumulation 32 = 64.

**Attention aux machines** : les runs en graine 42 ont été entraînés en Pologne, ceux en graine 7 à Paris (environnement dans `03_fine_tuning/artifacts/_paris_logs/pip_freeze_paris.txt`). On ne compare donc qu'à l'intérieur d'une même graine.

## Ce qu'il reste à faire

### 1. Ajouter une évaluation test seule (en local, sans GPU)

`03_fine_tuning/03_01_train_main/train.py --evaluate-run <run> --evaluate-test` pose deux problèmes :
- il réévalue d'abord la validation des 3 époques et **écrase** les fichiers `eval/epoch-N/val_predictions.jsonl`. Ce sont ceux qu'utilisent `rescore_validation.py` et le compte rendu ;
- il demande l'époque à tester de façon interactive.

→ Ajouter une option `--test-epoch N` : elle charge seulement le checkpoint `epoch-N`, évalue le test, écrit dans `eval/epoch-N/final_test/` et ne touche pas à la validation.
- Le code s'appuie sur `evaluate_existing_run()` (vers la ligne 665) et `evaluate_test_execution()` (vers la ligne 335).
- Pour les runs dont le nom ne commence pas par `Qwen3.5-2B` (A2 et `baseline-r8-seed7`), il faut aussi passer `--models Qwen3.5-2B`.
- Vérifier aussi le lanceur `03_fine_tuning/03_04_train_more_data/train.py` : il ajoute ses options fixes, sauf quand `--evaluate-run` est présent.
- Commiter et pousser.

### 2. Lancer les 4 évaluations sur le H100

- **Toutes sur la même machine**, à l'époque 3, une seule fois.
- **Si le H100 est nouveau**, les checkpoints n'y sont pas : il faut envoyer les 4 dossiers de runs, ou au minimum `checkpoints/epoch-3`, `tokenizer/` et `eval/` (environ 33 Mo par checkpoint). Il faut aussi les poids du modèle dans `models/Qwen3.5-2B` et le `.venv`.
- **Rapatriement** : le script `03_fine_tuning/download_fine_tuning_artifacts.py` ignore les dossiers qui existent déjà en local. Les fichiers `final_test/` ne seront donc **pas** récupérés par ce script : utiliser `scp` sur les sous-dossiers `eval/epoch-3/final_test/` de chaque run.
- **Clé SSH** : elle est maintenant `~/.ssh/scaleway_fine_tuning`. La modification n'est pas commitée, voir l'étape 4.

### 3. Analyser et rédiger

- Comparer baseline et A2 sur le test, graine par graine : test de McNemar apparié et bootstrap par base. On peut s'inspirer de `03_fine_tuning/03_05_diagnostic/compare_runs.py`, qui ne lit pour l'instant que `val_predictions.jsonl` : il faut l'adapter aux prédictions du test, sans oublier la comparaison façon Spider.
- Points de comparaison existants sur le test : 2B r=16 à 67,44 % et 4B à 73,17 % (chapitre 03). Le 67,44 % ne vient pas de la configuration r=8, d'où l'intérêt d'évaluer le baseline r=8.
- Ajouter les résultats sur le test au chapitre 06.6, avec une conclusion de chapitre, puis pousser pour publier.

### 4. Ménage

- Commiter le changement de clé SSH dans `02_evaluation_brut_models/download_scaleway_runs.py` et `03_fine_tuning/download_fine_tuning_artifacts.py`.

## Pistes après cette expérience (hors périmètre)

- **B2** : 100 bases fictives, même méthode que A2. Le prompt de génération est dans `03_06_diverse_data/create_requests.py` (`--bases new`).
- **A2 sur Qwen 3.5 4B.**
- **Points faibles restants** : requêtes à 2 jointures ou plus (36 % de réussite) et niveau extra sous-représenté dans A2 (15,8 % contre 20,3 % dans Spider).

## Pièges déjà rencontrés

- **Échappements dans le shell** : sous Git Bash, les `\b` et `\n` à l'intérieur de scripts Python passés par heredoc sont corrompus. Écrire les fichiers avec l'outil d'écriture, pas avec `python - <<EOF`.
- **Arguments commençant par `/`** : Git Bash les transforme en chemins Windows. Écrire `A/more-data-a`, pas `/A/more-data-a`.
- **Bases Spider vides** : 7 bases sont livrées sans aucune ligne (geo, scholar, academic, imdb, yelp, restaurants, music_2). Un résultat vide y est normal.
- **Clés d'API** : OpenAI et Anthropic sont dans `.env` à la racine. Le SDK `anthropic` est installé dans le Python global.
