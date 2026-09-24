# Pipeline de données

Les dossiers suivent toujours l’ordre réel du pipeline. À chaque niveau, le préfixe numérique fixe la lecture : scripts, environnement (`pilot`, puis `production`), puis artefacts.

```text
00_documentation/pipeline_visualisation/       documentation HTML du pipeline
00_documentation/dataset_statistics/            interface de statistiques des datasets finaux
00_shared/01_python/                           utilitaires Python partagés
01_prepare/
  01_scripts/                                  vérifier et préparer Spider anglais
  02_output/                                   JSONL anglais enrichis du schéma
02_translation_batches/
  01_scripts/                                  créer et soumettre les lots Mistral
  02_pilot/ | 03_production/
    01_requests/  02_batch_state/  03_manual_translations/
03_batch_responses/
  01_scripts/
  02_pilot/ | 03_production/
04_merge_translations/
  01_scripts/
  02_pilot/ | 03_production/
05_quality_control/
  01_scripts/
  02_pilot/ | 03_production/
    01_deterministic_checks/  02_manual_corrections/  03_judge_selection/
    04_judge_requests/  05_judge_requests_test/  06_judge_raw_response/
    07_judge_raw_response_test/  08_judge_batch_state/  09_judgments/
06_training_dataset/
  01_scripts/  02_pilot/ | 03_production/
07_evaluation_dataset/
  01_scripts/  02_pilot/ | 03_production/
```

Le pipeline part de Spider original en anglais. `BRUT_spider-fr` reste une ressource de référence, sans être une entrée active.

Ordre d’exécution — uniquement lorsqu’une nouvelle production est souhaitée :

```powershell
python 01_data/01_prepare/01_scripts/01_check_original_dataset.py
python 01_data/01_prepare/01_scripts/02_prepare_english_dataset.py
python 01_data/02_translation_batches/01_scripts/01_create_and_send_batches.py
python 01_data/03_batch_responses/01_scripts/01_download_batch_responses.py
python 01_data/04_merge_translations/01_scripts/01_merge_translations.py
python 01_data/05_quality_control/01_scripts/01_check_translations.py
python 01_data/05_quality_control/01_scripts/02_judge_translations.py
python 01_data/06_training_dataset/01_scripts/01_build_dataset.py
python 01_data/07_evaluation_dataset/01_scripts/01_prepare_test.py
```

Les artefacts existants ont été déplacés sans être recalculés. `test` passe par la traduction et les contrôles, mais ne rejoint pas le dataset de fine-tuning ; il est réservé à l’évaluation.

Les réponses brutes, sélections du juge, fichiers de relance et corrections manuelles restent séparés pour permettre d’auditer chaque erreur sans confondre les résultats finaux.
