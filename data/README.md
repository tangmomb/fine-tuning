# Pipeline de données

Le pipeline actif part de Spider original en anglais : les questions anglaises sont traduites directement en français. `BRUT_spider-fr` est conservé comme ressource de référence, mais n'est plus l'entrée du pipeline.

```text
01_processed/          JSONL anglais enrichis du schéma SQLite
02_batch/              requêtes Batch prêtes à transmettre au fournisseur configuré
03_mistral_response/   réponses JSONL brutes téléchargées depuis Mistral
04_translated_fr/      données finales : anglais source + question française
05_checks/             contrôles locaux des traductions
06_fine_tuning_ready/  données chat JSONL pour entraîner le modèle
07_evaluation/         entrées de test et SQL gold séparés pour l'évaluation
```

Chaque script est rangé dans le dossier de l'étape qu'il produit.

L'ordre d'exécution est le suivant :

```powershell
python data/01_processed/01_check_original_dataset.py
python data/01_processed/02_prepare_english_dataset.py
python data/02_batch/01_create_and_send_batches.py
python data/03_mistral_response/01_download_batch_responses.py
python data/04_translated_fr/01_merge_translations.py
python data/05_checks/01_check_translations.py
python data/05_checks/02_judge_translations.py
python data/06_fine_tuning_ready/01_build_dataset.py
python data/07_evaluation/01_prepare_test.py
```

Les scripts des étapes 02, 03, 04, 05 et 06 demandent au démarrage le dossier cible : `pilot` ou
`production`. Les artefacts restent isolés dans le sous-dossier choisi.

Le script `02_batch/01_create_and_send_batches.py` prépare localement les lots Z.ai GLM 5.3 hébergés par Mistral, puis demande explicitement avant de les envoyer. Le mode `pilot` prépare deux lots de 50 requêtes ; le mode `production` prépare les quatre splits, dont `test`. Il utilise \`MISTRAL_API_KEY\`, le modèle \`zai-glm-5-3\` et l'endpoint Batch Mistral. Chaque requête demande uniquement la phrase française : l'identifiant est porté par `custom_id` du Batch.

Le script `03_mistral_response/01_download_batch_responses.py` affiche l'état des lots puis télécharge leurs sorties JSONL brutes lorsqu'ils sont terminés. Le script `04_translated_fr/01_merge_translations.py` valide les réponses, applique les éventuelles corrections de `02_batch/<dossier>/manual_translations.jsonl`, puis les fusionne avec les données sources enrichies (question anglaise, SQL et schéma).

Le split `test` traverse les étapes 02 à 05 pour obtenir et contrôler ses questions françaises, mais l'étape 06 l'exclut du dataset de fine-tuning : il reste réservé à l'évaluation finale.

Le script `07_evaluation/01_prepare_test.py` prépare les 2 147 entrées à envoyer au modèle et leurs SQL gold normalisés dans deux fichiers distincts. Les prompts ne contiennent jamais le SQL gold.

Pour lancer un jugement sémantique séparé du seul split de test, sans relancer train/dev, utilisez `python data/05_checks/02_judge_translations.py --splits test`.

Pour soumettre volontairement de nouveaux lots alors qu'un état existe déjà, utilisez :

\`\`\`powershell
python data/02_batch/01_create_and_send_batches.py --resubmit
\`\`\`

Lorsque train/dev ont déjà été traduits et que seul le test doit être ajouté, utilisez plutôt `python data/02_batch/01_create_and_send_batches.py --splits test` : aucun batch existant n'est recréé.

Pour renvoyer uniquement les réponses brutes invalides ou incomplètes déjà téléchargées, utilisez `--retry-missing` avec ce même script.

Le quatrième script produit un fichier `05_checks/<dossier>/<split>_deterministic_checks.jsonl`. Il vérifie localement les champs obligatoires, les nombres, les pourcentages et l'absence apparente de SQL dans la traduction.

Le script `data/05_checks/02_judge_translations.py` remet en place le juge sémantique Batch avec `gpt-5.6-sol`. Il prépare deux lots, demande confirmation avant l'envoi, puis écrit les verdicts dans `05_checks/pilot/sol_judgments.jsonl`.
