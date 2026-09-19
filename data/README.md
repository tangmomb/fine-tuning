# Pipeline de données

Le pipeline actif part de Spider original en anglais : les questions anglaises sont traduites directement en français. `BRUT_spider-fr` est conservé comme ressource de référence, mais n'est plus l'entrée du pipeline.

```text
01_processed/          JSONL anglais enrichis du schéma SQLite
02_batch/              requêtes Batch prêtes à transmettre au fournisseur configuré
03_mistral_response/   réponses JSONL brutes téléchargées depuis Mistral
04_translated_fr/      données finales : anglais source + question française
05_checks/             contrôles locaux des traductions
```

Chaque script est rangé dans le dossier de l'étape qu'il produit.

L'ordre d'exécution est le suivant :

```powershell
python data/01_processed/01_check_original_dataset.py
python data/01_processed/02_prepare_english_dataset.py
python data/02_batch/02_translate_all.py
python data/05_checks/01_check_translations.py
python data/05_checks/02_judge_translations.py
python data/06_fine_tuning_ready/01_build_dataset.py
```

Les scripts des étapes 02, 05 et 06 demandent au démarrage le dossier cible : `pilot` ou
`production`. Les artefacts restent isolés dans le sous-dossier choisi.

Le troisième script prépare localement les lots Z.ai GLM 5.3 hébergés par Mistral, puis demande explicitement avant de les envoyer. Le mode `pilot` prépare deux lots de 50 requêtes ; le mode `production` prépare les trois splits. Il utilise \`MISTRAL_API_KEY\`, le modèle \`zai-glm-5-3\` et l'endpoint Batch Mistral. Chaque requête demande uniquement la phrase française : l'identifiant est porté par `custom_id` du Batch, pas par la réponse du modèle. Après l'envoi, le même script affiche le statut des derniers Batch et propose de récupérer les traductions lorsqu'ils sont terminés.

Si un lot est tronqué avant sa réponse finale, relancez-le explicitement avec :

\`\`\`powershell
python data/02_batch/02_translate_all.py
\`\`\`

Le quatrième script produit un fichier `05_checks/<dossier>/<split>_deterministic_checks.jsonl`. Il vérifie localement les champs obligatoires, les nombres, les pourcentages et l'absence apparente de SQL dans la traduction.

Le script `data/05_checks/02_judge_translations.py` remet en place le juge sémantique Batch avec `gpt-5.6-sol`. Il prépare deux lots, demande confirmation avant l'envoi, puis écrit les verdicts dans `05_checks/pilot/sol_judgments.jsonl`.
