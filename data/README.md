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
python data/02_batch/pilot/01_pilot_translate_to_french.py
python data/05_checks/01_check_translations.py
```

Le troisième script prépare localement le pilote Z.ai GLM 5.3 hébergé par Mistral (deux lots de 50 requêtes), puis demande explicitement avant de l'envoyer. Il utilise \`MISTRAL_API_KEY\`, le modèle \`zai-glm-5-3\` et l'endpoint Batch Mistral. Après l'envoi, le même script affiche le statut des derniers Batch et propose de récupérer les traductions lorsqu'ils sont terminés.

Si un lot est tronqué avant sa réponse finale, relancez-le explicitement avec :

\`\`\`powershell
python data/02_batch/pilot/01_pilot_translate_to_french.py --resubmit
\`\`\`

Le quatrième script produit `05_checks/pilot/deterministic_checks.jsonl`. Il vérifie localement les champs obligatoires, les nombres, les pourcentages et l'absence apparente de SQL dans la traduction.
