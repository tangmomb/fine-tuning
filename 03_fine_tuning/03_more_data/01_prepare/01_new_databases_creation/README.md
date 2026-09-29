# Création des nouvelles bases fictives

Ce sous-pipeline crée les bases SQLite destinées aux expériences B/D. Elles sont
différentes des 146 bases Spider : un Batch demande au LLM de proposer 100
schémas et jeux de données cohérents, puis le script de matérialisation les
contrôle avant de produire le catalogue lu par `../prepare_seeds.py`.

## Étapes

Depuis la racine du dépôt :

```powershell
# 1. Écrire 100 requêtes Batch, sans appel réseau.
python 03_fine_tuning/03_more_data/01_prepare/01_new_databases_creation/01_create_requests.py

# 2. Soumettre seulement après revue du fichier créé (action facturée).
python 03_fine_tuning/03_more_data/01_prepare/01_new_databases_creation/01_create_requests.py --submit

# 3. Quand le Batch est terminé, récupérer sa réponse.
python 03_fine_tuning/03_more_data/01_prepare/01_new_databases_creation/02_download_batch.py

# 4. Construire les SQLite, valider leur structure et écrire le catalogue.
python 03_fine_tuning/03_more_data/01_prepare/01_new_databases_creation/03_materialize_databases.py

# 5. Répartir les 5 000 exemples entre ces bases.
python 03_fine_tuning/03_more_data/01_prepare/prepare_seeds.py --source new
```

Tous les artefacts restent dans ce dossier, organisés ainsi :

- `01_requests/` : JSONL brut envoyé au Batch ;
- `02_batch/` : identifiants et état du Batch ;
- `03_responses/` : JSONL brut reçu du Batch ;
- `04_final/` : SQLite validées, catalogue `new_databases.jsonl` et rapport.

Après des Batch de complément, `consolidate_artifacts.py --replace` peut
réduire les artefacts à un unique fichier de demandes et un unique fichier de
réponses, chacun avec les 100 bases finalement validées.

Par défaut, les 100 bases ont le même poids (`source_examples: 1`) : le plan
final attribue donc environ 50 exemples à chacune. Modifiez `--count` ou
`--source-examples` au besoin. `--overwrite` est requis pour remplacer tout
artefact ou fichier SQLite existant.
