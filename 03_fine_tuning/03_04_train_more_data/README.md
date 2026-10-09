# Entraînement avec données supplémentaires

Ce dossier lance les quatre expériences définies au chapitre 06 du compte rendu :

| Expérience | Jeu utilisé |
| --- | --- |
| A | train Spider initial + `synthetic_data_A.jsonl` |
| B | train Spider initial + `synthetic_data_B.jsonl` |
| C | train Spider initial + `synthetic_data_C.jsonl` |
| D | train Spider initial + `synthetic_data_D.jsonl` |

Les expériences restent volontairement séparées : elles servent à comparer l'apport de chaque nature de données, pas à entraîner sur A+B+C+D simultanément.

## Pré-requis

Exporter le dataset validé de l'expérience concernée depuis `03_more_data` :

```powershell
python 03_fine_tuning/03_03_more_data/03_03_04_export/build_sft.py --experiment A
```

## Lancement

Depuis la racine du dépôt, le lanceur demande uniquement l'expérience. La configuration est fixe pour que les résultats A à D soient comparables : **Qwen 3.5 2B · LoRA A (`r=8`, `alpha=16`) · learning rate `1e-4` · 3 époques · 1 H100 80 Go · batch local 2 · accumulation 32 · batch effectif 64**.

```powershell
python 03_fine_tuning/03_04_train_more_data/train.py
```

Pour un lancement non interactif reproductible :

```powershell
python 03_fine_tuning/03_04_train_more_data/train.py --experiment A
```

Le lanceur :

- crée `01_datasets/train_plus_A.jsonl` en concaténant le train initial et les exemples synthétiques inédits ;
- écrit le manifeste associé (volumes et empreintes SHA-256 des sources) ;
- délègue ensuite à `01_train_main/train.py`, avec le même entraînement LoRA, la même validation et les mêmes garde-fous pour le test ;
- regroupe les expériences lancées le même jour dans `03_fine_tuning/artifacts/data-scaling-new-data-YYYYMMDD/`, puis les sépare dans les sous-dossiers `A/`, `B/`, `C/` et `D/`.

Le lancement utilise exclusivement **une H100 80 Go**, via `torchrun`. Le batch local est fixé à 2 et l'accumulation de gradients à 32 : le batch effectif demeure 64 (`1 GPU × 2 × 32`). C'est le profil historique validé sur les séquences les plus longues ; il privilégie la fiabilité mémoire tout en préservant un batch effectif comparable.

La question est posée **avant** le démarrage de `torchrun`. Le processus GPU reçoit ensuite les options déjà résolues et ne lit jamais le terminal pendant l'entraînement.

Le nombre de GPU, le batch local et l'accumulation sont verrouillés par le protocole afin de conserver des comparaisons A à D cohérentes.

Après chaque entraînement, consultez `artifacts/<exp>/<run>/logs/gpu_rank-0_telemetry.jsonl` et `gpu_rank-1_telemetry.jsonl`. Ils consignent le débit par fenêtre de logs ainsi que la mémoire allouée, réservée et de pic. Pour un résumé :

```powershell
python 03_fine_tuning/03_04_train_more_data/summarize_telemetry.py 03_fine_tuning/artifacts/data-scaling-new-data-YYYYMMDD/A/<run>
```

Si le pic reste largement sous 80 Go, augmentez progressivement `--per-device-batch-size` (par exemple 6, puis 8) et ajustez `--gradient-accumulation-steps` pour conserver un batch effectif de 64.

Par défaut, un dataset fusionné existant est réutilisé afin de rendre un run reproductible. Utilisez `--rebuild-dataset` uniquement après avoir réexporté ou remplacé le dataset synthétique correspondant.

Toutes les options de `01_train_main/train.py` restent disponibles après `--experiment`, y compris `--evaluate-run` et `--evaluate-test`.
