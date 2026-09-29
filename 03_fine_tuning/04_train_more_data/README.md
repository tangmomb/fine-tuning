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
python 03_fine_tuning/03_more_data/04_export/build_sft.py --experiment A
```

## Lancement

Depuis la racine du dépôt, par exemple pour reproduire la configuration retenue (Qwen 3.5 2B, LoRA A, LR 1e-4, 3 époques) :

```powershell
python 03_fine_tuning/04_train_more_data/train.py --experiment A --models Qwen3.5-2B --epochs 3 --lora-preset A --learning-rate 1e-4
```

Le lanceur :

- crée `01_datasets/train_plus_A.jsonl` en concaténant le train initial et les exemples synthétiques inédits ;
- écrit le manifeste associé (volumes et empreintes SHA-256 des sources) ;
- délègue ensuite à `01_train_main/train.py`, avec le même entraînement LoRA, la même validation et les mêmes garde-fous pour le test ;
- sépare les runs dans `artifacts/A/`, `artifacts/B/`, `artifacts/C/` ou `artifacts/D/`.

Par défaut, le lancement utilise les **deux H100** de l'instance Scaleway `H100-2-80G`, via `torchrun`. Le batch par GPU est porté à 4 et l'accumulation de gradients passe automatiquement à 8 : le batch effectif demeure donc 64 (`2 GPU × 4 × 8`), comme dans le protocole mono-GPU historique (`1 × 2 × 32`). Cela réduit le nombre de micro-batches et exploite mieux les H100, tout en gardant des résultats comparables.

Le lancement sur deux GPU doit être non interactif : fournissez donc `--models`, `--epochs`, `--lora-preset` et `--learning-rate`, comme dans l'exemple ci-dessus. Cela évite que les deux processus attendent une saisie terminal différente.

Pour garder un lancement mono-GPU, ajoutez `--gpus 1`. Si vous fournissez explicitement `--gradient-accumulation-steps`, cette valeur est conservée.

Après chaque entraînement, consultez `artifacts/<exp>/<run>/logs/gpu_rank-0_telemetry.jsonl` et `gpu_rank-1_telemetry.jsonl`. Ils consignent le débit par fenêtre de logs ainsi que la mémoire allouée, réservée et de pic. Pour un résumé :

```powershell
python 03_fine_tuning/04_train_more_data/summarize_telemetry.py artifacts/A/<run>
```

Si le pic reste largement sous 80 Go, augmentez progressivement `--per-device-batch-size` (par exemple 6, puis 8) et ajustez `--gradient-accumulation-steps` pour conserver un batch effectif de 64.

Par défaut, un dataset fusionné existant est réutilisé afin de rendre un run reproductible. Utilisez `--rebuild-dataset` uniquement après avoir réexporté ou remplacé le dataset synthétique correspondant.

Toutes les options de `01_train_main/train.py` restent disponibles après `--experiment`, y compris `--evaluate-run` et `--evaluate-test`.
