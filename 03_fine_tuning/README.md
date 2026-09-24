# Fine-tuning Qwen Text-to-SQL — H100

Ce dossier entraîne un adaptateur LoRA récupérable sur `01_data/06_training_dataset/03_production/train.jsonl` et valide chaque époque sur `01_data/06_training_dataset/03_production/validation.jsonl` (1 034 exemples, split `dev`). Le jeu `01_data/07_evaluation_dataset` n'est jamais utilisé pendant l'entraînement.

Les chiffres de `02_evaluation_brut_models/interface/index.html` placent Qwen 9B (63,81 % zero-shot) parmi les meilleurs candidats Qwen. `train.py` accepte une ou plusieurs tailles : 0.8B, 2B, 4B et 9B.

## Configuration appliquée

| Paramètre | Valeur |
| --- | --- |
| Précision | BF16 (TF32 activé pour les multiplications) ; les rares paramètres de stabilité publiés en FP32 des 0.8B/2B restent en FP32 |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05`, `target_modules=all-linear` |
| Entraînement | 3 epochs, AdamW fused, LR `1e-4`, warmup `0.03`, cosine |
| Validation | à la fin de chaque époque ; un adaptateur est archivé pour chaque époque et le plus faible `eval_loss` est indiqué comme meilleur checkpoint |
| Stabilité | gradient clipping `1.0` |
| Séquence | 4 096 tokens |
| Batch effectif | 64 = micro-batch 2 × accumulation 32, mono-H100 |

Si la H100 est une 80 Go peu chargée, `--per-device-batch-size 4 --gradient-accumulation-steps 16` maintient le batch effectif 64 et accélère souvent le run. Réduisez à 2 048 seulement en cas de manque de mémoire : le schéma SQL rend 4 096 préférable.

## Exécution

Sur l'instance H100, après avoir copié le dépôt, les checkpoints dans `models/` et installé les dépendances du venv :

```bash
python 03_fine_tuning/train.py
```

Comme la H100 a déjà exécuté `02_evaluation_brut_models`, PyTorch CUDA, Transformers, Accelerate et Safetensors y sont déjà disponibles. Le seul ajout requis pour le LoRA est `peft` :

```bash
python -m pip install -r 03_fine_tuning/requirements.txt
```

Chaque entraînement crée un dossier autonome horodaté, par exemple
`03_fine_tuning/artifacts/qwen3.5-4b-20260924-211500Z/` :

```text
artifacts/qwen3.5-4b-<date_heure>/
├── checkpoints/epoch-1/   # adapter_config.json + adapter_model.safetensors
├── checkpoints/epoch-2/
├── checkpoints/epoch-3/
├── tokenizer/             # tokenizer et template de chat
├── training/              # training_args.bin + run_config.json
├── eval/epoch-1/          # métriques de validation de l'époque 1
├── eval/epoch-2/
├── eval/epoch-3/
├── eval/test_metrics.json # évaluation finale, après sélection du meilleur checkpoint
├── eval/test_predictions.jsonl
├── logs/training_log.jsonl
└── README.md
```

Le chemin du meilleur adaptateur est écrit dans `training/run_config.json` sous
`best_checkpoint`. Les artefacts contiennent uniquement les poids LoRA et doivent
être chargés avec le checkpoint de base Qwen correspondant ; ce ne sont pas des
copies fusionnées de plusieurs dizaines de Go des modèles de base.

Pour éviter la question interactive (par exemple dans `tmux`), indiquez directement les modèles :

```bash
python 03_fine_tuning/train.py --models Qwen3.5-4B Qwen3.5-9B
```

Avant la campagne finale, un smoke test peut se faire avec `python 03_fine_tuning/train.py --epochs 1`.
