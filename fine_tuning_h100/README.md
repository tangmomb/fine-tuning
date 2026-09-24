# Fine-tuning Qwen Text-to-SQL — H100

Ce dossier entraîne un adaptateur LoRA récupérable sur `data/06_training_dataset/03_production/train.jsonl` et valide chaque époque sur `data/06_training_dataset/03_production/validation.jsonl` (1 034 exemples, split `dev`). Le jeu `data/07_evaluation_dataset` n'est jamais utilisé pendant l'entraînement.

Les chiffres de `evaluation_brut_models/interface/index.html` placent Qwen 9B (63,81 % zero-shot) parmi les meilleurs candidats Qwen. `train.py` accepte une ou plusieurs tailles : 0.8B, 2B, 4B et 9B.

## Configuration appliquée

| Paramètre | Valeur |
| --- | --- |
| Précision | BF16 (TF32 activé pour les multiplications) ; les rares paramètres de stabilité publiés en FP32 des 0.8B/2B restent en FP32 |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05`, `target_modules=all-linear` |
| Entraînement | 3 epochs, AdamW fused, LR `1e-4`, warmup `0.03`, cosine |
| Validation | à la fin de chaque époque ; le checkpoint avec la plus faible `eval_loss` est conservé |
| Stabilité | gradient clipping `1.0` |
| Séquence | 4 096 tokens |
| Batch effectif | 64 = micro-batch 2 × accumulation 32, mono-H100 |

Si la H100 est une 80 Go peu chargée, `--per-device-batch-size 4 --gradient-accumulation-steps 16` maintient le batch effectif 64 et accélère souvent le run. Réduisez à 2 048 seulement en cas de manque de mémoire : le schéma SQL rend 4 096 préférable.

## Exécution

Sur l'instance H100, après avoir copié le dépôt, les checkpoints dans `models/` et installé les dépendances du venv :

```bash
python fine_tuning_h100/train.py
```

Comme la H100 a déjà exécuté `evaluation_brut_models`, PyTorch CUDA, Transformers, Accelerate et Safetensors y sont déjà disponibles. Le seul ajout requis pour le LoRA est `peft` :

```bash
python -m pip install -r fine_tuning_h100/requirements.txt
```

Chaque adaptateur entraîné est écrit dans `fine_tuning_h100/artifacts/adapters/<modèle>/`. Récupérez le ou les dossiers entiers : ils contiennent les poids LoRA (`adapter_model.safetensors`), leur configuration, le tokenizer et `run_config.json`. Ils doivent ensuite être chargés avec les checkpoints de base Qwen correspondants ; ce ne sont pas des copies fusionnées de plusieurs dizaines de Go des modèles de base.

Pour éviter la question interactive (par exemple dans `tmux`), indiquez directement les modèles :

```bash
python fine_tuning_h100/train.py --models Qwen3.5-4B Qwen3.5-9B
```

Avant la campagne finale, un smoke test peut se faire avec `python fine_tuning_h100/train.py --epochs 1`.
