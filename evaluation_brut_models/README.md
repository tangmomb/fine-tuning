# Évaluation des modèles

`run.py` est le point d'entrée unique : il demande l'environnement (machine
locale ou instance GPU Scaleway), les modèles et le mode. Il délègue la
sélection et l'orchestration à `assets/run_benchmark.py`, puis l'exécution d'un
run à `assets/run_evaluation.py`.
Les SQL gold ne sont jamais lus pendant la génération : ils sont utilisés
uniquement au moment du scoring.

Chaque exécution écrit dans `runs/<run-name>/` :

- `predictions.jsonl` : sortie brute et SQL extrait ;
- `few_shot_examples.jsonl` : démonstrations réellement employées ;
- `results.jsonl` : comparaison détaillée par exemple ;
- `metrics.json` : paramètres et métriques agrégées.

`metrics.json` contient aussi `total_execution_seconds`, la durée complète de
l'exécution : chargement des données et du modèle, génération, scoring et
écriture des artefacts (hors écriture finale de `metrics.json`, négligeable).
Les champs `mean_batch_latency_ms` et `p95_batch_latency_ms` mesurent la
latence d'un batch, pas celle d'une requête isolée. La taille de batch associée
est indiquée par `batch_size`.
Les métriques de performance comprennent `generated_output_tokens`,
`generation_examples_per_second` et `generation_output_tokens_per_second`, qui
ne couvrent que `model.generate()`. Les champs
`end_to_end_examples_per_second` et `end_to_end_output_tokens_per_second`
utilisent `total_execution_seconds` et incluent donc le chargement, la
préparation, le scoring et les écritures. L'utilisation GPU est échantillonnée pendant la
génération via NVML et enregistrée sous `mean_gpu_utilization_percent` et
`peak_gpu_utilization_percent`; ces deux champs valent `null` si NVML n'est pas
disponible. Si le module NVML Python manque mais que `nvidia-smi` est installé,
le script l'utilise automatiquement comme repli, avec un échantillon par
seconde.

Le score principal est `execution_accuracy` : les résultats SQLite de la
prédiction et du SQL gold doivent être identiques. `exact_match` est une mesure
secondaire, basée sur une normalisation des espaces et de la casse. Les requêtes
sont exécutées en lecture seule contre `test_database`.

Le lanceur interactif demande si le thinking du modèle doit être autorisé. Le
défaut est non, pour garder une baseline directe comparable. Ce choix est
enregistré dans `metrics.json` sous `thinking_enabled`; l'option non interactive
est `--thinking` (ou `--no-thinking`).
Le plafond de génération est désormais de 1 024 tokens sans thinking et de
2 048 tokens avec thinking, afin d'éviter de tronquer des requêtes SQL longues.
`--max-new-tokens` permet de remplacer ces valeurs pour un run donné.

Pour choisir une taille de batch, l'utilitaire suivant reproduit les prompts et
les batches du benchmark puis calcule le gaspillage de padding :

```powershell
.\.venv\Scripts\python.exe utils\measure_padding_waste.py --model models\Qwen3.5-2B --batch-sizes 1 2 4 8 16 32
```

Il ne charge que le tokenizer, pas le modèle en VRAM. Le `padding_waste` global
est `1 - sum(longueurs) / sum(taille_batch × longueur_max_du_batch)`.

## Baseline comparable avant fine-tuning

Lancez `run.py`. Il demande l'environnement, les modèles disponibles dans cet
environnement et le mode à lancer. Il impose les mêmes paramètres pour les runs
choisis :
test complet, prompt contenu dans `test_inputs.jsonl`, schéma, température 0,
limite de 256 tokens, et les mêmes quatre démonstrations (seed 42). Elles sont
prélèvées en lecture seule depuis `data/06_training_dataset/03_production/train.jsonl` :
elles servent uniquement de contexte au prompt et ne modifient jamais le modèle.
Le fichier `few_shot_examples.jsonl` de chaque run permet de vérifier ce point.
Il demande aussi la taille de lot : 2 est proposée par défaut sur la RTX 2080
locale, et 1 sur Scaleway tant que la VRAM de l'instance n'est pas connue.

```powershell
.\.venv\Scripts\python.exe evaluation_brut_models\run.py
```

Pour une exécution non interactive :

```powershell
.\.venv\Scripts\python.exe evaluation_brut_models\run.py --environment scaleway --models models\Qwen3.5-4B --mode both
```

Ajoutez `--batch-size 2` pour ne pas répondre à la question interactive ; la
taille est enregistrée dans `metrics.json`. Pour un essai court, ajoutez
`--limit 10`; ne l'utilisez pas pour communiquer la baseline finale.

Avant le lancement sur Scaleway, copiez le dépôt, les checkpoints 4B/9B et les
dépendances d'inférence (`torch`, `transformers`) sur l'instance GPU. Les
résultats créés sur l'instance doivent ensuite être récupérés avec le dossier
`runs/` pour les comparer à ceux produits localement.

## Prérequis local

La machine locale détectée dispose d'une RTX 2080 (8 Go). Elle peut viser les
checkpoints 0.8B et 2B en FP16 ; les 4B et 9B restent destinés à Scaleway. Il
faut installer une version CUDA de PyTorch, Transformers et Accelerate dans le
venv. Le paquet PyTorch fournit le runtime CUDA : le CUDA Toolkit complet n'est
pas requis lorsque le pilote NVIDIA est déjà installé.

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install torchvision pillow transformers accelerate
```

Vérification après installation :

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
