# Qwen 3.5 checkpoints

The four official checkpoints used for the text-to-SQL comparison live here:

| Local directory | Hugging Face repository | Parameters |
|---|---|---:|
| `Qwen3.5-0.8B` | `Qwen/Qwen3.5-0.8B` | 0.8B |
| `Qwen3.5-2B` | `Qwen/Qwen3.5-2B` | 2B |
| `Qwen3.5-4B` | `Qwen/Qwen3.5-4B` | 4B |
| `Qwen3.5-9B` | `Qwen/Qwen3.5-9B` | 9B |

Qwen publishes this generation as a single checkpoint per size, rather than a separate `-Instruct` checkpoint. Use its chat template in non-thinking/instruct mode:

```python
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
)
```

Run `./models/download_qwen35_instruct.ps1` from the project root to resume or fetch the checkpoints. It places all Hugging Face and Xet cache data in `./.hf`, within this repository.

## Contenu de chaque modèle

Les quatre dossiers ont la même structure. Seul le nombre de fichiers de poids varie avec la taille du modèle : 1 shard pour 0.8B et 2B, 2 shards pour 4B, 4 shards pour 9B.

| Fichier | Rôle |
|---|---|
| `model.safetensors-*-of-*.safetensors` | Les poids du réseau de neurones : ce sont les gros fichiers indispensables pour exécuter le modèle. Le format `safetensors` permet un chargement sûr, sans code exécutable intégré. |
| `model.safetensors.index.json` | La table de correspondance qui indique dans quel shard se trouve chaque tenseur du modèle. Nécessaire dès que les poids sont découpés en plusieurs fichiers. |
| `config.json` | L’architecture : nombre de couches, dimensions, taille du vocabulaire, identifiants de tokens, etc. Transformers l’utilise pour construire le bon modèle avant de charger ses poids. |
| `tokenizer.json`, `vocab.json`, `merges.txt` | Le tokenizer. Il transforme le texte en identifiants numériques compris par le modèle, puis reconvertit les identifiants générés en texte. |
| `tokenizer_config.json` | Les réglages du tokenizer : tokens spéciaux, longueurs maximales et comportement de chargement. |
| `chat_template.jinja` | Le format exact des messages `system`, `user` et `assistant`. C’est ici que le paramètre `enable_thinking` décide entre raisonnement actif et réponse directe. |
| `preprocessor_config.json`, `video_preprocessor_config.json` | Réglages de préparation des entrées multimodales (images et vidéo). Ils ne sont pas nécessaires pour une requête text-to-SQL uniquement textuelle, mais font partie du checkpoint officiel. |
| `README.md` | Carte officielle du modèle : usage, limitations et exemples. |
| `LICENSE` | Licence d’utilisation du modèle. |
| `.gitattributes` | Indications Git/Hugging Face, notamment pour les gros fichiers stockés avec LFS/Xet sur le Hub. |
| `.cache/huggingface/` | Métadonnées locales de Hugging Face servant à reprendre et valider les téléchargements. Ce n’est pas le modèle et ce dossier reste ignoré par Git car il est à l’intérieur du dossier `Qwen3.5-*`. |
