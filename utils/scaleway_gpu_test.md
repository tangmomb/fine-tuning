# Lancer ponctuellement une GPU Scaleway pour un test

Ce guide correspond au projet Scaleway `fine_tuning_qwen`, en zone `pl-waw-2`.

## État de repos attendu

Pour éviter les ressources inutiles entre deux sessions :

- les instances `qwen-l4-benchmark` et `qwen-h100-finetuning` sont arrêtées ;
- aucune IP publique et aucun volume ne sont conservés ;
- le seul stockage persistant est le Block Snapshot
  `qwen-l4-root-shutdown-2026-09-21`
  (`9cfcea5c-8bab-46ba-847d-4a92798da98d`).

Un snapshot est une sauvegarde : il ne peut pas être monté directement. Pour
l'utiliser, créer un Block volume temporaire depuis ce snapshot, dans la même
zone (`pl-waw-2`).

## Avant de démarrer

1. Vérifier le stock GPU :

   ```powershell
   python utils/check_scaleway_gpu_availability.py --gpu H100 --zone pl-waw-2
   # ou
   python utils/check_scaleway_gpu_availability.py --gpu L4 --zone pl-waw-2
   ```

   `low_stock` signifie que la création peut échouer ; relancer la vérification
   juste avant la création.

2. Choisir le GPU :

   | Usage | Choix |
   | --- | --- |
   | Test rapide ou QLoRA d'un 4B | `L4-1-24G` |
   | Fine-tuning QLoRA d'un 9B | `L40S` si disponible, sinon H100 |
   | Itérations rapides, contexte long ou gros batch | `H100-1-80G` |

3. Préparer le script de test et le dataset **avant** d'allumer la GPU. Le
   disque local est éphémère : ne pas y placer le seul exemplaire d'un dataset
   ou d'un checkpoint important.

## Créer une session GPU dans la console

Les deux instances actuellement présentes n'ont plus de disque de démarrage.
Pour une session propre, le plus simple est donc de créer une nouvelle instance
GPU avec une image Ubuntu, plutôt que de tenter de démarrer une instance sans
disque.

1. Dans Scaleway Console, choisir **Instances** puis **Create instance**.
2. Sélectionner la zone `pl-waw-2` et le type GPU choisi, par exemple
   `H100-1-80G`.
3. Choisir une image Ubuntu 24.04.
4. Créer le petit disque local de démarrage proposé par l'assistant. Il contient
   seulement l'OS de la session.
5. Ne pas réserver d'IP publique tant qu'un accès SSH n'est pas nécessaire.
   Si nécessaire, réserver une Flexible IP juste pour la session et la supprimer
   à la fin.
6. Démarrer l'instance.

### Restaurer les données du snapshot (optionnel)

Effectuer ceci uniquement si les anciens datasets, scripts ou checkpoints sont
nécessaires :

1. Ouvrir **Block Storage > Snapshots** et sélectionner
   `qwen-l4-root-shutdown-2026-09-21`.
2. Créer un volume depuis ce snapshot dans `pl-waw-2`, par exemple
   `qwen-session-data` (100 Go).
3. Attacher le volume à l'instance GPU arrêtée ou en cours de création.
4. Après démarrage, identifier le disque et le monter. Ne jamais supposer que
   son nom est toujours `/dev/sdb` : vérifier d'abord avec `lsblk -f`.

   ```bash
   lsblk -f
   sudo mkdir -p /mnt/qwen-data
   # Remplacer /dev/XXX par la partition constatée avec lsblk.
   sudo mount /dev/XXX /mnt/qwen-data
   ```

## Connexion et test GPU

Se connecter en SSH si une IP publique temporaire est attachée, puis lancer :

```bash
nvidia-smi
python3 --version
```

Test PyTorch minimal :

```bash
python3 - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA n'est pas disponible"
print("GPU :", torch.cuda.get_device_name(0))
print("CUDA :", torch.version.cuda)
x = torch.randn((4096, 4096), device="cuda")
y = x @ x
torch.cuda.synchronize()
print("Test CUDA OK ; résultat :", tuple(y.shape))
PY
```

Si PyTorch n'est pas installé, créer un environnement virtuel et installer une
version CUDA compatible avant ce test. Vérifier avec `nvidia-smi` que le driver
est bien visible avant d'installer des dépendances de fine-tuning.

## Fin de session : ordre de nettoyage

1. Copier vers un stockage persistant les checkpoints et résultats à conserver.
2. Arrêter l'instance.
3. Détacher puis supprimer le Block volume temporaire, s'il a été recréé depuis
   le snapshot.
4. Supprimer l'IP publique temporaire, si elle a été réservée.
5. Supprimer le disque local et l'instance si l'objectif est de ne conserver
   que le snapshot. Conserver une instance arrêtée avec un disque local peut
   laisser du stockage à gérer ; une instance supprimée est plus nette.

Ne supprimer le snapshot que lorsqu'une autre sauvegarde vérifiée existe : c'est
le seul état actuellement conservé des données de la précédente session.
