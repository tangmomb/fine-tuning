# Données synthétiques diversifiées (v2)

Remplace la génération par gabarits de `03_03_more_data` (voir le diagnostic dans
`03_05_diagnostic/README.md`). Les gabarits uniques par difficulté produisaient
40 à 71 squelettes SQL pour 5 000 exemples, et aucun JOIN, ORDER BY ou LIMIT.

## Principe

Chaque prompt montre **16 exemples réels du train Spider** (questions françaises, autres
bases que la base cible), tirés proportionnellement à Spider et sans squelette SQL en
double. Le modèle écrit, pour chaque exemple, une paire sur la base cible qui reprend
**la même structure SQL** avec d'autres tables, colonnes et valeurs. Le prompt
contient aussi **3 lignes réelles par table**, pour que les valeurs littérales existent.

Différences avec l'ancien prompt :

- aucun gabarit imposé : la distribution des structures suit celle de Spider ;
- UNION, INTERSECT, EXCEPT, LIKE, BETWEEN et les sous-requêtes comparées sont autorisés (le parseur Spider les accepte) ;
- alias de tables au format Spider (`AS T1`) ;
- questions d'utilisateur, sans paraphrase littérale du SQL ;
- rejet des tautologies et des requêtes au résultat vide.

## Scripts

```bash
# Lot pilote (6 bases existantes + 6 nouvelles, 192 paires), exécuté directement
python create_requests.py --pilot --output pilot_v2 --run-sync
python check_pilot.py --input pilot_v2

# Requêtes au format Batch pour un vrai dataset (non soumises)
python create_requests.py --bases existing --prompts-per-base 3 --output batch_existing
python create_requests.py --bases new --prompts-per-base 4 --hardness hard extra --output batch_new_hard
```

`--hardness` restreint les exemples Spider à certaines difficultés (équivalent de C/D).

## Résultats du pilote v2 (2026-10-11, gpt-5.6-luna, environ 65 000 tokens)

191 paires reçues : **88,5 % valides**, 10,5 % de résultats vides, 1 % d'erreurs.
157 des 169 paires valides reprennent la structure de leur exemple.

| À taille égale (n = 169) | Spider train | Ancien A | Pilote v2 |
|---|---|---|---|
| Squelettes distincts / 100 ex. | 68,0 | 16,0 | **68,6** |
| JOIN | 42,0 % | 1,8 % | 52,7 % |
| 2+ JOIN | 20,1 % | 0,6 % | 17,8 % |
| ORDER BY | 21,3 % | 0 % | 23,7 % |
| LIMIT | 12,4 % | 0 % | 15,4 % |
| GROUP BY | 21,9 % | 59,8 % | 28,4 % |
| HAVING | 3,0 % | 0 % | 8,9 % |
| UNION/INTERSECT/EXCEPT | 6,5 % | 0 % | 6,5 % |

Difficulté Spider des paires valides : easy 37, medium 62, hard 30, extra 40, proche
des exemples ciblés. Le pilote v1 (8 exemples × 2 paires, structures toutes
différentes) donnait 46,8 squelettes pour 100 exemples et sur-représentait HAVING.

## Pipeline complet (dataset A2 : 146 bases existantes, tous niveaux)

```bash
python create_requests.py --bases existing --total-prompts 400 --output A_v2 --submit
python create_requests.py --output A_v2 --download          # jusqu'à « completed »
python check_pilot.py --input A_v2                           # validated.jsonl
python semantic_judge.py --input A_v2 --prepare
python semantic_judge.py --input A_v2 --submit
python semantic_judge.py --input A_v2 --collect              # judged.jsonl
python export_sft.py --input A_v2 --name A2                  # 03_03_04_export/synthetic_data_A2.jsonl
python ../03_04_train_more_data/train.py --experiment A2 --seed 42
```

## Juge sémantique v2

- Juge **toutes** les paires valides ; l'export choisit ensuite 5 000 paires acceptées au
  prorata des bases. Il n'y a pas de boucle de réécriture : l'ancienne boucle réécrivait
  les questions jusqu'à ce qu'elles décrivent le SQL mot à mot, ce qui les rendait littérales.
- Critère : « un expert écrirait-il une requête au résultat identique ? ». Les conventions
  Spider (jointure interne pour « chaque », regroupement par nom, doublons sans DISTINCT)
  sont acceptées explicitement : sans cela, Terra rejetait 24 % du pilote, souvent sur ces
  conventions. Avec : 86 % de pass, et les 23 rejets examinés sont tous de vraies erreurs
  (contrainte ajoutée, mauvaise colonne, LIMIT 10 pour « le plus »…).
- Coût mesuré sur le pilote : environ 190 tokens d'entrée et 45 de sortie par paire jugée.

## Ensuite

Entraîner baseline et A2 avec 2 graines chacun, puis comparer avec
`03_05_diagnostic/rescore_validation.py` (comparaison façon Spider, test apparié).
