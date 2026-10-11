# Diagnostic — pourquoi les datasets A à D n'améliorent pas la validation

## 1. Les écarts mesurés sont dans le bruit

`rescore_validation.py` ré-exécute toutes les prédictions de validation archivées.
L'évaluateur actuel compare les résultats en liste stricte : l'ordre des lignes
compte même sans `ORDER BY`, et l'ordre des colonnes aussi. Une comparaison façon
Spider (multiset si le gold n'a pas d'`ORDER BY`, colonnes permutables) ajoute
environ +3 points à tous les runs, sans changer le classement.

Comparaison appariée au baseline (r=8, α=16, lr 1e-4, epoch 3 : 69,83 % façon Spider) :

| Run | Δ | Gagnés / perdus | McNemar p | IC95 bootstrap par base |
|---|---|---|---|---|
| A | +2,22 | +78 / −55 | 0,06 | [+0,11 ; +4,03] |
| B | +0,48 | +57 / −52 | 0,70 | [−2,57 ; +3,12] |
| C | +1,06 | +65 / −54 | 0,36 | [−2,84 ; +4,63] |
| D | −0,29 | +53 / −56 | 0,85 | [−3,11 ; +2,14] |
| r=16 (contrôle) | +0,39 | +48 / −44 | 0,75 | [−2,57 ; +3,00] |

Un simple changement d'hyperparamètre fait déjà basculer environ 90 exemples.

## 2. Cause principale : les données synthétiques sont dégénérées

`03_03_02_batch/create_batch.py` (`HARDNESS_RECIPES`) impose **un seul gabarit SQL
par difficulté**, pour que le classifieur Spider accepte les requêtes :

| Niveau | Gabarit imposé |
|---|---|
| easy | `SELECT col FROM t [WHERE filtre]` |
| medium | `SELECT col, COUNT(*) FROM t GROUP BY col` |
| hard | `SELECT col FROM t WHERE cle IN (SELECT cle FROM t)` — tautologie sur la même table |
| extra | `SELECT col, COUNT(*) FROM t WHERE f1 AND f2 GROUP BY col` |

Conséquences mesurées :

| | Spider train | Spider val | A | B | C | D |
|---|---|---|---|---|---|---|
| Squelettes SQL distincts | — | — | 71 | 45 | 40 | 44 |
| JOIN | 44,0 % | 39,5 % | 1,2 % | 0 % | 0 % | 0 % |
| ORDER BY | 20,8 % | 22,9 % | 0 % | 0 % | 0 % | 0 % |
| LIMIT 1 | 12,1 % | 17,3 % | 0 % | 0 % | 0 % | 0 % |
| EXCEPT/INTERSECT/UNION | 6,1 % | 7,7 % | 0 % | 0 % | 0 % | 0 % |

Les niveaux « hard » et « extra » synthétiques le sont seulement au sens du
classifieur : ils ne contiennent aucune des opérations difficiles de Spider. C et
D n'apportent donc aucun exemple de vraie difficulté.

## 3. Ce que le modèle rate réellement (baseline, 312 erreurs)

- 40 % : structure et colonnes différentes (requête mal construite)
- 25 % : erreur d'exécution, presque toujours `no such column`, souvent une colonne attribuée à la mauvaise table dans une jointure
- 20 % : bonnes colonnes, mauvaise structure
- 8 % : bonne requête, mauvaise valeur littérale (`'T'`/`'F'`, `'L'`/`'R'`…)
- 7 % : même structure, mauvaise colonne

Une grande partie de ces erreurs porte sur des jointures multi-tables, absentes des données synthétiques.
