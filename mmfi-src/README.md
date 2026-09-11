# Pipeline JEPA temporel — nuages de points LiDAR (MM-Fi)

Implémentation du pré-entraînement auto-supervisé de type JEPA sur des séquences
de nuages de points LiDAR, développée et testée sur le dataset public MM-Fi
comme premier dataset (cf. `Tracking/mois/2-aout/`, notes 1 à 6, pour l'analyse
de littérature et les décisions de conception qui motivent chaque choix ci-dessous).

Ce document décrit **ce qui a été implémenté et vérifié**. Le *pourquoi* de
chaque choix de conception est dans les notes de suivi référencées ; ce README
ne le répète pas en détail, il y renvoie.

**Deux méthodes de tokenisation, pour comparaison** : grille (projection 2D +
conv 3D, cf. note 5) et points (FPS+kNN, décidée le 31 août). Les fichiers sont
organisés par **rôle**, pas par méthode — chaque fichier contient les deux
implémentations côte à côte plutôt que d'avoir deux fichiers quasi identiques
(consolidation faite le 31 août pour minimiser le nombre de fichiers ; 14→9).
`Encodeur`/`Predicteur`/`decouper_train_val` sont réellement partagés (un seul
code pour les deux méthodes) ; `tokenisation.py`/`masquage.py` regroupent deux
algorithmes différents dans un même fichier par commodité, sans code commun
entre eux.

## État actuel (4 septembre matin, résumé pour reprise rapide)

- **Architecture grille reconstruite pour mirroirer V-JEPA** (code source
  réel téléchargé dans `reference_vjepa/`, cf. section dédiée ci-dessous) :
  masquage par-échantillon à deux échelles simultanées, perte L1 (plus
  cosinus), scheduleurs LR/weight-decay/EMA, init façon DeiT/BEiT. La
  méthode **points est laissée intacte**, non mirroirée (portée décidée
  avec l'utilisateur).
- **Trouvaille clé** : le `reg_coeff: 0.0` de V-JEPA (aucune régularisation
  anti-effondrement explicite en production, juste EMA+stop-grad+masquage)
  **ne suffit pas sur notre donnée** — testé, effondrement quasi identique
  à avant tout correctif (~0,93-0,96 poolé). La régularisation d'uniformité
  reste donc nécessaire ; combinée au reste du mirroir V-JEPA, l'entraînement
  est sain (`sim_poolee` ~0,02-0,06 en fin d'entraînement).
- **Linear probing du mirroir** : sous-estime fortement la capacité du modèle
  (0,120/0,102 précision pré-entraîné/aléatoire) — cf. point suivant.
- **Fine-tuning complet (`geler_encodeur=False`) — la mesure qui compte** :
  les conditions montent à ~0,49-0,52 (4x le probing linéaire), mais
  **l'avantage du pré-entraînement JEPA n'est pas clairement démontré une
  fois l'encodeur dégelé**, vérifié sur 3 graines à chaque fois : MM-Fi+JRDB
  x1 (0,498±0,044) et MM-Fi+JRDB x2/volume doublé (0,517±0,015) sont tous
  deux dans le bruit de from-scratch (0,509±0,044) — écart-type ≥ écart entre
  moyennes. Observation secondaire à confirmer : doubler JRDB **resserre
  nettement la variance inter-graines** (±0,015 contre ±0,044) sans
  clairement déplacer la moyenne — hypothèse non établie (n=3 par condition).
  Voir [Fine-tuning en aval](#fine-tuning-en-aval-3-septembre-soir--despite)
  et [Dataset 3 — JRDB](#dataset-3--jrdb-corpus-domaine-proche-3-septembre-soir).
  Leçon retenue : ne jamais conclure sur une seule graine.
- **JRDB intégré** (27+27 séquences train+test, capteur Velodyne VLP-16, cf.
  section dédiée) — combiné à MM-Fi via projection `PROFIL_MMFI`, deux
  volumes testés (x1 : 25 séquences train ; x2 : ~50, réutilisant le split
  "test" de JRDB sans étiquette). Entraînements 3000 pas sains dans les deux
  cas.
- **ONCE supprimé par l'utilisateur** ("useless") — la section ONCE
  ci-dessous documente le travail passé mais le dataset n'est plus
  disponible localement ; ne pas planifier de nouveau run dessus sans
  re-téléchargement.
- **Points : problème non résolu**, méthode inchangée depuis avant la
  reconstruction V-JEPA — le pré-entraînement reste très en dessous de
  l'aléatoire (0,014 vs 0,181) en probing linéaire (fine-tuning jamais testé
  côté points). Hypothèse documentée (positional embedding absolu comme
  raccourci), pas testée — portée exclue de cette nuit sur décision
  utilisateur ("Drop points, mirror grille only").
- Voir [Linear probing — résultats](#linear-probing--résultats-3-septembre-après-perte_uniformite_poolee)
  pour le détail et [perte.py](#pertepy--perte-jepa-partagé) pour l'historique
  complet du diagnostic (effondrement → 2 correctifs testés → 1 qui marche).
- Checkpoints pré-correctif conservés dans `checkpoints_{grille,points}_pre_fix/`.
- **Piste bloquée (pas actionnable sans l'utilisateur)** : génération
  synthétique LiDAR à partir de maillages SMPL/AMASS (cf. LIPD,
  `Tracking/mois/3-septembre/1-despite_smpl_lidar_recherche.txt`) — prototype
  de ray-casting validé (`lidar_sim.py`), mais nécessite une inscription
  AMASS que je ne peux pas faire à la place de l'utilisateur.
- **PERCÉE (4 septembre matin) — deux nouvelles pistes HAR, toutes deux
  positives** : (1) une méthode classique (non deep learning, spectre du
  Laplacien d'un graphe de proximité + SVM) **bat largement notre meilleur
  résultat deep learning** — 0,617 contre 0,514 (single-run, à valider avec
  un split alternatif) ; (2) régression auxiliaire du squelette (signal
  jamais utilisé jusqu'ici) en perte conjointe avec la classification —
  0,549 ± 0,009 contre 0,509 ± 0,044 from-scratch, **vérifié sur 3 graines,
  le résultat le plus rigoureusement établi de la nuit** (écart-type 5x plus
  resserré que toute autre comparaison). Cf. section
  [Nouvelles pistes HAR](#nouvelles-pistes-har-4-septembre-matin--squelette-auxiliaire--spectre-laplacien)
  ci-dessous.
- **Suite spectre Laplacien (4 septembre après-midi)**, sur demande
  explicite ("push Laplacian spectral further") : réglage d'hyperparamètres
  (HistGradientBoosting, jamais essayé, devient le meilleur modèle réglé) +
  validation par split alternatif (protocole environnement du papier,
  résultats quasi identiques au split sujet — la mise en garde
  méthodologique de la nuit est levée) + fenêtrage doublé → **nouveau
  meilleur résultat du projet : 0,659 accuracy / 0,655 f1 macro** (contre
  0,617 la nuit précédente). Cf. section dédiée en fin de
  [Piste B](#piste-b--spectre-du-laplacien-classique-non-deep-learning--meilleur-résultat-absolu).
- **Recherche architectures + squelette-plafond + distillation (5 septembre)**,
  sur demande explicite ("keep pushing new methods, architectures... search
  github code from articles", 8h d'autonomie) : recherche multi-agents (8
  agents, 39 candidats catalogués avec vérification réelle du code, cf.
  `Tracking/mois/3-septembre/5-recherche_architectures_5sept.txt`) — la
  découverte la plus importante n'est PAS venue de la littérature mais du
  dépôt lui-même : un ST-GCN simplifié entraîné sur le squelette
  VÉRITÉ-TERRAIN (jamais utilisé comme entrée PRINCIPALE jusqu'ici, cf.
  `modele_stgcn.py`) atteint **0,994 ± 0,007** (3 graines, vérifié par un
  contrôle négatif label-aléatoire — cf. section dédiée
  [Nouvelles pistes HAR](#nouvelles-pistes-har-4-septembre-matin--squelette-auxiliaire--spectre-laplacien))
  contre 0,659 LiDAR-only. Motive une distillation squelette→LiDAR
  ("hallucination de modalité", Garcia/Morerio/Murino ECCV 2018) — enseignant
  squelette figé + étudiant LiDAR-seul, mécanisme différent de notre
  précédente tentative multi-tâche naïve. Sweep d'hyperparamètres
  prometteur sur une graine (0,537 à 800 pas) mais **le run complet 3
  graines (2000 pas, protocole identique) ramène cet optimisme à sa juste
  mesure : 0,523 ± 0,037, statistiquement INDISCERNABLE du from-scratch
  (0,509 ± 0,044) et pas meilleur que le multi-tâche naïf (0,549 ± 0,009,
  variance 4x plus resserrée)** — même verdict que toutes nos tentatives
  précédentes d'améliorer l'encodeur LiDAR par un signal d'entraînement
  plus riche (JEPA, multi-tâche, maintenant distillation) : le plafond
  semble être l'ARCHITECTURE encodeur elle-même (confirmé indépendamment
  par X-Fi, ICLR 2025, qui rapporte 52,7% LiDAR-seul sur MM-Fi avec son
  propre transformer dédié), pas le signal supervisé qui l'entraîne.
  RANSAC+retrait de bruit avant DBSCAN (piste bon marché suggérée par la
  recherche) **testé et invalidé** : effondre la précision (0,659→0,24) en
  mutilant la personne elle-même sur ce faible nombre de points, pas
  adopté. Cf. section
  [Recherche architectures + squelette-plafond + distillation](#recherche-architectures--squelette-plafond--distillation-5-septembre)
  pour le détail complet des 5 expériences de cette session.
- **PERCÉE (5 septembre après-midi) — transfer learning ShapeNet
  (Point-MAE figé) : 0,949, NOUVEAU MEILLEUR RÉSULTAT DU PROJET**, et un
  vrai résultat LiDAR-only (pas une modalité privilégiée). Tête linéaire
  apprise sur les features gelées : 0,934 ± 0,006 (split sujet) ; un SVM
  réglé sur les MÊMES features gelées fait encore mieux, sans ré-entraîner
  quoi que ce soit : **0,949 (split sujet) / 0,926 (split environnement)**.
  +0,29 sur l'ancien meilleur (0,659, spectre Laplacien), dépasse le 81-90%
  de la littérature de référence sur le split environnement comparable,
  vérifié anti-fuite. Contourne l'absence de CUDA Toolkit/MSVC sur cette
  machine (extensions CUDA des dépôts officiels non compilables) en
  téléchargeant uniquement le checkpoint officiel (poids ShapeNet-pretrained)
  et en réimplémentant l'encodeur en pur PyTorch par rétro-ingénierie du
  state_dict. Confirme directement l'hypothèse "pretraining sur une source
  plus grande/différente" de la toute première recherche (4 septembre) —
  cf. section dédiée
  [PERCÉE — transfer learning ShapeNet](#percée-5-septembre-après-midi--transfer-learning-shapenet-point-mae--0934)
  juste en dessous.
- **Suite (5 septembre soir) — 6 "add-ins" JEPA testés sur Point-MAE, 5
  échecs robustes + 1 gain non-JEPA** : sur demande explicite d'étendre la
  comparaison ("adding jepa architecture", puis "get a vit trained... on
  the web or training one yourself"). Trois variantes de continuation JEPA
  de l'encodeur (masquage aléatoire agressif/doux, masquage par bloc style
  V-JEPA) dégradent TOUTES le résultat de ~15-17 points, peu importe le
  réglage — conclusion robuste, pas un problème de tuning. Un agrégateur
  temporel appris par-dessus les features gelées fait pire encore (0,404).
  Le vrai checkpoint V-JEPA officiel (ViT-L/16, 305M, vidéo naturelle,
  téléchargé et chargé sans aucune clé manquante) transfère nettement moins
  bien que Point-MAE (0,444 contre 0,949) mais reste très au-dessus du
  hasard. **Seul gain trouvé : fusionner les features Point-MAE avec le
  spectre Laplacien (pas du JEPA) — nouveau meilleur résultat du projet,
  0,954 (split sujet) / 0,937 (split environnement).** Cf. section
  [Suite — "add-ins" JEPA sur Point-MAE](#suite-5-septembre-soir--add-ins-jepa-sur-point-mae--tous-négatifs-sauf-la-fusion).

## Nouvelles pistes HAR (4 septembre matin) — squelette auxiliaire + spectre Laplacien

Suite à la demande explicite de l'utilisateur ("find any solution for my har
project... try new architecture fitting our datasets sizes", 6h d'autonomie
complète) : recherche approfondie (workflow multi-agents, 6 rapports —
benchmarks MM-Fi publiés, distillation par information privilégiée,
architectures petits-échantillons, surveys LiDAR-HAR, apprentissage
multi-tâches) avant d'agir, cf. `Tracking/mois/4-septembre/` pour le détail.

**Découverte clé** : la littérature MM-Fi montre un écart ÉNORME entre notre
meilleur résultat (0,514, fine-tuning complet) et ce qui est publié comme
atteignable en classification directe depuis le LiDAR brut MM-Fi : **81-90%**
(arXiv:2509.23255, "LiDAR-based HAR through Laplacian Spectral Analysis",
2025 — spectre du Laplacien combinatoire d'un graphe de proximité + SVM/RF,
94,4% sur le sous-ensemble 13-classes réadaptation, 90,3% sur les 27
classes). Les propres chiffres de l'article original MM-Fi (NeurIPS 2023,
arXiv:2305.10345) — pipeline squelette-proxy AGCN/CTRGCN — donnent
seulement 54,44%/35,97% en LiDAR, ce qui explique pourquoi notre 0,51
semblait déjà "raisonnable" par rapport à cette référence-là : la vraie
référence à battre est la classification directe (81-90%), pas le pipeline
squelette-proxy de l'article original.

### Découverte annexe : le squelette 3D vérité-terrain n'a jamais été utilisé

MM-Fi fournit `ground_truth.npy` (T,17,3) par séquence — squelette 3D
COCO-17-joints, parfaitement aligné frame-à-frame avec le LiDAR (vérifié :
297 frames LiDAR == 297 frames squelette sur E01/S01/A01) — jamais exploité
jusqu'ici, seul le label d'action (1 valeur/séquence) servait de signal
supervisé. Tentative de recadrage du nuage LiDAR brut sur la bounding-box du
squelette (pour isoler la personne) : **0 points en commun** — le squelette
est dans un repère DIFFÉRENT du nuage LiDAR brut (calibration inter-capteurs
non fournie dans `MMFi_lidar_only`), donc PAS utilisable pour recadrer le
nuage de points sans transformation supplémentaire non résolue cette nuit.

### Piste A — Régression auxiliaire du squelette (multi-tâche, deep learning)

`squelette_dataset.py` (nouveau) : fenêtre LiDAR grille + squelette
sous-échantillonné (moyenne par paires de frames, même réduction temporelle
que le tubelet embedding) et CENTRÉ sur son propre centroïde par frame (pose
relative, pas position absolue — même mise en garde que pour la méthode
points, cf. section reconstruction V-JEPA). `modele_squelette.py` : encodeur
partagé (tokenizer + Transformer, mêmes briques que `ModeleJEPA`) + deux
têtes — classification (27 actions) et régression squelette (MLP par pas de
temps sur pooling mean+max spatial). `entrainer_squelette.py` : perte
conjointe CE + λ·MSE, from scratch, **from scratch uniquement** (le mirroir
JEPA n'apporte rien, cf. section fine-tuning — pas la peine d'y superposer
davantage de complexité).

Recherche (rapport "multitask_joint_training") : poids fixe recommandé
plutôt que pondération automatique (incertitude/GradNorm) — sweep manuel
λ∈{0,1 ; 1 ; 10}, 800 pas, 1 graine :

| λ | precision | f1 |
|---|---|---|
| 0,1 | 0,421 | 0,407 |
| **1,0** | **0,523** | **0,545** |
| 10,0 | 0,472 | 0,453 |

λ=1,0 retenu, run complet 2000 pas × 3 graines (même protocole que toutes
nos autres comparaisons de la nuit) :

| graine | precision | f1 |
|---|---|---|
| 0 | 0,546 | 0,547 |
| 1 | 0,560 | 0,552 |
| 2 | 0,542 | 0,577 |
| **moyenne ± écart-type** | **0,549 ± 0,009** | **0,559 ± 0,016** |

**Comparaison à la référence from-scratch (classification seule, 3 graines,
déjà mesurée la nuit précédente) : 0,509 ± 0,044.**

**C'est le résultat le plus SOLIDEMENT établi de la nuit** — contrairement à
toutes les comparaisons JEPA/JRDB précédentes où le bruit inter-graines
(±0,04-0,05) rendait toute amélioration apparente indiscernable du hasard,
ici la moyenne progresse (+0,040) ET l'écart-type se resserre nettement
(±0,009 contre ±0,044, un facteur ~5) — les 3 graines squelette (0,542 à
0,560) sont beaucoup plus cohérentes entre elles que n'importe quelle autre
comparaison multi-graines de la nuit. Le signal auxiliaire dense (squelette,
jamais utilisé jusqu'ici) semble stabiliser l'entraînement en plus
d'améliorer légèrement la moyenne — cohérent avec la littérature
(rapport "multitask_joint_training" : les tâches de régression auxiliaires
denses agissent comme régularisateur en régime de données limité).

### Piste B — Spectre du Laplacien (classique, non deep learning) — MEILLEUR RÉSULTAT ABSOLU

`laplacien_spectral.py` (nouveau) : réimplémentation simplifiée de
arXiv:2509.23255 (code source non disponible, "coming soon" au 4
septembre) à partir de la description méthodologique (WebFetch de
l'article) :
1. **Isolation de la personne** — le nuage LiDAR brut MM-Fi n'est PAS
   recadré sur la personne (~1100-1200 points/frame, plusieurs mètres
   d'étendue, mur/sol inclus — découvert cette nuit). Le squelette ne
   pouvant pas servir de repère (cf. ci-dessus), isolation auto-contenue :
   soustraction de fond (voxels 10cm occupés dans >80% des frames d'une
   séquence = statique = mur/sol) puis DBSCAN pour isoler le plus gros amas
   spatialement cohérent (la personne, en mouvement) — réduit à ~50-150
   points/frame, étendue verticale ~1,75m (cohérent avec une personne
   debout).
2. **Graphe epsilon** (arête si distance < 0,20m, valeur optimale du
   papier) → Laplacien combinatoire L=D-A.
3. **Descripteur de pose** — valeurs propres (K=30) ± statistiques de
   vecteurs propres (K=40, 7 stats chacun), par partie du corps.
4. **Agrégation temporelle** — 4 fenêtres/séquence, 8 frames
   sous-échantillonnées/fenêtre, statistiques (moyenne/écart-type/min/max).
5. **Classifieur** — SVM (rbf, C=1,0) et RandomForest, même split
   train/val que le reste du projet (sujets tenus à l'écart) — DIFFÉRENT du
   split "30 sujets E1-E3/10 sujets E4" du papier.

**Résultats (ablation, comme le papier) :**

| version | SVM | RF |
|---|---|---|
| v1 — corps entier seul | 0,218 | 0,205 |
| v2 — + 4 quadrants spatiaux | **0,617** | 0,601 |
| v3 — + statistiques vecteurs propres (6200 dims) | 0,590 | 0,611 |

**v1 confirme le papier dans le mauvais sens** — sans partition spatiale, le
signal est noyé (bien en dessous de notre 0,514 deep learning). **v2
confirme le papier dans le bon sens et de façon spectaculaire** — la
partition en 4 quadrants (l'ablation la plus impactante du papier, 61%→94%
sur leur 13-classes) est AUSSI l'ablation décisive ici : 0,218→0,617, qui
**bat notre meilleur résultat deep learning (0,514) de 10 points**, avec une
méthode entièrement classique (pas de GPU, extraction de features en
~25-30 min CPU, entraînement SVM en secondes).

**v3 NE confirme PAS le papier** — ajouter les statistiques de vecteurs
propres (deuxième levier du papier, +6pts sur leur 13-classes) ne donne
AUCUN gain net ici (SVM même légèrement pire : 0,590 vs 0,617 ; RF quasi
identique : 0,611 vs 0,601) malgré une dimension de features 50x plus
grande (6200 vs 120). Lecture honnête : probablement un sur-ajustement —
6200 dims pour 3456 échantillons d'entraînement, SVM/RF utilisés avec leurs
hyperparamètres par défaut (pas de recherche de grille), alors que le
papier a probablement optimisé les siens. **v2 (quadrants seuls) reste
notre meilleur résultat** — retenu comme référence pour ce projet.

**Prudence méthodologique à noter** : split train/val différent du papier
(sujets vs environnements), une seule extraction de features (pas de
répétition multi-graines côté SVM — déterministe à data fixée, mais
`RandomForestClassifier(random_state=0)` fixe la seule source de hasard
restante), fenêtrage simplifié (4 fenêtres/séquence sous-échantillonnées
plutôt que le stride dense du papier). Le résultat est fort mais mérite
d'être validé plus rigoureusement (répéter avec un split alternatif, plus
de fenêtres) avant de le considérer comme définitivement supérieur.

### Suite (4 septembre après-midi) — réglage d'hyperparamètres + validation par split alternatif : 0,659

Sur demande explicite de l'utilisateur ("push Laplacian spectral further"),
les deux prudences ci-dessus ont été levées — détail complet dans
`Tracking/mois/3-septembre/4-laplacien_tuning_validation.txt`.

**Réglage d'hyperparamètres** (`tuner_laplacien_v2.py`, GridSearchCV,
`GroupKFold` groupé par sujet — pas un K-Fold aveugle, qui surestimerait le
score vu que chaque séquence produit 4 lignes très corrélées) sur les
features v2 déjà extraites : SVM (0,617→**0,638**, C=3,0/gamma=scale) et
surtout **HistGradientBoosting** (jamais essayé jusqu'ici, **0,649**) —
RandomForest, lui, ne gagne rien au réglage (0,601→0,605, dans le bruit).

**Validation par split alternatif** : MM-Fi partitionne déjà ses 40 sujets
par environnement (10 sujets/environnement, aucun chevauchement — vérifié),
donc `decouper_train_val_environnement()` (nouveau, E01-E03 train / E04 val)
reproduit *exactement* le protocole du papier ("30 sujets E1-E3/10 sujets
E4"). Ré-extraction complète des 1080 séquences en une passe, **fenêtrage
doublé** (8 fenêtres/séquence au lieu de 4) et **parallélisée** (14
processus, `multiprocessing.Pool` — 1080 séquences en 5,5 min contre
25-30 min séquentiel pour moitié moins de fenêtres) :

| split | fenêtres | SVM défaut | SVM tune (C=3) | HGB tune |
|---|---|---|---|---|
| sujet (comme avant) | 4 (ancien) | 0,617 | — | — |
| sujet (comme avant) | 8 | 0,631 | **0,659** (f1=0,655) | 0,649 |
| environnement (papier) | 8 | 0,625 | 0,643 | **0,653** (f1=0,650) |

**Le split par environnement donne des résultats très proches du split par
sujet** (écart <0,02, bien plus petit que les gains de réglage/fenêtrage) —
**la mise en garde méthodologique précédente est levée** : 0,617 (et sa
version améliorée ~0,65) n'est pas une coïncidence du split sujet choisi.
**Nouveau meilleur résultat du projet, toutes méthodes confondues : 0,659
accuracy / 0,655 f1 macro** (SVM réglé, split sujet, 8 fenêtres) — +0,042
sur l'ancien meilleur (0,617), +0,11 sur le meilleur résultat deep learning
(0,549, squelette auxiliaire). Écart restant avec la littérature (81-90%)
toujours large — probablement des différences d'implémentation plus
profondes (isolation de la personne, ré-essai des statistiques de vecteurs
propres avec régularisation) plutôt qu'un simple réglage supplémentaire.

## PERCÉE (5 septembre après-midi) — transfer learning ShapeNet (Point-MAE) : 0,934

Suite à "run everything you need" (poursuite des deux pistes ouvertes de la
session précédente) : la piste transfer-learning ShapeNet-pretrained
(PointGST/Point-MAE, suitability=5 dans la recherche) était bloquée par
l'absence de CUDA Toolkit/MSVC sur cette machine (`nvcc`/`cl.exe`
introuvables — vérifié — donc impossible de compiler les extensions CUDA
des dépôts officiels, `pointnet2_ops`/`KNN_CUDA`/Chamfer distance ; installer
une chaîne CUDA Toolkit + Visual Studio Build Tools complète pour ce seul
test aurait été un changement système lourd, non fait sans consultation).
**Contournement** : téléchargement direct du checkpoint officiel Point-MAE
pré-entraîné sur ShapeNet (`pretrain.pth`, 332 Mo, poids seuls — AUCUN code
de leur dépôt exécuté) + réimplémentation PURE PYTORCH de l'encodeur
(`modele_pointmae.py`), architecture reconstruite en inspectant les CLÉS et
FORMES du state_dict (pas leur code source) : mini-PointNet local+global
(conv1d) + positional embedding + 12 blocs Transformer pré-normés
(dim=384, 6 têtes) + LayerNorm finale — chargement `strict=True` réussi,
confirmant la reconstruction exacte. Le FPS+kNN de groupement est
**réutilisé tel quel** de `tokenisation.py` (méthode points existante,
mêmes fonctions `fps`/`grouper_knn`, juste avec K=64/32 au lieu de 32/24) —
et a d'ailleurs révélé **deux bugs de padding réels** dans ce code partagé
(silencieusement invisibles tant qu'appelé avec les valeurs par défaut du
projet, cf. commentaires dans `tokenisation.py`), maintenant corrigés.

**Pipeline** : isolation de la personne (même soustraction-de-fond+DBSCAN
que le pipeline Laplacien) → groupement FPS+kNN (64 centres/32 voisins) →
encodeur Point-MAE **FIGÉ** (linear probing, recommandé par PointGST à
cette échelle plutôt que le fine-tuning complet) → pooling mean+max sur les
groupes (768 dims) PAR FRAME, précalculé une fois pour toutes les séquences
(`extraire_pointmae.py`, CPU parallèle + GPU par lots, ~7,5 min pour les
1080 séquences — entraînement "à la volée" testé d'abord : ~17s/pas, bien
trop lent pour un protocole 3 graines, cf. Tracking) → à l'entraînement,
juste un pooling mean+max TEMPOREL (8 frames) + une tête linéaire.

**Résultat, même protocole exact que toutes les autres comparaisons (2000
pas, lot=64, AdamW, graines 0/1/2) :**

| condition | précision |
|---|---|
| from-scratch LiDAR | 0,509 ± 0,044 |
| squelette auxiliaire (naïf) | 0,549 ± 0,009 |
| distillation squelette→LiDAR | 0,523 ± 0,037 |
| spectre Laplacien + SVM/HGB réglé | 0,659 |
| **Point-MAE (ShapeNet, figé) — tête linéaire, split sujet** | **0,934 ± 0,006** |
| **Point-MAE (ShapeNet, figé) — tête linéaire, split environnement (protocole papier)** | **0,908 ± 0,009** |
| squelette-plafond (vérité-terrain, pas LiDAR-only) | 0,994 ± 0,007 |

> ⚠️ **Chiffres split environnement à régénérer** — aucun script du dépôt ne
> les produisait avant la refactorisation du 11 septembre (cf. section
> dédiée). Le code corrigé les calcule désormais ; les valeurs ci-dessus
> viennent d'exécutions ad hoc non commitées.

**+0,275 sur l'ancien meilleur résultat LiDAR-only (0,659), dépasse même le
81-90% de la littérature de référence (arXiv:2509.23255) sur un protocole
comparable (split environnement), et approche le plafond squelette
vérité-terrain (0,994) — TOUT EN RESTANT UN RÉSULTAT LIDAR-ONLY LÉGITIME**
(l'encodeur ne voit jamais le squelette, seulement le nuage de points
isolé). **Vérifié anti-fuite** avec le même contrôle négatif que le
squelette-plafond (label aléatoire indépendant par fenêtre, train ET val
séparément) : précision s'effondre à 0,03-0,06 (≈1/27 attendu), confirmant
qu'aucune fuite n'explique le 0,934.

### Suite (5 septembre, "keep going") — un classifieur classique fait encore mieux : 0,949

Sur les MÊMES features gelées (pas de nouvel entraînement d'encodeur),
`tuner_pointmae_classique.py` essaie un SVM/RandomForest/HistGradientBoosting
à la place de la tête linéaire+pooling appris (mean+max temporel figé,
1536 dims, mêmes hyperparamètres SVM déjà réglés que pour le pipeline
Laplacien) :

| classifieur | split sujet | split environnement |
|---|---|---|
| tête linéaire (NN, 3 graines) | 0,934 ± 0,006 | 0,908 ± 0,009 |
| SVM (C=1,0 défaut) | 0,946 | — |
| RandomForest (défaut) | 0,846 | — |
| HistGradientBoosting (hyperparams Laplacien) | 0,910 | — |
| **SVM réglé (C=3,0, gamma=scale, GroupKFold sujet)** | **0,949** | **0,926** |

> ⚠️ **Chiffres split environnement à régénérer** — aucun script du dépôt ne
> les produisait avant la refactorisation du 11 septembre (cf. section
> dédiée). Le code corrigé les calcule désormais ; les valeurs ci-dessus
> viennent d'exécutions ad hoc non commitées.

Le SVM réglé (déterministe à features fixées, pas de graine à faire varier)
**dépasse légèrement la tête linéaire sur les deux splits** — nouveau
meilleur résultat du projet : **0,949 (split sujet) / 0,926 (split
environnement)**. RandomForest reste nettement en retrait (cohérent avec
son comportement sur le pipeline Laplacien). Confirme que le SIGNAL vient
des features Point-MAE elles-mêmes (le choix du classifieur en aval compte
peu une fois les features assez riches).

**Fine-tuning complet de l'encodeur testé et NON retenu** : nécessite un
pipeline séparé (`extraire_pointmae_groupes.py` + `entrainer_pointmae_finetune.py`
— les groupes bruts, pas encore encodés, sont précalculés une fois, mais
l'encodeur restant entraînable, son forward/backward doit être refait à
chaque pas, ~5,5s/pas, ~9h pour un protocole complet 3 graines). Sweep
préalable (1 graine, 800 pas) : trajectoire **bruitée et oscillante** (pic
à 0,934 puis redescend sans converger, résultat final 0,894) — **ne dépasse
ni la tête linéaire figée (0,934) ni le SVM (0,949)**, et l'instabilité
suggère un risque de sur-apprentissage à dégeler un Transformer 12 couches
sur seulement 864 séquences (même schéma que le fine-tuning JEPA complet,
qui effaçait déjà l'avantage du pré-entraînement — section dédiée
plus haut). Le signal du sweep étant déjà clair, le protocole complet 3
graines (~9h) n'a pas été lancé — cf.
`Tracking/mois/3-septembre/5-recherche_architectures_5sept.txt` pour le détail complet.

**Confirme directement l'hypothèse de la toute première recherche
multi-agents** (4 septembre : "pretraining ShapeNet/ModelNet40 utile
surtout quand le jeu de données cible est petit ET que la source de
pré-entraînement est plus grande/différente") — exactement ce qui manquait
à notre propre tentative JEPA (pré-entraînée ET fine-tunée sur le MÊME
petit corpus MM-Fi). Résout aussi le mystère du plateau transformer
(~0,49-0,55 partout : from-scratch, JEPA, X-Fi indépendant) : ce n'est pas
que les point-clouds LiDAR de MM-Fi manquent d'information exploitable,
c'est qu'un encodeur appris UNIQUEMENT sur ~864-1080 séquences n'a pas
assez de données pour apprendre de bonnes features géométriques — un
encodeur pré-entraîné sur ~50k objets ShapeNet, si, même transféré à un
domaine totalement différent (objets CAD rigides complets vs silhouette
humaine éparse).

**Prudence méthodologique** : seul le linear probing a été testé (pas le
fine-tuning complet de l'encodeur, par contrainte de temps ET parce que
PointGST recommande justement le probing à cette échelle) ; l'isolation de
la personne reste la même heuristique DBSCAN que le pipeline Laplacien
(mêmes limites documentées) ; seulement 4 fenêtres/séquence, 8 frames
échantillonnées/fenêtre (réduit pour la vitesse d'extraction, pas ré-essayé
avec plus de fenêtres vu la marge déjà large). Nouveau meilleur résultat du
projet, de loin — cf. `Tracking/mois/3-septembre/5-recherche_architectures_5sept.txt`
pour le détail complet.

## Suite (5 septembre soir) — "add-ins" JEPA sur Point-MAE : tous négatifs, sauf la fusion

Sur demande explicite ("try to get results adding jepa architecture to
current performing method" puis "get a vit trained... downloading a
pre-trained one... or training one by yourself") : six variantes testées,
toutes comparées à la référence Point-MAE+SVM (0,949/0,926). Détail complet
dans `Tracking/mois/3-septembre/5-recherche_architectures_5sept.txt`
(étapes 6 bis à 6 sextuple).

| add-in | sujet | environnement | verdict |
|---|---|---|---|
| Point-MAE + SVM (référence) | 0,949 | 0,926 | — |
| + JEPA continu (masquage aléatoire, LR=1e-4) | 0,793 | 0,763 | échec |
| + JEPA continu, plus doux (LR=1e-5, 10x plus faible) | 0,795 | 0,776 | échec, même ampleur |
| + JEPA continu, masquage par bloc (style V-JEPA) | 0,789 | 0,760 | échec, même ampleur |
| + JEPA temporel (agrégateur appris sur les 8 frames) | 0,404 | — | échec net |
| Autre checkpoint : Point-MAE fine-tuné ModelNet40 | 0,940 | 0,905 | neutre/légèrement pire |
| **V-JEPA officiel réel (ViT-L/16, 305M, vidéo naturelle)** | 0,444 | 0,333 | échec, mais >> hasard |
| **Fusion Point-MAE + Laplacien** | **0,954** | **0,937** | **seul gain, nouveau meilleur** |

> ⚠️ **Chiffres split environnement à régénérer** — aucun script du dépôt ne
> les produisait avant la refactorisation du 11 septembre (cf. section
> dédiée). Le code corrigé les calcule désormais ; les valeurs ci-dessus
> viennent d'exécutions ad hoc non commitées.

**JEPA continu sur l'encodeur (3 variantes : masquage aléatoire agressif,
masquage aléatoire doux, masquage par bloc façon V-JEPA/Point-MAE original)** :
`modele_pointmae_jepa.py`/`entrainer_pointmae_jepa.py` — repart des poids
ShapeNet (pas d'un init aléatoire, contrairement à l'échec JEPA initial),
continue l'adaptation par masquage de groupes + prédiction en espace
d'embedding (EMA + `perte_uniformite_poolee` activée dès le départ,
leçon retenue). Entraînement toujours SAIN (aucun effondrement, `sim_poolee`
0,01-0,15) mais **les trois variantes dégradent le résultat de ~15-17 points,
peu importe le taux d'apprentissage ou le style de masquage** — conclusion
robuste : ce n'est ni un problème de réglage ni de style de masquage, la
continuation auto-supervisée sur seulement ~900 séquences efface les
features ShapeNet utiles plus vite qu'elle n'apprend quoi que ce soit
d'utile sur MM-Fi.

**JEPA temporel** (`entrainer_temporal_jepa.py`) : au lieu de ré-adapter
l'encodeur spatial, garde les features Point-MAE gelées et ajoute un petit
Transformer contexte/cible/prédicteur PAR-DESSUS (remplace le simple
pooling mean+max par un agrégateur temporel appris, auto-supervisé sur les
8 frames/fenêtre). Résultat net : 0,404, bien pire que le pooling fixe —
un agrégateur appris from scratch sur 864 séquences n'apporte rien face à
un pooling non-appris déjà excellent.

**V-JEPA officiel réel** (`extraire_vjepa_features.py`) : checkpoint
Meta AI téléchargé directement (`vitl16.pth.tar`, 5,1 Go, ViT-L/16 305M
paramètres, pré-entraîné vidéo naturelle Kinetics-710/SSv2/HowTo100M) —
code source RÉEL de `reference_vjepa/` réutilisé tel quel (pur PyTorch,
aucune extension CUDA, contrairement à Point-MAE), chargé avec 0 clé
manquante/inattendue. Appliqué à la méthode GRILLE (structure de clip
vidéo native) : 3 canaux (x,y,z, canal de validité retiré) + standardisation
par-fenêtre, `interpolate_pos_encoding` de leur propre code gère
automatiquement notre grille 4×6×8 (192 tokens) au lieu de leur 14×14×8
natif (1568), aucune modification d'architecture nécessaire. Résultat :
0,444/0,333 — **très en dessous de Point-MAE mais nettement au-dessus du
hasard** : un vrai transfert existe, mais bien plus faible que le transfert
point-cloud→point-cloud (ShapeNet→MM-Fi). Confirme que la NATURE du domaine
source (point cloud 3D vs vidéo RGB naturelle) compte plus que la taille ou
la qualité du pré-entraînement.

**Fusion Point-MAE + Laplacien** (`tuner_fusion.py`) : concatène simplement
les features gelées Point-MAE et les valeurs propres du spectre Laplacien
(moyennées par séquence pour aligner les deux caches, nombres de fenêtres
différents) puis SVM réglé. **Seul add-in qui améliore la référence** :
0,954/0,937 (contre 0,949/0,926) — gain modeste mais présent sur les DEUX
splits. Pas du JEPA, mais répond à la demande de "comparaison plus large" :
les deux familles de features (géométrie spectrale classique + features
apprises génériques) sont légèrement complémentaires.

**Conclusion generale de cette manche** : ajouter du JEPA (sous quelque
forme testée) au pipeline Point-MAE ne marche pas, de façon robuste
(3 variantes de continuation, 1 agrégateur temporel, 1 vrai ViT vidéo
téléchargé — 6 tentatives, 6 échecs relatifs à la référence). Le seul gain
vient d'une fusion de features non-JEPA. Nouveau meilleur résultat du
projet : **0,954 (split sujet) / 0,937 (split environnement)**.

## Recherche architectures + squelette-plafond + distillation (5 septembre)

Suite à la demande explicite de l'utilisateur ("keep pushing new methods,
architectures... search github code from articles", 8h d'autonomie) —
détail complet dans
`Tracking/mois/3-septembre/5-recherche_architectures_5sept.txt`.

### v3 (statistiques de vecteurs propres) + régularisation PCA — n'apporte rien de plus

`tuner_laplacien_v3_pca.py` : PCA avant SVM/HGB sur les features v3 (6200
dims) déjà extraites, pour vérifier si l'absence de gain de v3 (README,
section Piste B) venait d'un sur-apprentissage (hypothèse non testée à
l'époque) plutôt que d'une absence de signal.

| variante | meilleur n_comp | SVM VAL |
|---|---|---|
| PCA(v3 seul) | 400 | 0,626 (vs 0,590 sans PCA) |
| PCA(concat[v2,v3]) | 100 | 0,641 |

La PCA régularise bien v3 seul (0,590→0,626) mais **ne dépasse pas v2 seul**
(0,638-0,649 réglé) même combinée — les statistiques de vecteurs propres
n'apportent pas de signal net supplémentaire avec NOTRE isolation de la
personne. Piste fermée.

### Recherche multi-agents (8 agents, 39 candidats, ~800k tokens)

7 agents en parallèle (SOTA MM-Fi, suites Laplacien, architectures
petit-échantillon, GNN squelette, LiDAR-HAR récent, distillation LUPI,
variantes transformer) + 1 agent de synthèse, avec vérification RÉELLE du
code (listing de fichiers via API GitHub, pas seulement lecture du README).

**Découverte clé** (lecture du texte complet d'arXiv:2509.23255, pas
seulement l'abstract) : le papier de référence (81-90%) isole la personne
avec un **détecteur 3D supervisé (PV-RCNN via OpenPCDet)**, pas une
heuristique comme notre soustraction-de-fond+DBSCAN — explique
plausiblement une bonne partie de l'écart restant. Repo du papier toujours
vide (vérifié à nouveau, 0 activité depuis fin septembre 2025 — accepté
entre-temps à ICASSP 2026).

**Confirmation indépendante utile** : X-Fi (ICLR 2025, arXiv:2410.10167,
code vérifié réel) rapporte **52,7% en LiDAR-seul sur MM-Fi** avec son
propre transformer point-cloud dédié — cohérent avec notre plateau
transformer (~0,49-0,55 toutes variantes confondues : from-scratch, JEPA,
multi-tâche), signe que ce plafond est une propriété du TYPE d'architecture
sur ce type de données, pas un défaut de notre implémentation.

L'agent de synthèse a lui-même découvert, en lisant `mmfi-src/` directement
(pas depuis la littérature), que le ST-GCN squelette-plafond était déjà en
cours d'écriture en parallèle cette nuit — jugeant la plupart des
architectures GNN squelette plus sophistiquées de la littérature (CTR-GCN,
InfoGCN, Hyperformer, SkateFormer, toutes avec du code réel vérifié)
redondantes une fois ce plafond mesuré par un modèle plus simple.

### ST-GCN "squelette-plafond" — 0,994 ± 0,007, vérifié anti-fuite

`modele_stgcn.py` (nouveau) : ST-GCN simplifié (Yan et al., AAAI 2018) — une
seule matrice d'adjacence normalisée + pondération d'arête apprise (edge
importance weighting, conservé de l'article original), **pas** le
partitionnement à 3 sous-ensembles de l'article (simplification documentée,
justifiée par la taille de notre graphe : 17 nœuds contre 25 pour NTU-RGBD).
`squelette_dataset.MMFiDatasetSqueletteSeul` (nouveau) : squelette
vérité-terrain SEUL comme entrée PRINCIPALE (pas un signal auxiliaire de
régression comme `MMFiDatasetSquelette`), pleine résolution temporelle.
`entrainer_stgcn.py` : même protocole exact que toutes les autres
comparaisons (2000 pas, lot=64, AdamW 1e-3, graines 0/1/2).

| graine | précision |
|---|---|
| 0 | 1,000 |
| 1 | 0,995 |
| 2 | 0,986 |
| **moyenne ± écart-type** | **0,994 ± 0,007** |

**Résultat quasi-parfait, suffisamment surprenant pour exiger une
vérification anti-fuite avant d'être cru :**
1. Premier contrôle négatif RATÉ (erreur de protocole, pas de bug) :
   permutation des indices de classe (renommer la classe 5 en 12 partout) —
   toujours 0,963. Une renumérotation systématique ne détruit aucune
   structure apprenable, ce n'est pas un vrai contrôle de bruit.
2. Contrôle négatif CORRECT : un label ALÉATOIRE INDÉPENDANT par séquence
   (train et val séparément). Sous ce vrai bruit, la précision plafonne à
   0,074 (attendu ~0,037) et la perte d'entraînement elle-même ne descend
   jamais sous ~3,0 — le modèle ne peut même pas mémoriser le bruit sur le
   train. **Confirmé : aucune fuite, 0,994 est un signal réel.**

Cohérent avec la littérature skeleton-HAR sur actions distinctes/scriptées
avec squelette VÉRITÉ-TERRAIN (pas estimé) : l'écart avec le 54,44%/35,97%
du pipeline original MM-Fi (AGCN/CTRGCN sur squelette ESTIMÉ par
pose-estimation) quantifie l'ampleur de l'erreur de pose-estimation dans ce
pipeline, pas une limite du classifieur squelette lui-même. **Pas un
résultat LiDAR-only** — le squelette vérité-terrain est une modalité
privilégiée (repère différent du nuage LiDAR brut, aucune calibration
disponible pour le dériver du LiDAR seul).

### Distillation squelette→LiDAR ("hallucination de modalité") — négatif, honnêtement mesuré

Mécanisme trouvé par la recherche (code source réel vérifié, pas
l'abstract) : Garcia, Morerio & Murino (ECCV 2018, extension de Hoffman et
al. CVPR'16), `github.com/ncgarcia/modality-distillation` — enseignant FIGÉ
(ST-GCN squelette, 0,994) supervise un étudiant LiDAR-seul via **trois**
pertes : CE dure + CE molle (température T, distille la frontière de
décision) + L2 sur les features poolées (distille la géométrie interne).
À l'inférence, l'enseignant est jeté — vraie modalité-dropout au test,
contrairement à MTGEA dont le CODE RÉEL (vérifié par la même recherche)
garde le squelette en entrée même au test malgré ce que dit le texte de
leur article.

Différence avec notre précédente tentative multi-tâche naïve
(`entrainer_squelette.py`, encodeur PARTAGÉ régressant directement les xyz,
+0,04 seulement) : ici deux modèles SÉPARÉS, l'étudiant n'apprend jamais à
produire du squelette, seulement à imiter la distribution de décision +
la géométrie de features de l'enseignant.

`modele_stgcn.ModeleSTGCN.forward(retourner_features=True)` expose la
représentation poolée pré-tête (512 dims) ; nouveau
`modele_squelette.ModeleClassificationLidar` (tokenizer+Transformer, même
architecture que la référence from-scratch) + adaptateur linéaire appris
384→512 pour la perte de features. `squelette_dataset.MMFiDatasetDistillation`
(nouveau) : paire (lidar, squelette PLEINE résolution, label) sur la même
fenêtre.

Sweep (`sweep_distillation.py`, 800 pas, 1 graine, λ1=λ2=0,5 par défaut de
Garcia et al., T et λ3 balayés) : meilleur config **T=4,0, λ3=1,0 → 0,537**
à 800 pas, déjà au-dessus de la référence from-scratch à 2000 pas —
prometteur en apparence.

**Run complet 3 graines (`entrainer_distillation_seeds.py`, 2000 pas, même
protocole exact) — l'optimisme du sweep ne survit PAS à la répétition :**

| graine | précision |
|---|---|
| 0 | 0,560 |
| 1 | 0,523 |
| 2 | 0,486 |
| **moyenne ± écart-type** | **0,523 ± 0,037** |

**Comparaison aux références (même protocole exact) :**

| condition | précision |
|---|---|
| from-scratch LiDAR (classification seule) | 0,509 ± 0,044 |
| **distillation squelette→LiDAR** | **0,523 ± 0,037** |
| squelette auxiliaire (naïf, multi-tâche) | 0,549 ± 0,009 |
| spectre Laplacien + SVM réglé (meilleur LiDAR-only) | 0,659 |
| squelette-plafond (vérité-terrain, pas LiDAR-only) | 0,994 ± 0,007 |

**Lecture honnête** : la distillation (0,523 ± 0,037) est statistiquement
**INDISCERNABLE** du from-scratch (0,509 ± 0,044, écart-type plus grand que
l'écart entre moyennes) et **pas meilleure** que le multi-tâche naïf déjà
mesuré (0,549 ± 0,009, variance 4x plus resserrée malgré un mécanisme
bien plus sophistiqué/mieux fondé théoriquement). Même verdict que TOUTES
nos tentatives précédentes d'améliorer l'encodeur LiDAR via un signal
d'entraînement plus riche (JEPA, multi-tâche, maintenant distillation) :
le plafond semble être l'**architecture encodeur elle-même**
(tokenizer+Transformer sur grille LiDAR), pas le signal supervisé qui
l'entraîne — cohérent avec le 52,7% indépendant de X-Fi (ci-dessus).
Hypothèse plausible (non testée) : un enseignant quasi-parfait (0,994)
produit des cibles molles trop proches de one-hot, même adoucies par
température, pour apporter beaucoup plus d'information que les labels durs
eux-mêmes une fois l'étudiant raisonnablement entraîné. Piste NON retenue
comme amélioration — documentée pour éviter de la retenter sans nouvelle
idée sur POURQUOI elle échouerait différemment.

### RANSAC + retrait de bruit statistique (isolation Laplacien) — échec net

Suggestion de la recherche (risque annoncé "faible") : ajouter un retrait
de plans (sol/mur) par RANSAC + retrait de bruit statistique AVANT le
DBSCAN existant, réimplémentation numpy (Open3D indisponible sur Python
3.13 cette plateforme). `laplacien_spectral._supprimer_plans()`/
`_retirer_bruit_statistique()` (nouveau), paramètre `nettoyage=True/False`.

| | split sujet | split environnement |
|---|---|---|
| SANS nettoyage (référence) | SVM=0,659 | SVM=0,643 |
| **AVEC nettoyage** | **0,248** | **0,237** |

**Effondrement net (0,66→0,24)** — le RANSAC générique (3 points au hasard,
sans contrainte de taille/orientation), appliqué sur le nuage déjà réduit
au foreground (~50-150 points), trouve trop facilement un "plan" dans une
partie du corps lui-même (dos, torse, bras tendu — localement plats à cette
faible densité) et le retire comme s'il s'agissait d'un mur, mutilant la
personne au lieu du fond. **Non adopté.** Piste d'amélioration non testée
si repris : appliquer le RANSAC sur la frame BRUTE (avant soustraction de
fond, quand sol/mur sont de vrais grands plans) plutôt qu'après, et/ou
contraindre les plans à l'horizontale/verticale avec une étendue minimale.

## Vue d'ensemble du pipeline

```
frame*.bin (MM-Fi)                          mmfi_dataset.py
        │                                   (chargement, mapping actions)
        ▼
nuage de points (N,3) x plusieurs frames
        │
        ├── méthode grille ─────────────┐   tokenisation.py
        │   projection multi-canaux     │   (projeter_multicanaux,
        │   repère capteur brut         │    TokenizerSpatioTemporel)
        │   → tenseur (T,4,H,W)         │
        │   → conv 3D → patches         │
        │                               │
        └── méthode points ─────────────┤   tokenisation.py
            FPS + kNN par frame         │   (tokeniser_frame_points,
            → mini-PointNet → patches   │    TokenizerPoints)
                                         │
                                         ▼
                           masquage multi-blocs plein-temps    masquage.py
                           (cible/contexte disjoints,           (MasquageSpatioTemporel
                            token masqué)                        / MasquagePoints)
                                         │
                                         ▼
        contexte ──► encodeur (entraîné) ──┐        encodeur.py
                                            ├─► prédicteur ──► perte   predicteur.py
        patches complets ──► encodeur EMA ─┘   (contexte+cible,       perte.py
        (cible, stop-grad)                      stop-grad)

                assemblage (ModeleJEPA / ModeleJEPAPoints) : modele.py
                données batchées : dataset_jepa.py
                boucle d'entraînement (une seule, paramétrée) : entrainement.py
```

## Modules

**Note (3 septembre)** : `tokenisation.py`, `masquage.py`, `perte.py`,
`modele.py`, `entrainement.py` decrits ci-dessous concernent la methode
grille TELLE QU'ELLE ETAIT AVANT la reconstruction V-JEPA -- l'etat courant
de la methode grille (masquage par-echantillon a deux groupes, perte L1,
scheduleurs) est documente dans la section "Reconstruction V-JEPA"
plus bas. La description de la methode POINTS dans ces sous-sections reste
a jour (non touchee par la reconstruction).

### `protocole.py` — protocole d'évaluation commun (11 septembre)

Module partagé par **toutes** les pipelines de features gelées (Laplacien,
Point-MAE, fusion). Il regroupe ce que chaque script d'évaluation
réimplémentait dans son coin :

- `decouper_train_val()` / `decouper_train_val_environnement()` — les deux
  splits, déplacés depuis `dataset_jepa.py` (qui importe torch, inutile pour
  du scikit-learn ; il les ré-exporte pour les appelants historiques).
- `SPLITS` — le couple (sujet, environnement) sur lequel tout résultat doit
  être rapporté.
- `masque_lignes()` — masque booléen des lignes de features d'un split
  (anciennement `entrainer_laplacien_v4._masque`, importé depuis un script
  d'entraînement par sept autres fichiers).
- `pool_temporel()` / `moyenne_par_sequence()` — les deux agrégations
  (anciennement dans `tuner_pointmae_classique` / `tuner_fusion`).
- `SVM_TUNE_PARAMS`, `HGB_TUNE_PARAMS`, `GRILLE_SVM` — les hyperparamètres et
  la grille de référence du projet, en un seul endroit.
- `regler_svm()` — GridSearchCV + GroupKFold **par sujet**, sur le **train
  seul**, scaler à l'intérieur du pipeline (donc refité dans chaque fold).
- `evaluer_classifieurs()` — SVM défaut / RF défaut / HGB réglé / SVM réglé.
- **`evaluer_deux_splits()`** — fait passer n'importe quelle fonction
  d'évaluation sur les **deux** splits. C'est le garde-fou : une nouvelle
  pipeline ne peut plus publier un résultat split sujet sans son équivalent
  split environnement.

### `mmfi_dataset.py` — chargement MM-Fi (partagé)

- `lister_sequences()` : énumère les triplets (environnement, sujet, action)
  disposant d'un dossier `lidar/`.
- `charger_frame_lidar()` : lit un `frame*.bin` (float64 brut) → `(N, 3)`
  x,y,z. **Pas de canal d'intensité** dans MM-Fi (vérifié sur données réelles,
  contrairement à l'hypothèse initiale de la note 5).
- `charger_verite_terrain()` : squelette 3D `(T, 17, 3)` par séquence — MM-Fi
  fournit un vrai squelette annoté, pas utilisé pour l'instant, mais
  disponible si besoin de validation croisée plus tard.
- `ACTIONS` : mapping A01–A27 → nom (Table 5 du papier MM-Fi, arXiv:2305.10345).

Vérifié sur les **1080 séquences** du dataset (`rapport_generalisation.txt`) :
0 erreur de chargement, 1009–1660 points par frame, aucune séquence dégénérée.

### `tokenisation.py` — les deux tokenizers

**Méthode grille** — `projeter_multicanaux()`/`empiler_fenetre_temporelle()`
(prétraitement, repris de
`lidar-src/lidar-exploration/data_projection/projection_unitree_l2_simple.py`,
détaché de ROS2, étendu à 4 canaux x,y,z,validité) + `TokenizerSpatioTemporel`
(appris, conv 3D à stride fixe + positional embedding sin-cos 3D repris de
V-JEPA — la grille empilée dans le temps a exactement la forme d'un clip
vidéo, donc la même tokenisation s'applique sans adaptation, cf. note 4/5).

- **Pas de recentrage** (retiré le 31 août). Deux raisons : (1) sur MM-Fi le
  bénéfice était limité — protocole très contraint (x toujours entre 2 et 4 m
  sur les 1080 séquences) ; (2) aligner les deux méthodes sur le même
  traitement des coordonnées (repère capteur brut) isole la comparaison sur
  la vraie variable d'intérêt (tokenisation), pas sur un choix de repère
  différent.
- **Bug trouvé et corrigé** (pendant l'implémentation du recentrage, avant
  son retrait — leçon toujours valable) : les angles de projection (yaw/pitch)
  doivent être calculés à partir des coordonnées **brutes**, jamais
  recentrées, sinon la géométrie sphérique est faussée (x = axe de profondeur
  du capteur, toujours positif ; le recentrer casse `atan2`/`arcsin`).
- FOV et taille de grille (`HAUTEUR=64, LARGEUR=96`, yaw∈[-60°,60°],
  pitch∈[-35°,20°]) calibrés empiriquement sur MM-Fi. Validé sur les 1080
  séquences : occupation moyenne 12,49 % (±1,07 %), stable entre les 4
  environnements, aucun point clippé même en se penchant (vérifié sur
  A19/Picking up things).
- `TokenizerSpatioTemporel` : la validité ne passe pas dans la conv (pas une
  caractéristique géométrique à apprendre) ; sert à déterminer si un patch
  est structurellement vide (`max_pool3d`), remplacé alors par un **token
  vide appris** plutôt que la sortie non significative de la conv (mécanisme
  AD-L-JEPA, note 6). 192 patches (8×4×6) pour une fenêtre de 16 frames ;
  128/192 occupés (66,7 %) sur la fenêtre de test — bien plus que
  l'occupation pixel (~10,7 %), un seul pixel valide marquant tout un tubelet.

**Méthode points** — `tokeniser_frame_points()` (prétraitement non appris,
FPS puis kNN, fait côté `Dataset`/CPU pour une sortie de forme fixe malgré le
nombre de points brut variable) + `TokenizerPoints` (appris, mini-PointNet
repris de Point-JEPA + positional embedding sur coordonnées **réelles
continues** x,y,z,t plutôt que sur un indice de grille).

- K=32 centres, k=24 voisins — réduit par rapport à Point-JEPA (K=64/k=32 sur
  ShapeNet) pour rester tractable. Avec T=16 frames : 512 patches au total
  contre 192 pour la grille — différence assumée (propriété intrinsèque de
  chaque tokenisation), pas quelque chose à corriger.
- **Pas de séquenceur glouton** : Point-JEPA en a besoin pour une seule frame
  statique (nuage invariant par permutation) ; ici on compte sur l'attention
  + la position réelle continue pour recoller les patches d'une frame à
  l'autre (option B retenue le 31 août — moins risqué à implémenter que le
  tracking explicite de centres, pas encore comparé à cette alternative).

### `masquage.py` — masquage multi-blocs, les deux méthodes

**Méthode grille** — `MasquageSpatioTemporel` : bloc échantillonné dans les
dimensions spatiales (`nb_h,nb_w`) puis répété à l'identique sur toute la
profondeur temporelle — point critique de V-JEPA qui empêche le modèle de
copier une frame voisine non masquée (note 4). Presets adaptés à notre grille
de 24 cellules (les comptes de blocs de V-JEPA/Point-JEPA saturent une grille
aussi petite). **Bug trouvé et corrigé** : le contexte peut, par tirage, être
entièrement recouvert par la cible (contexte vide) — corrigé par réessais
puis repli sur "tout ce qui n'est pas la cible".

**Méthode points** — `MasquagePoints` : blocs définis comme boîte (x,y,z),
pas comme indices de grille. Le masquage plein-temps est **automatique** ici
(une boîte n'a pas de dimension temporelle). **Bug trouvé et corrigé** : les
boîtes échantillonnées uniformément dans les bornes spatiales ratent le nuage
de points dans 15-26 % des tirages (test Monte-Carlo) — les bornes sont
presque vides d'air (points = fine surface corporelle dans un volume
englobant bien plus grand). Corrigé en ancrant chaque boîte sur un centre de
patch **réel** tiré au hasard. Après correctif : 0 % de tirages à contexte
vide. **Échelles recalibrées** (31 août) pour aligner les proportions
cible/contexte sur celles de la grille (~50 %/~29 % médian) — les valeurs
reprises telles quelles de la grille donnaient 59 %/19 %, un écart qui aurait
confondu la comparaison entre méthodes.

Commun aux deux : token "masqué" appris, partagé entre positions vides et
occupées masquées, pour ne pas laisser le masque révéler l'occupation
(AD-L-JEPA/Voxel-MAE, note 6). Restriction NOMAE non implémentée (occupation
patch déjà élevée des deux côtés, loin du déséquilibre extrême qui la
motivait). Masquage échantillonné une fois par lot, pas par échantillon
(simplification — V-JEPA échantillonne un masque différent par clip).

### `encodeur.py`, `predicteur.py` — architecture (strictement partagés)

- `Encodeur` : transformer standard (blocs pré-norme, attention multi-tête,
  MLP GELU) — même code pour l'encodeur de contexte et l'encodeur cible des
  deux méthodes.
- `Predicteur` : transformer plus étroit (`dim_predicteur < dim`), asymétrie
  contexte/prédicteur — mécanisme anti-effondrement standard.

Les deux méthodes instancient ces classes avec les **mêmes hyperparamètres**
par défaut (dim=192, dim_predicteur=96, profondeur_encodeur=4,
profondeur_predicteur=2, nb_tetes=6) — condition nécessaire pour que leur
comparaison porte sur la tokenisation, pas sur l'architecture du transformer.

### `perte.py` — perte JEPA (partagé)

- `perte_jepa` (grille) : similarité cosinus (comme AD-L-JEPA), **scindée en
  deux termes pondérés** (0,25 vide / 0,75 occupé, repris tels quels
  d'AD-L-JEPA, note 6).
- `perte_jepa_points` (points) : même perte sans le split vide/occupé (sans
  objet — FPS ne sélectionne que des points réels).
- `perte_variance` (partagée) : régularisation VICReg — **ajoutée suite à un
  effondrement partiel observé empiriquement** (voir Résultats). Hinge sur
  l'écart-type par dimension (seuil 0.2, calibré sur l'échelle de nos
  embeddings). Garde-fou contre NaN si moins de 2 lignes.
- `perte_uniformite_poolee` (partagée, **3 septembre**) — régularise la
  représentation POOLÉE (mean+max sur les patches, exactement ce qu'évalue le
  linear probing, cf. `evaluation.py`), pas seulement les patches individuels
  comme `perte_variance`. Historique : le linear probing (2 septembre) a
  révélé un effondrement quasi total de cette représentation poolée
  (similarité cosinus **~0,999** entre fenêtres différentes) alors que
  `perte_variance` par patch semblait fonctionner normalement (~0,25-0,36).
  Premier correctif tenté — `perte_variance_poolee`, même recette VICReg mais
  sur la moyenne des patches — s'est avéré **inefficace** (0,9933 après
  600 pas, quasi inchangé) : la variance BRUTE (non normalisée) d'un vecteur
  peut être satisfaite par une simple variation d'ÉCHELLE (magnitude) sans que
  sa DIRECTION change, or c'est justement le cosinus (donc la direction)
  qu'évalue le probing en aval. Remplacée par une perte d'uniformité
  hypersphérique (Wang & Isola 2020, "Understanding Contrastive
  Representation Learning through Alignment and Uniformity on the
  Hypersphere") : normalise la représentation poolée en norme unité puis
  pénalise directement la proximité angulaire moyenne entre échantillons
  différents du lot — cible exactement la quantité mesurée en aval. Résultat
  sur le même diagnostic (600 pas) : similarité cosinus **0,75**, et
  **0,09-0,21** sur les 600 premiers pas d'un entraînement complet
  (3000 pas) — voir Résultats.

### `modele.py` — assemblage des deux méthodes

`ModeleJEPA` (grille) / `ModeleJEPAPoints` (points) : tokenizer partagé entre
contexte et cible (comme Point-JEPA) → masquage → encodeur de contexte
(entraîné) + encodeur cible (copie EMA, `requires_grad=False`, mise à jour
via `mettre_a_jour_cible()`) → prédicteur.

Vérifications d'intégration effectuées (les deux méthodes) : gradients non
nuls sur tokenizer/encodeur de contexte/prédicteur, gradient **exactement
nul** sur l'encodeur cible, mise à jour EMA validée numériquement.

### `dataset_jepa.py` — données batchées (partagé)

- `_MMFiFenetreDataset` : sélection de fenêtre partagée — position aléatoire
  dans la séquence, ré-échantillonnée à chaque appel (comme le tirage de
  clips vidéo de V-JEPA).
- `MMFiDatasetJEPA` (grille) / `MMFiDatasetJEPAPoints` (points) : appliquent
  le prétraitement propre à chaque méthode sur cette même sélection de
  fenêtre.
- `decouper_train_val()` : découpage par sujet (864 séquences train / 216
  val), même principe que le split "cross-subject" du toolbox MM-Fi officiel.

### `entrainement.py` — boucle d'entraînement (une seule fonction, deux méthodes)

`entrainer(methode="grille"|"points", ...)` : une seule fonction paramétrée
plutôt que deux scripts dupliqués, pour garantir que les deux méthodes
tournent avec **exactement les mêmes hyperparamètres** d'entraînement
(nb_etapes, taille_lot, taux d'apprentissage, décroissance EMA, écrêtage de
gradient, graine) — condition nécessaire pour une comparaison pertinente.

- **Bug trouvé et corrigé** (31 août, présent sur tous les runs précédents,
  les deux méthodes) : l'optimiseur n'incluait pas `modele.masquage` — le
  token masqué n'était donc jamais entraîné, figé à son initialisation
  aléatoire. Corrigé.
- **Graine aléatoire fixée** (`graine=0` par défaut) — absente avant le
  31 août, ajoutée pour la reproductibilité et pour que la comparaison entre
  méthodes ne soit pas bruitée par une initialisation différente.
- Validation périodique, diagnostic d'effondrement (similarité cosinus entre
  tokens indépendants), **et depuis le 3 septembre, diagnostic d'effondrement
  POOLÉ** (`diagnostic_effondrement_poolee` — même calcul que le linear
  probing, suivi en direct pendant l'entraînement plutôt que découvert après
  coup), checkpoints intermédiaires (`checkpoints_{methode}/`), historique CSV
  complet (`historique_entrainement_{methode}.csv`).

### `evaluation.py` — linear probing (métrique en aval, 27 actions MM-Fi)

Ajouté le 2 septembre : jusque-là, seuls des diagnostics de santé de
l'entraînement pretexte (perte, similarité d'effondrement par patch) étaient
suivis — aucune métrique ne mesurait si le pré-entraînement produit
réellement une représentation UTILE, contrairement aux articles de référence.
Même principe que Point-JEPA (note 3) : encodeur gelé, pooling mean+max sur
les patches, tête linéaire entraînée sur les 27 actions de MM-Fi
(`probe_lineaire`). Compare 4 conditions (grille/points ×
pré-entraîné/aléatoire) — le contrôle aléatoire (encodeur jamais entraîné)
vérifie que le pré-entraînement apporte réellement quelque chose, pas
seulement que la tête linéaire sait séparer les classes toute seule.

**Découverte critique (2 septembre)** : sur les checkpoints d'avant
`perte_uniformite_poolee`, les encodeurs pré-entraînés étaient
systématiquement BATTUS par les encodeurs aléatoires (grille : perte
~3,1-3,5 bloquée près du hasard / précision 4-7% pour le pré-entraîné, contre
perte 1,0-1,7 / précision 10-13% pour l'aléatoire ; même schéma pour points).
Diagnostiqué comme un effondrement de la représentation POOLÉE (cf.
`perte.py` ci-dessus) — invisible aux diagnostics d'entraînement existants,
qui ne regardaient que les patches individuels. `PoolingAttentif` (pooling
par attention, à la V-JEPA) testé comme correctif alternatif avant
d'identifier la vraie cause — **sans effet** (toujours bloqué au hasard),
confirmant que le problème était la représentation elle-même, pas la méthode
de pooling. A motivé `perte_uniformite_poolee` et un ré-entraînement complet
des deux méthodes — voir Résultats.

## Résultats obtenus jusqu'ici (méthode grille)

**Smoke test initial (200 pas, sans régularisation de variance)** : perte
1,03 → 0,03. Diagnostic post-hoc : similarité cosinus moyenne entre tokens
indépendants = **0,94** — effondrement partiel (dimensionnel, pas total).
Confirme le risque anticipé en note 3 (la recette EMA+stop-grad seule suffit
à Point-JEPA sur des objets denses et variés, pas ici où beaucoup de
structure se répète).

**Après ajout de la régularisation de variance (200 pas)** : bug de contexte
vide → NaN intermittents, corrigé. Une fois corrigé : perte 1,07 → 0,10,
similarité cosinus réduite à **0,75**.

**Entraînement long (3000 pas, ~27 epochs)** : terminé sans plantage ni NaN
(un premier essai a été interrompu à l'étape 904 par un redémarrage de la
machine, sans lien avec le code). Voir `courbes_entrainement.png`.

- Similarité d'effondrement : 0,89 → majoritairement 0,15–0,30 en fin
  d'entraînement, mais **pas monotone** (deux remontées notables). La
  régularisation de variance aide clairement mais ne règle pas le problème
  de façon stable.
- **Perte de validation très instable** : oscille entre 0,03 et >1,0 sans
  tendance de baisse soutenue après les ~400 premiers pas, alors que la perte
  d'entraînement continue de descendre. Signe probable de sur-apprentissage
  sur un corpus d'entraînement minuscule pour du SSL (864 séquences) —
  cohérent avec le risque anticipé dans le plan d'action (note 2). Pas un
  défaut du pipeline : c'est pourquoi la stratégie prévue combine plusieurs
  datasets (tiers 1-3) plutôt que MM-Fi seul.

**Méthode points** : implémentée et vérifiée mécaniquement (gradients,
masquage plein-temps, token masqué entraîné). Entraînement long (3000 pas)
terminé le 3 septembre — voir section linear probing ci-dessous pour la
comparaison directe avec grille.

## Linear probing — résultats (3 septembre, après `perte_uniformite_poolee`)

Voir `courbes_entrainement_v2.png` (perte + similarités par méthode) et
`comparaison_probing_v2.png` (précision des 4 conditions au fil du probing) —
générées par `tracer_courbes.py`, à relancer après tout nouvel entraînement.

Les deux méthodes ont été **ré-entraînées de zéro** (3000 pas chacune,
mêmes hyperparamètres, anciens checkpoints sauvegardés dans
`checkpoints_{grille,points}_pre_fix/`) avec la perte d'uniformité poolée
(cf. `perte.py`), puis comparées via `evaluation.py` (4 conditions,
2000 pas, lot=64, pooling mean+max).

**Résultats finaux (`log_evaluation_v2.txt`, checkpoints post-correctif) :**

| condition | précision val | f1 macro val |
|---|---|---|
| grille pré-entraîné | **0,139** | 0,082 |
| grille aléatoire | 0,111 | 0,085 |
| points pré-entraîné | 0,014 | 0,004 |
| points aléatoire | **0,181** | **0,129** |

**Grille : le correctif fonctionne, gain net mais modeste.** Précision
pré-entraîné > aléatoire (0,139 vs 0,111 — l'effondrement qui bloquait tout
apprentissage en aval est bien résolu), mais le f1 macro est quasiment à
égalité (0,082 vs 0,085) : le pré-entraînement aide à mieux classer en
moyenne mais pas de façon uniforme sur les 27 classes. Similarité poolée
pendant l'entraînement : partie de ~0,97 (init) à des valeurs oscillant
entre -0,02 et 0,20 tout au long des 3000 pas (contre ~0,999 constant avant
correctif) — effondrement résolu.

**Points : le correctif résout l'effondrement du pooling PENDANT
l'entraînement (même diagnostic sain que grille, ~-0,07 à 0,31), mais le
pré-entraînement reste très nettement INFÉRIEUR à l'aléatoire en aval**
(précision 0,014 vs 0,181 — l'aléatoire est même la MEILLEURE des 4
conditions). Vérifié pas à pas sur les 2000 pas de probing : le
pré-entraîné reste bloqué au hasard (perte ~3,2-3,4, 1/27 ≈ 0,037) du
début à la fin, sans aucune tendance à la hausse, alors que l'aléatoire
apprend normalement et régulièrement dès les 200 premiers pas (comme avant
le correctif). Donc : contrairement à grille, corriger l'effondrement
POOLÉ n'a **pas** rendu la représentation points utile en aval — deux
problèmes distincts, le second seulement résolu pour grille.

**Hypothèse (non testée à ce stade)** : `positional_embedding_points`
(`tokenisation.py`) encode les coordonnées (x,y,z,t) **absolues**, repère
capteur brut (pas de recentrage, cf. décision du 31 août), directement
additionnées à l'embedding de contenu de CHAQUE patch. Contrairement à la
grille — où l'information de position n'entre que via le PATTERN
d'occupation dans une grille canonique fixe, jamais une valeur de coordonnée
brute injectée telle quelle — le pooling moyen des patches points peut
satisfaire la pression d'uniformité en encodant simplement "où se trouvait
la personne dans la pièce durant cette fenêtre" (qui varie sujet à sujet,
prise à prise) plutôt que le contenu de l'action — un raccourci sémantiquement
vide mais qui suffit à séparer les fenêtres les unes des autres. La grille
n'a pas ce raccourci disponible. Piste de correctif non testée par manque de
temps : rendre l'embedding de position RELATIF à la fenêtre (ex. centré sur
le centroïde de la fenêtre) plutôt qu'absolu — attention à ne pas casser le
rôle de cet embedding pour le prédicteur (qui a besoin de savoir OÙ prédire).

## Reconstruction V-JEPA (3 septembre) — méthode grille uniquement

Demande explicite : reconstruire la méthode grille pour mirroirer fidèlement
V-JEPA plutôt que s'en inspirer librement. Code source réel de
`facebookresearch/jepa` téléchargé verbatim dans `reference_vjepa/` (pas
depuis la mémoire — chaque fichier lu en entier avant de porter quoi que ce
soit) : `src/models/vision_transformer.py`, `predictor.py`,
`utils/{patch_embed,pos_embs,modules}.py`, `src/masks/multiblock3d.py`,
`src/utils/tensors.py`, `app/vjepa/train.py`, `configs/pretrain/vitl16.yaml`.

**Portée décidée avec l'utilisateur** : méthode grille seulement (V-JEPA n'a
pas d'équivalent pour le tokenizer points, FPS+kNN) ; conception
dataset-agnostique (fonctionne sur la grille MM-Fi 8×4×6 et ONCE 8×4×18 sans
changement), même si seul MM-Fi a été validé par un entraînement complet
cette session (ONCE ~5h/run).

**Changements par fichier :**

- **`tokenisation.py`** : le token "vide" appris (qui remplaçait la sortie
  de la conv sur les cellules sans retour LiDAR) est retiré. La validité
  devient un 4e canal d'ENTRÉE de la conv (x,y,z,validité) plutôt qu'un
  post-traitement — V-JEPA n'a pas de notion de pixel "vide" et n'en a pas
  besoin, le modèle apprend à représenter l'absence de retour implicitement.
- **`masquage.py`** (la plus grosse réécriture) : masquage par-ÉCHANTILLON
  (avant : un seul masque tiré par lot de 8, partagé) — chaque séquence du
  lot a maintenant sa propre position de bloc aléatoire, comme
  `_sample_block_mask` dans multiblock3d.py. Surtout : **deux groupes de
  masques simultanés** (courte-portée : 4 petits blocs / longue-portée : 1
  grand bloc), mirroir exact des deux entrées `mask:` de `vitl16.yaml` — un
  token masque appris par groupe. Les deux groupes ont des tailles de
  contexte différentes par construction (pas concaténables sur la dimension
  batch comme le fait `apply_masks(..., concat=True)` de V-JEPA — testé,
  ça casse `torch.cat`, cf. `rassembler`/commentaires dans le code) ; traités
  par un appel encodeur/prédicteur séparé par groupe à la place.
- **`encodeur.py`/`predicteur.py`** : init troncature-normale (std=0,02) +
  rescale résiduelle par profondeur (`÷ sqrt(2·layer_id)` sur les
  projections de sortie attention/MLP), mirroir de `_rescale_blocks`
  (stabilité standard DeiT/BEiT, absente avant). Le prédicteur était déjà
  structurellement proche de `VisionTransformerPredictor` (token masque +
  position en requête, concaténé au contexte) — pas de réécriture.
- **`perte.py`** : `perte_jepa` (grille) passe de similarité cosinus scindée
  vide/occupé à **L1 pure** (`mean(|pred-cible|)`), moyennée PAR GROUPE de
  masque puis non-pondérée entre groupes (mirroir exact de `loss_fn` dans
  `train.py` — pas une moyenne aplatie, qui pondérerait par la taille de
  cible de chaque groupe). `perte_variance`/`perte_uniformite_poolee`
  restent définies mais à poids 0.0 par défaut (mirroir de `reg_coeff: 0.0`
  de vitl16.yaml) — **testé et invalidé** (voir plus bas), réactivées.
- **`modele.py`** : `ModeleJEPA.forward` retourne des LISTES
  (predictions/cibles/contexte_encode, une entrée par groupe de masque) au
  lieu de tenseurs uniques — nécessaire pour que `perte.py` moyenne les
  groupes à égalité et parce que leurs tailles diffèrent.
- **`entrainement.py`** : ajout des trois scheduleurs de `train_step`
  (`reference_vjepa/app/vjepa/train.py`) absents avant : warmup linéaire +
  décroissance cosinus du taux d'apprentissage, weight decay croissant
  linéairement, momentum EMA croissant linéairement (0,996→0,9999),
  écrêtage de gradient actif seulement après le warmup. Formes identiques à
  V-JEPA, amplitudes réduites à notre échelle.

**Le test empirique qui a motivé de garder la régularisation** : `reg_coeff: 0.0`
dans `vitl16.yaml` — le régularisateur de variance de V-JEPA est calculé
mais ne contribue AUCUN gradient en production, ils comptent entièrement sur
EMA+stop-gradient+masquage. Testé fidèlement (`diag_vjepa_mirror.log`, 500
pas, MM-Fi grille, tout le reste du mirroir actif) : **échec** — similarité
poolée restée ~0,93-0,96 tout le run (contre ~0,999 sans AUCUN correctif,
~0,0-0,2 avec la régularisation d'uniformité), quasi identique à
l'effondrement original de la note 3 (0,94, EMA+stop-grad seuls, avant
même l'ancien correctif). Réactiver `poids_uniformite_poolee=1.0` sur ce
même mirroir (`diag_vjepa_mirror_reg.log`) résout l'effondrement
immédiatement (0,16 à l'étape 200, ~0,02-0,06 en fin d'entraînement complet).
Conclusion : l'absence de régularisation explicite est spécifique au régime
de V-JEPA (vidéo dense à grande échelle), pas transférable à notre donnée
LiDAR petite échelle/parcimonieuse/structure répétitive, même avec un
masquage fidèlement reproduit.

**Résultat final (entraînement complet 3000 pas, `checkpoints_grille_vjepa/`,
linear probing `log_eval_vjepa.txt`)** :

| condition | précision | f1 macro |
|---|---|---|
| grille_vjepa pré-entraîné | 0,120 | 0,072 |
| grille_vjepa aléatoire | 0,102 | 0,100 |

Pré-entraîné bat l'aléatoire en précision (comme l'ancienne architecture,
0,139 vs 0,111) mais perd en f1 macro plus nettement que l'ancienne
architecture (0,072 vs 0,100, contre 0,082 vs 0,085 avant). Sur un seul run
sans graines répétées, pas une nette amélioration par rapport à l'ancienne
architecture — mais une architecture plus fidèle/mieux fondée, avec les
mêmes performances en aval. Courbes/diagnostics d'entraînement complets
dans `historique_entrainement_grille_vjepa.csv`.

## Fine-tuning en aval (3 septembre soir) — DeSPITE

Suite a la lecture de DeSPITE (ICCV 2025, arXiv:2506.13897, cf. Tracking/mois/
3-septembre/1-despite_smpl_lidar_recherche.txt) : leurs propres resultats
montrent que le linear probing SOUS-ESTIME l'apport du pre-entrainement
("linear probing shows weaker performance, indicating pretraining benefits
accumulate through fine-tuning"). `evaluation.py` supporte maintenant un mode
fine-tuning (`geler_encodeur=False` dans `probe_lineaire`/`construire_modele`)
qui degele `tokenizer`+`encodeur_contexte` (les deux seuls sous-modules
exerces en probing) avec un taux d'apprentissage separe, plus faible que
celui de la tete (1e-4 vs 1e-3 par defaut).

**Resultat (meme protocole que le linear probing, 2000 pas, lot=64,
`eval_finetune_vjepa.py`, resultats finaux dans `log_eval_finetune_vjepa.txt`) :**

| condition | precision | f1 macro |
|---|---|---|
| grille_vjepa pre-entraine + fine-tune | 0,486 | 0,505 |
| grille_vjepa from-scratch (aleatoire + fine-tune) | **0,514** | 0,508 |

Deux lecons, une confirmee, une qui nuance fortement la premiere :

1. **Le linear probing sous-estimait severement la capacite du modele** —
   confirme : les DEUX conditions montent de ~0,04 (hasard) a ~0,49-0,51 des
   que l'encodeur peut s'adapter, plus de 4x le resultat en probing lineaire
   (0,120/0,102). Le modele (architecture, taille) sait clairement resoudre
   la tache une fois autorise a ajuster ses features.
2. **Mais l'avantage du pre-entrainement disparait, voire s'inverse
   legerement, en fine-tuning complet** — from-scratch (0,514) devance
   legerement pre-entraine (0,486). Avec un budget de fine-tuning suffisant
   (2000 pas sur 864 sequences), le pre-entrainement n'apporte pas de gain
   net mesurable sur CE run (une seule graine, a confirmer/infirmer avec
   repetition). Lecture honnete : soit le pre-entrainement JEPA actuel (MM-Fi
   seul, 864 sequences, 3000 pas) n'encode pas d'information utile au-dela
   de ce qu'un entrainement supervise direct capture deja sur ce meme
   dataset, soit la tache est assez facile a apprendre from-scratch avec ce
   volume de donnees labellisees pour que l'avantage du pre-entrainement
   s'estompe une fois l'encodeur completement degele. Motive d'autant plus
   la piste JRDB/donnees combinees (cf. section suivante) : le
   pre-entrainement doit apporter des priors que MM-Fi seul, deja disponible
   en supervise, ne peut pas deja fournir.

## Dataset 3 — JRDB (corpus domaine-proche, 3 septembre soir)

`jrdb_dataset.py` (nouveau fichier) : chargement du split train de JRDB
(jrdb.erc.monash.edu, telechargement "Train Data, Subsets" — nuages de
points uniquement, pas les images/labels). Format verifie sur donnees
reelles : `train/pointclouds/lower_velodyne/{sequence}/{frame:06d}.pcd`, PCD
binaire compresse (`pypcd4` pour la lecture — `open3d` indisponible pour
Python 3.13 sur cette plateforme). 27 sequences, 27 893 frames au total,
capteur = un des deux Velodyne VLP-16 du robot JackRabbot (lower_velodyne
seul, upper_velodyne ignore pour eviter une fusion multi-capteurs non
calibree).

Calibre empiriquement (echantillon de 30 frames, ~475k points) : 16 canaux
verticaux distincts a exactement -15/+15 degres (pas 2 degres, VLP-16
standard confirme), azimut 360 degres complets, portee p1=0,9m / p50=3,7m /
p99=34,8m. `PROFIL_JRDB` dans `tokenisation.py` : grille 16x180
(`noyau_spatial=4`, seul diviseur commun pratique de 16 et 180).

**Limite connue, non resolue** : le robot JackRabbot SE DEPLACE pendant la
capture (comme le vehicule ONCE) mais aucune donnee de pose/odometrie n'a ete
telechargee (seulement les nuages de points) — PAS de compensation
d'ego-mouvement appliquee, contrairement a `once_dataset.py`. Le masquage
plein-temps suppose une meme cellule de grille = meme endroit reel d'une
frame a l'autre au sein d'une fenetre, invalide pendant les deplacements du
robot. A corriger si le pre-entrainement JRDB semble degrade par ce biais —
necessiterait un telechargement supplementaire des donnees de pose du robot
(compte JRDB deja cree, pas de nouvelle inscription).

**Combinaison avec MM-Fi** : `JrdbDatasetJEPA` accepte un `profil` (defaut
`PROFIL_JRDB`, standalone) ou `tok.PROFIL_MMFI` explicitement — projeter
JRDB a travers le profil MM-Fi (FOV etroit, courte portee) donne la MEME
forme de grille que MM-Fi (verifie : 20,95% d'occupation sur JRDB avec ce
profil, contre 95,55% avec son propre PROFIL_JRDB natif — plausible, pas
degenere), condition necessaire pour batcher les deux sources ensemble dans
un seul entrainement. `entrainement.py` : nouvelle source
`source="mmfi_jrdb"`, `ConcatDataset([MMFiDatasetJEPA(...),
JrdbDatasetJEPA(..., profil=PROFIL_MMFI)])`, 889 sequences train (864 MM-Fi
+ 25 JRDB) / 218 val.

**Entrainement complet (3000 pas, `entrainer_combine.py`,
`checkpoints_mmfi_jrdb/`) : sain de bout en bout**, `sim_effondrement`
0,17-0,72, `sim_poolee` 0,02-0,18 tout le run (meme plage saine que les runs
MM-Fi seul et JRDB seul), aucune instabilite.

**Fine-tuning (meme protocole exact que la section precedente : 2000 pas,
lot=64, `eval_finetune_combine.py`, `log_eval_finetune_combine.txt`) :**

| condition | precision | f1 macro |
|---|---|---|
| grille pre-entraine MM-Fi+JRDB + fine-tune | 0,514 | **0,523** |
| grille pre-entraine MM-Fi seul + fine-tune | 0,486 | 0,505 |
| grille from-scratch (aleatoire) + fine-tune | 0,514 | 0,508 |

**A graine=0, egalite stricte en precision (0,514/0,514) et leger mieux en
f1 (0,523 vs 0,508) — mais cette lecture ne survit PAS a la repetition (cf.
ci-dessous).**

**Repetition multi-graines (4 septembre matin, `eval_finetune_seeds.py`,
`log_eval_finetune_seeds.txt`)** : le run a graine=0 avait sa propre mise en
garde des l'ecriture (trajectoire de validation oscillant entre 0,458 et
0,565 sur les 500 derniers pas) — verifiee en repetant les DEUX conditions
sur 2 graines supplementaires (1 et 2, meme protocole exact) :

| graine | combine (MM-Fi+JRDB) precision | scratch precision |
|---|---|---|
| 0 | 0,514 | 0,514 |
| 1 | 0,449 | 0,551 |
| 2 | 0,532 | 0,463 |
| **moyenne ± ecart-type (n=3)** | **0,498 ± 0,044** | **0,509 ± 0,044** |

**Conclusion honnete, corrigee par rapport a la lecture initiale a graine=0
seule** : l'ecart-type inter-graines (±0,044) depasse largement l'ecart entre
les deux moyennes (0,011) — combine et from-scratch sont statistiquement
INDISCERNABLES a ce volume de donnees. La "fermeture de l'ecart" observee a
graine=0 etait du bruit d'echantillonnage, pas un effet de JRDB. Ajouter 25
sequences JRDB (projetees via `PROFIL_MMFI`) sur un corpus de 864 sequences
MM-Fi est un ajout trop marginal (moins de 3% de donnees supplementaires)
pour produire un effet mesurable — leçon methodologique a retenir : la
lecture a une seule graine du 3 septembre soir (MM-Fi seul legerement
derriere from-scratch, 0,486 vs 0,514) merite la MEME prudence et n'a
elle-meme jamais ete repetee sur plusieurs graines.

**Suite : doubler le volume JRDB (4 septembre matin, `entrainer_combine_jrdb2x.py`
/ `eval_finetune_jrdb2x_seeds.py`)** — teste directement l'hypothese "le
volume ajoute est le facteur limitant" en reutilisant le split "test" de
JRDB (`D:/jrdb/test/`, deja extrait le 3 septembre, jamais utilise) : 27
sequences supplementaires, aucune etiquette JRDB requise nulle part dans ce
pipeline (JEPA auto-supervise ; seul MM-Fi fournit les etiquettes HAR du
fine-tuning). ~50 sequences JRDB train au lieu de 25 (~5-6% du corpus
combine). Entrainement 3000 pas sain (`checkpoints_mmfi_jrdb2x/`,
`sim_effondrement` 0,20-0,68, `sim_poolee` 0,02-0,16). Fine-tuning, 3 graines
(memes graines 0/1/2, meme protocole) :

| graine | JRDB x2 precision | JRDB x1 precision | scratch precision |
|---|---|---|---|
| 0 | 0,500 | 0,514 | 0,514 |
| 1 | 0,523 | 0,449 | 0,551 |
| 2 | 0,528 | 0,532 | 0,463 |
| **moyenne ± ecart-type** | **0,517 ± 0,015** | 0,498 ± 0,044 | 0,509 ± 0,044 |

Lecture honnete : la moyenne (0,517) est marginalement au-dessus de
from-scratch (0,509), mais l'ecart (0,008) reste bien plus petit que
l'ecart-type de from-scratch (0,044) — PAS une amelioration demontree sur
n=3. Ce qui est plus notable et digne d'interet, a prendre avec la meme
prudence (n=3 est trop petit pour estimer fiablement une variance) :
l'ecart-type inter-graines du JRDB x2 (±0,015) est nettement plus resserre
que JRDB x1 (±0,044) et que from-scratch (±0,044) — le run avec plus de
JRDB est plus STABLE d'une graine a l'autre, meme si son avantage en
moyenne reste faible. Hypothese a confirmer avec plus de graines et/ou
plus de volume : le pre-entrainement pourrait reduire la variance de
l'initialisation aleatoire avant meme d'ameliorer la moyenne — un effet
plausible mais non etabli ici. Piste naturelle si le temps le permet :
grossir encore (autres sources Tier 1/2 du 28 aout) et/ou repeter avec
plus de graines pour trancher entre "effet reel mais faible" et "encore
du bruit".

## Dataset 2 — ONCE (corpus volumineux, 1er septembre)

`once_dataset.py` (nouveau fichier — nouvelle source de données, pas une
duplication d'un rôle existant) : chargement des scènes ONCE (`raw_small`,
100k frames, extrait dans `D:/once_lidar/small/`). `raw_medium` (400k frames
supplémentaires, 251 scènes) également extrait dans `D:/once_lidar/medium/`
mais **pas encore utilisé pour l'entraînement** (le premier entraînement long
a tourné sur `small` seul — combiner les deux nécessiterait un support
multi-racine non implémenté).

**Compensation de l'ego-mouvement (2 septembre)** : contrairement à MM-Fi
(capteur fixe), le véhicule ONCE se déplace entre les frames — vérifié sur
données réelles, ~21,8 m parcourus en 2,5 s (5 frames à 500 ms) sur la scène
000000, ~31 km/h. Sans compensation, "la même cellule de grille" à deux
instants d'une fenêtre ne représente pas le même endroit réel, ce qui casse
l'hypothèse même du masquage plein-temps (V-JEPA) — cf. AD-LiST-JEPA
(arXiv:2602.12540), qui résout ce problème en ramenant chaque frame dans un
repère commun avant le masquage. ONCE fournit la pose ego-véhicule par frame
(quaternion + translation, téléchargement séparé "Annotations") ;
`once_dataset.aligner_fenetre()` recompose la transformation relative à la
première frame de CHAQUE FENÊTRE (pas de la scène) à partir des poses
relatives-à-la-scène fournies, et `_OnceFenetreDataset` l'applique
systématiquement avant projection.

Format vérifié sur données réelles : `data/{scene}/lidar_roof/{timestamp}.bin`,
float32, `(N,4)` → x,y,z,r (intensité — **non utilisée pour l'instant**, canal
r ignoré pour garder le même format à 4 canaux (x,y,z,validité) que MM-Fi ;
pourrait être ajoutée comme 5e canal plus tard). ~67k points/frame, azimut sur
360° complets (LiDAR tournant), 40 canaux verticaux distincts (confirmé par
histogramme du pitch — 40 pics nets), portée jusqu'à 200 m (99e percentile à
111 m) — échelle et capteur complètement différents de MM-Fi.

`tokenisation.py` : `projeter_multicanaux()`/`empiler_fenetre_temporelle()`
paramétrées par un `profil` (dict FOV/grille/distance) au lieu de constantes
figées, pour partager le même code entre MM-Fi et ONCE. `PROFIL_ONCE` :
grille 40×180 (40 = nombre réel de canaux du capteur, pas une valeur
arbitraire — évite le sous/sur-échantillonnage vertical), azimut ±180°,
pitch [-25°,15°], distance [0.5,120] m. `ModeleJEPA` accepte maintenant
`noyau_spatial`/`noyau_t` en paramètres (`noyau_spatial=10` pour ONCE, seul
diviseur commun pratique de 40 et 180 — 16 comme MM-Fi ne divise pas 40).

`dataset_jepa.py` : `_OnceFenetreDataset`/`OnceDatasetJEPA` (méthode grille
seulement pour l'instant — la méthode points nécessiterait son propre
recalibrage de `BORNES`/échelles, non fait), `decouper_train_val_once()`
(découpage par scène, pas de notion de sujet sur ONCE).

Vérifié sur données réelles (scène 000000) : 576 patches (8×4×18), 57,2%
d'occupation pixel (bien plus dense que MM-Fi ~10,7%, scène complète pas un
sujet isolé), gradients corrects, cible gelée à 0. Vérification visuelle :
structure cohérente d'une scène de conduite réelle (bandes de canaux
systématiquement occupées = route/sol, formes arrondies = objets proches,
canaux du haut pointant vers le ciel avec retours épars).

**Entraînement long terminé (2 septembre, 3000 pas, avant `perte_uniformite_poolee`)** :
poids → `checkpoints_once_grille/final.pt`. Perte train 0,98 → ~0,18-0,25 en
fin d'entraînement (bruitée, comme sur MM-Fi), similarité d'effondrement PAR
PATCH descendue à 0,07-0,16 (raisonnable). Similarité poolée non mesurée à
l'époque (diagnostic ajouté après coup) — donc ce checkpoint souffre
vraisemblablement du même effondrement poolé que les checkpoints MM-Fi
pré-correctif ; pas encore ré-entraîné avec `perte_uniformite_poolee` (coût
~5h20 pour 3000 pas sur `small`, contre ~5-10 min sur MM-Fi — bien plus lent,
nuages ~67k points vs quelques centaines sur MM-Fi).

Pas encore fait : ré-entraînement ONCE avec `perte_uniformite_poolee`,
combiner `small`+`medium`, évaluation croisée (checkpoint ONCE → linear
probing sur les labels MM-Fi, pour vérifier empiriquement si le
pré-entraînement sur scènes de conduite transfère à la reconnaissance
d'activité) — nécessite de projeter MM-Fi avec `PROFIL_ONCE` plutôt que
`PROFIL_MMFI` pour que les formes du tokenizer correspondent (le noyau de
convolution dépend de `noyau_spatial`, différent entre les deux profils :
10 vs 16 — un chargement direct du state_dict échouerait), canal d'intensité,
méthode points pour ONCE.

## Refactorisation (11 septembre) — protocole commun + consolidation

Déclenchée par un audit du réglage du SVM (`tuner_pointmae_classique.py`).

**Le réglage lui-même est sain** — vérifié point par point : `GridSearchCV`
ne voit que le train (`gs.fit(X_train, y_train, groups=groupes_train)`), la
validation ne sert qu'à un `predict` final ; `GroupKFold` est groupé par
sujet, donc les fenêtres corrélées d'une même séquence ne traversent pas les
folds ; le `StandardScaler` est dans le `Pipeline`, donc refité par fold ;
les sujets train/val sont disjoints (32 / 8). Le `cv_acc=0.920` contre
`VAL=0.949` est cohérent (les folds s'entraînent sur 4/5 des sujets, le
modèle final sur 32). **Le 0,949 n'est pas gonflé par une fuite de réglage.**

**Mais l'audit a trouvé autre chose** : aucun des chiffres « split
environnement » de la branche Point-MAE n'était reproductible depuis le
dépôt.

| chiffre publié | script censé le produire | ce que le script faisait |
|---|---|---|
| tête linéaire, env : 0,908 ± 0,009 | `entrainer_pointmae_seeds.py` | split sujet seulement |
| Point-MAE + SVM, env : 0,926 | `tuner_pointmae_classique.py` | split sujet seulement |
| fusion + Laplacien, env : 0,937 | `tuner_fusion.py` | split sujet seulement |

Seuls `entrainer_laplacien_v4/v5.py` et `entrainer_temporal_jepa.py`
appelaient `decouper_train_val_environnement()`. Les trois valeurs
n'existaient qu'en prose dans
`Tracking/mois/3-septembre/5-recherche_architectures_5sept.txt` (lignes 317,
388, 620) : calculées par une édition ad hoc, jamais commitée, sans log.
C'est gênant parce que le split environnement est **le seul protocole
homogène avec la littérature** (arXiv:2509.23255) — la comparaison « on
dépasse leurs 81-90% » reposait entièrement sur ces trois chiffres-là.

**Corrigé** : les trois scripts passent maintenant par
`protocole.evaluer_deux_splits` et produisent systématiquement les deux
splits. ⚠️ **Les valeurs 0,908 / 0,926 / 0,937 restent à régénérer** (les
caches `.npz` sont gitignorés, donc l'exécution doit se faire sur la machine
qui les détient) avant d'être citées ailleurs.

Un changement de comportement à noter : `tuner_fusion.py` utilisait une
grille SVM plus petite que le reste du projet. Il passe sur `GRILLE_SVM`,
qui la contient strictement — le `cv_acc` retenu ne peut donc que monter ou
rester égal, mais le score de validation peut bouger de quelques millièmes.

**Consolidation** : 15 scripts remplacés par 3.

| supprimés | remplacés par |
|---|---|
| `diag_combine`, `diag_jrdb`, `diag_vjepa_mirror`, `diag_vjepa_mirror_reg`, `entrainer_combine`, `entrainer_combine_jrdb2x`, `entrainer_vjepa` | `lancer_jepa.py` (presets) |
| `eval_vjepa`, `eval_finetune_vjepa`, `eval_finetune_combine`, `eval_finetune_seeds`, `eval_finetune_jrdb2x_seeds`, `eval_finetune_jrdb2x_graine2` | `lancer_finetune.py` (presets + `--graines`) |
| `diag_uniformite_poolee`, `diag_variance_poolee` | `diag_effondrement_poole.py` (`--poids` explicite) |

Chacun n'était qu'un appel à `entrainement.entrainer` ou
`evaluation.probe_lineaire` avec d'autres arguments, **exécuté à l'import**
(aucune garde `__main__`) : importer le module lançait un entraînement.
`tracer_courbes.py` et `tracer_vjepa.py` avaient le même défaut — ils
réécrivaient leurs PNG à l'import — et sont désormais gardés eux aussi.

En passant, `diag_uniformite_poolee.py` et `diag_variance_poolee.py`
laissaient tous deux `poids_uniformite_poolee` à sa valeur par défaut (0.0)
et mesuraient donc un run **sans** régularisation, malgré leurs docstrings.
Le remplaçant prend le poids en argument explicite.

**Délibérément conservés** : la lignée Laplacien v1/v2/v3 et
`tuner_laplacien_v2/v3_pca` sont redondantes en apparence, mais elles seules
régénèrent le tableau d'ablation de la piste B (0,218 → 0,617 → 0,590). Les
supprimer reproduirait exactement le problème que cette refactorisation
corrige.

## Limites connues / points ouverts

- **Les trois chiffres split environnement de la branche Point-MAE
  (0,908 / 0,926 / 0,937) n'ont pas encore été régénérés** par le code
  corrigé du 11 septembre — à faire avant toute citation externe.
- Un seul ensemble de validation a servi à départager tout le projet
  (tête linéaire vs SVM vs RF vs HGB, puis les 6 add-ins JEPA, puis la
  fusion) : c'est donc un ensemble de DÉVELOPPEMENT, et 0,954 est une
  estimation légèrement optimiste. Un test tenu à l'écart reste à faire.
- z non référencé au sol (pas de calibration scène-vide disponible pour MM-Fi).
- Nombre de tokens différent entre les deux méthodes (192 vs 512) — assumé,
  pas corrigé (cf. `tokenisation.py` ci-dessus).
- Hyperparamètres (dim=192, profondeurs, seuil de variance, poids de perte)
  choisis pour un prototype testable sur GPU modeste (RTX 3050, 8 Go), pas
  pour un pré-entraînement à l'échelle visée par le plan d'action.
- Disque `D:` largement libéré depuis (ONCE et L-CAS supprimés par
  l'utilisateur début septembre, ~940 Go libres au 4 septembre) — ne
  contient plus que JRDB (`D:/jrdb/`, ~22 Go).
- Le tokenizer grille (conv3D) n'est PAS transférable directement entre profils
  de grille différents (le noyau de convolution dépend de `noyau_spatial`,
  cf. section ONCE) — seuls `encodeur_contexte`/`predicteur` (Transformers,
  indifférents à la taille de séquence) sont réellement partageables entre
  MM-Fi et ONCE en l'état.

## Exécuter

**Important** : lancer depuis la racine du dépôt (`LIARA_2026_internship/`),
pas depuis `mmfi-src/` — les chemins par défaut (`data/MMFi_lidar_only/...`,
`mmfi-src/checkpoints_.../`) sont relatifs à la racine.

### Pré-entraînement JEPA

```bash
python mmfi-src/lancer_jepa.py --liste          # presets disponibles
python mmfi-src/lancer_jepa.py vjepa            # MM-Fi seul, 3000 pas
python mmfi-src/lancer_jepa.py combine-jrdb2x   # MM-Fi + JRDB x2
python mmfi-src/entrainement.py grille mmfi     # appel direct, paramétrable
```

`entrainement.py` prend `methode` (`grille`|`points`) et `source`
(`mmfi`|`jrdb`|`mmfi_jrdb`|`mmfi_jrdb2x` ; `once` existe encore dans le code
mais le dataset a été supprimé de la machine).

### Évaluation en aval des checkpoints JEPA

```bash
python mmfi-src/lancer_finetune.py --liste
python mmfi-src/lancer_finetune.py jrdb2x               # 3 graines
python mmfi-src/lancer_finetune.py jrdb2x --graines 2   # reprise d'une graine
python mmfi-src/evaluation.py                           # probing, 4 conditions
```

### Pipelines de features gelées — toujours les DEUX splits

Toutes passent par `protocole.evaluer_deux_splits` : chaque exécution produit
le split sujet **et** le split environnement.

```bash
python mmfi-src/extraire_pointmae.py            # cache features_pointmae.npz
python mmfi-src/entrainer_pointmae_seeds.py     # tête linéaire, 3 graines
python mmfi-src/tuner_pointmae_classique.py     # SVM/RF/HGB (meilleur simple)
python mmfi-src/extraire_laplacien_v4.py        # cache features_laplacien_v4_all.npz
python mmfi-src/entrainer_laplacien_v4.py       # spectre Laplacien
python mmfi-src/tuner_fusion.py                 # fusion (meilleur absolu)
```

### Diagnostics

```bash
python mmfi-src/diag_effondrement_poole.py --poids 1.0
python mmfi-src/smoke_test_vjepa.py
python mmfi-src/tracer_courbes.py
```

### Données requises

`data/MMFi_lidar_only/` (sous-ensemble lidar+ground_truth de MM-Fi, cf.
`Tracking/mois/2-aout/`) et, pour les sources `jrdb`/`mmfi_jrdb`,
`D:/jrdb/train/pointclouds/lower_velodyne/`. Poids et historiques écrits dans
`mmfi-src/checkpoints_{suffixe}/` et
`mmfi-src/historique_entrainement_{suffixe}.csv`.

Les caches de features (`features_*.npz`) sont gitignorés : ils doivent être
régénérés par les scripts `extraire_*.py` sur la machine qui détient les
données.
