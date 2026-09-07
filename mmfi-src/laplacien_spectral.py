"""Pipeline classique (non deep-learning) de reconnaissance d'action a partir
du spectre du Laplacien combinatoire d'un graphe de proximite, inspire de
"LiDAR-based Human Activity Recognition through Laplacian Spectral Analysis"
(arXiv:2509.23255, 2025), qui rapporte 90,3% (27 classes) / 94,4% (13 classes
readaptation) sur MM-Fi -- tres au-dessus de tout ce qu'on a obtenu avec
l'approche deep learning (grille+transformer, ~0,51 en fine-tuning complet).
Code source du papier non disponible ("code coming soon" au 4 septembre) --
reimplementation a partir de la description methodologique (WebFetch de
l'article), SIMPLIFIEE pour tenir dans le temps disponible (une nuit) :

- Pas de partition en 4 quadrants spatiaux (papier : ameliore 61%->94% sur le
  13-classes) -- corps entier seulement, a ajouter si le temps le permet.
- K_VAL=30 valeurs propres (papier : 60), pas de statistiques de vecteurs
  propres (papier : +6pts sur le 13-classes) -- version "Strategie A" du
  papier, pas la version complete "+eigenvectors".
- Fenetre/stride adaptes a notre volume (4 fenetres/sequence, sous-
  echantillonnees a 8 frames/fenetre) plutot que le stride dense du papier.

DECOUVERTE IMPORTANTE (4 septembre matin) : les frames LiDAR brutes de MM-Fi
ne sont PAS deja recadrees sur la personne -- une frame contient ~1100-1200
points s'etalant sur plusieurs metres (mur/sol inclus), pas juste la
personne (~50-150 points typiquement). Le squelette verite-terrain est dans
un repere DIFFERENT du nuage LiDAR brut (pas de correspondance directe
verifiee -- tentative de recadrage par bounding-box squelette -> 0 points
en commun, cf. session), donc PAS utilise ici pour le recadrage. A la place :
soustraction de fond auto-contenue (voxels occupes dans >80% des frames
d'une sequence = statique = mur/sol, retires), puis DBSCAN pour isoler le
plus grand amas coherent (la personne, en mouvement) parmi les points
restants -- ne suppose rien sur le repere du squelette, applicable a
l'usage reel (aucune information privilegiee)."""

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

VOXEL_FOND = 0.10       # taille de voxel pour la detection de fond statique (m)
SEUIL_STATIQUE = 0.8    # fraction de frames occupees -> voxel considere statique
DBSCAN_EPS = 0.25        # rayon de voisinage DBSCAN pour isoler la personne (m)
DBSCAN_MIN_SAMPLES = 5
RAYON_GRAPHE = 0.20       # rayon du graphe de proximite (epsilon-graph), cf. papier r=0.20 (optimal)
K_VAL = 30                # nombre de valeurs propres du Laplacien conservees


def _voxels_statiques(sequence_points, voxel=VOXEL_FOND, seuil=SEUIL_STATIQUE):
    """sequence_points: liste de (N_i,3). Retourne le set des voxels (tuples
    d'indices entiers) occupes dans plus de `seuil` fraction des frames --
    fond statique (mur/sol/mobilier) a retirer de chaque frame."""
    nb_frames = len(sequence_points)
    compteur = {}
    for pts in sequence_points:
        if pts.shape[0] == 0:
            continue
        idx = np.floor(pts / voxel).astype(np.int64)
        uniques = np.unique(idx, axis=0)
        for u in map(tuple, uniques):
            compteur[u] = compteur.get(u, 0) + 1
    return {u for u, c in compteur.items() if c / nb_frames > seuil}


def _ransac_plan(points, seuil, iterations, rng):
    """Un plan par RANSAC (3 points -> normale+offset) -- retourne le
    masque des points DANS le plan (inliers, a retirer : sol/mur)."""
    n = points.shape[0]
    meilleur_masque = np.zeros(n, dtype=bool)
    meilleur_score = 0
    for _ in range(iterations):
        p0, p1, p2 = points[rng.choice(n, size=3, replace=False)]
        normale = np.cross(p1 - p0, p2 - p0)
        norme = np.linalg.norm(normale)
        if norme < 1e-8:
            continue
        normale = normale / norme
        distances = np.abs((points - p0) @ normale)
        masque = distances < seuil
        score = masque.sum()
        if score > meilleur_score:
            meilleur_score, meilleur_masque = score, masque
    return meilleur_masque


def _supprimer_plans(points, n_plans=2, seuil=0.05, iterations=200, fraction_min=0.05):
    """Retire iterativement jusqu'a n_plans plans dominants (sol/mur) par
    RANSAC -- s'arrete si le plan trouve est trop petit (fraction_min du
    nuage COURANT) pour ne pas retirer du bruit comme un "plan". Piste
    "isolation moins fidele que le papier" (recherche 5 septembre,
    Tracking/mois/3-septembre/4-laplacien_tuning_validation.txt) :
    reimplementation numpy de segment_plane (Open3D, INDISPONIBLE sur
    Python 3.13 cette plateforme, cf. README section JRDB) -- meme principe
    standard suggere comme pretraitement bon marche avant DBSCAN."""
    restants = points
    rng = np.random.default_rng(0)
    for _ in range(n_plans):
        if restants.shape[0] < 10:
            break
        masque_plan = _ransac_plan(restants, seuil, iterations, rng)
        if masque_plan.sum() < fraction_min * restants.shape[0]:
            break
        restants = restants[~masque_plan]
    return restants


def _retirer_bruit_statistique(points, k=20, ratio_ecart_type=2.0):
    """Retire les points dont la distance moyenne aux k plus proches
    voisins depasse moyenne+ratio*ecart-type (globales) -- reimplementation
    numpy de remove_statistical_outlier (Open3D)."""
    n = points.shape[0]
    if n <= k:
        return points
    arbre = cKDTree(points)
    distances, _ = arbre.query(points, k=k + 1)  # +1 : le point lui-meme, distance 0
    distances_moyennes = distances[:, 1:].mean(axis=1)
    seuil = distances_moyennes.mean() + ratio_ecart_type * distances_moyennes.std()
    return points[distances_moyennes <= seuil]


def _isoler_personne(points, voxels_statiques, voxel=VOXEL_FOND,
                      eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, nettoyage=False):
    """Retire le fond statique, optionnellement nettoie (RANSAC plans +
    bruit statistique, cf. ci-dessus -- desactive par defaut pour ne pas
    changer les resultats deja publies dans le README sans comparaison
    explicite, cf. extraire_fenetres_sequence(nettoyage=...)), puis isole
    le plus grand amas spatialement coherent (DBSCAN) -- la personne, en
    mouvement, contrairement au fond. Retourne (M,3) ou un tableau vide si
    aucun amas valide."""
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    idx = np.floor(points / voxel).astype(np.int64)
    masque_fg = np.array([tuple(k) not in voxels_statiques for k in idx])
    fg = points[masque_fg]
    if nettoyage and fg.shape[0] >= 10:
        fg = _supprimer_plans(fg)
        fg = _retirer_bruit_statistique(fg)
    if fg.shape[0] < min_samples:
        return np.zeros((0, 3), dtype=np.float32)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(fg)
    valides = labels[labels >= 0]
    if valides.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    plus_gros = np.bincount(valides).argmax()
    return fg[labels == plus_gros]


def _laplacien(points, rayon=RAYON_GRAPHE):
    """Graphe epsilon (arete si distance < rayon) -> Laplacien combinatoire
    L=D-A (n,n), ou None si n<2."""
    n = points.shape[0]
    if n < 2:
        return None
    arbre = cKDTree(points)
    paires = arbre.query_pairs(r=rayon, output_type="ndarray")
    A = np.zeros((n, n), dtype=np.float64)
    if paires.shape[0] > 0:
        A[paires[:, 0], paires[:, 1]] = 1.0
        A[paires[:, 1], paires[:, 0]] = 1.0
    D = np.diag(A.sum(axis=1))
    return D - A


def _valeurs_propres_laplacien(points, rayon=RAYON_GRAPHE, k=K_VAL):
    """Retourne les k plus petites valeurs propres (triees), complete par
    des zeros si le graphe a moins de k noeuds/composantes."""
    vec = np.zeros(k, dtype=np.float32)
    L = _laplacien(points, rayon)
    if L is None:
        return vec
    valeurs = np.linalg.eigvalsh(L)  # deja triees, croissantes
    m = min(k, L.shape[0])
    vec[:m] = valeurs[:m]
    return vec


K_VEC = 40   # nombre de vecteurs propres (non triviaux) dont on tire des statistiques
NB_STATS_VEC = 7  # moyenne, ecart-type, max, min, kurtosis, entropie de Shannon, etendue


def _stats_vecteur_propre(u):
    """Vecteur propre u (n,), norme unite (sortie de eigh) -> 7 statistiques,
    cf. papier (arXiv:2509.23255) : moyenne, ecart-type, max, min, kurtosis
    (exces), entropie de Shannon des composantes au carre (deja normalisees
    -- somme=1 car ||u||=1 -- interpretees comme une distribution de
    "localisation" du mode sur les points du graphe), etendue des valeurs
    absolues."""
    p = u ** 2
    p_nz = p[p > 1e-12]
    entropie = -np.sum(p_nz * np.log(p_nz)) if p_nz.size > 0 else 0.0
    ecart = u.std()
    kurtosis = (((u - u.mean()) ** 4).mean() / (ecart ** 4) - 3.0) if ecart > 1e-12 else 0.0
    return np.array([
        u.mean(), ecart, u.max(), u.min(), kurtosis, entropie,
        np.abs(u).max() - np.abs(u).min(),
    ], dtype=np.float32)


def _valeurs_et_vecteurs_propres(points, rayon=RAYON_GRAPHE, k_val=K_VAL, k_vec=K_VEC):
    """Comme _valeurs_propres_laplacien, + statistiques des k_vec premiers
    vecteurs propres NON TRIVIAUX (on saute le premier, associe a la valeur
    propre 0 -- vecteur constant pour une composante connexe, sans
    information de forme). Retourne (k_val + k_vec*NB_STATS_VEC,)."""
    dim = k_val + k_vec * NB_STATS_VEC
    out = np.zeros(dim, dtype=np.float32)
    L = _laplacien(points, rayon)
    if L is None:
        return out
    valeurs, vecteurs = np.linalg.eigh(L)  # valeurs triees croissantes, vecteurs colonnes
    n = L.shape[0]
    m_val = min(k_val, n)
    out[:m_val] = valeurs[:m_val]
    m_vec = min(k_vec, max(0, n - 1))
    for i in range(m_vec):
        stats = _stats_vecteur_propre(vecteurs[:, 1 + i])
        out[k_val + i * NB_STATS_VEC: k_val + (i + 1) * NB_STATS_VEC] = stats
    return out


def descripteur_pose(points):
    """Nuage de points (deja isole/recadre sur la personne) -> vecteur (K_VAL,)
    de valeurs propres du Laplacien -- descripteur de "forme" de la pose."""
    return _valeurs_propres_laplacien(points)


def _decouper_quadrants(points):
    """Points isoles (personne) -> 4 sous-nuages "lateral x vertical" (papier)
    -- dans notre convention de coordonnees, z est l'axe vertical (plage
    ~1.75m apres isolation de la personne, coherent avec une hauteur
    debout), y l'axe lateral (plus grande etendue que x parmi les axes
    horizontaux, cf. session)."""
    my_, mz = np.median(points[:, 1]), np.median(points[:, 2])
    return [
        points[(points[:, 1] >= my_) & (points[:, 2] >= mz)],
        points[(points[:, 1] >= my_) & (points[:, 2] < mz)],
        points[(points[:, 1] < my_) & (points[:, 2] >= mz)],
        points[(points[:, 1] < my_) & (points[:, 2] < mz)],
    ]


def descripteur_pose_quadrants(points):
    """Version enrichie (4 septembre, apres le premier resultat decevant
    v1=0,22) : partitionne les points en 4 quadrants spatiaux + corps entier
    = 5 "parties", concatene les K_VAL valeurs propres de chacune ->
    (5*K_VAL,). Le papier rapporte que cette partition est l'ablation la
    plus impactante (61%->94% sur le 13-classes, largement devant les
    statistiques de vecteurs propres) -- confirme sur nos donnees (SVM
    0,22->0,62, cf. README/tracking)."""
    entier = _valeurs_propres_laplacien(points)
    if points.shape[0] < 4:
        return np.concatenate([entier] + [np.zeros(K_VAL, dtype=np.float32)] * 4)
    parties = [entier] + [_valeurs_propres_laplacien(q) for q in _decouper_quadrants(points)]
    return np.concatenate(parties)


def descripteur_pose_quadrants_vecteurs(points):
    """Version complete (4 septembre, suite au succes des quadrants) :
    quadrants + statistiques de vecteurs propres (deuxieme ablation la plus
    impactante du papier, +6pts sur le 13-classes AU-DESSUS des quadrants
    seuls) -- (5*(K_VAL+K_VEC*NB_STATS_VEC),) = 5*310 = 1550 dims par frame,
    6200 apres agregation temporelle (4 stats)."""
    dim_partie = K_VAL + K_VEC * NB_STATS_VEC
    entier = _valeurs_et_vecteurs_propres(points)
    if points.shape[0] < 4:
        return np.concatenate([entier] + [np.zeros(dim_partie, dtype=np.float32)] * 4)
    parties = [entier] + [_valeurs_et_vecteurs_propres(q) for q in _decouper_quadrants(points)]
    return np.concatenate(parties)


STATS_TEMPORELLES = ("moyenne", "ecart_type", "min", "max")


def _stats_temporelles(matrice):
    """matrice: (T, D) -- une ligne de descripteur par frame echantillonnee.
    Retourne (4*D,) : moyenne/ecart-type/min/max par dimension sur T."""
    return np.concatenate([
        matrice.mean(axis=0), matrice.std(axis=0),
        matrice.min(axis=0), matrice.max(axis=0),
    ]).astype(np.float32)


def extraire_fenetres_sequence(sequence_points, longueur_fenetre=40, nb_fenetres=4,
                                nb_frames_echantillon=8, quadrants=False, vecteurs=False,
                                nettoyage=False):
    """sequence_points : liste de (N_i,3), toutes les frames d'une sequence.
    Soustraction de fond calculee UNE FOIS sur toute la sequence (le fond est
    stable pendant tout l'enregistrement), puis `nb_fenetres` fenetres de
    `longueur_fenetre` frames reparties dans la sequence, sous-echantillonnees
    a `nb_frames_echantillon` frames chacune pour le calcul des descripteurs
    (cout domine par la decomposition spectrale, cf. docstring module).
    `nettoyage` : RANSAC (retrait de plans sol/mur) + retrait de bruit
    statistique AVANT DBSCAN dans l'isolation de la personne (cf.
    _isoler_personne) -- desactive par defaut, comparaison explicite requise
    (extraire_laplacien_v5.py) avant de changer la reference du README.
    Retourne une liste de vecteurs features (4*K_VAL,), un par fenetre."""
    nb_total = len(sequence_points)
    if nb_total < longueur_fenetre:
        longueur_fenetre = nb_total
        nb_frames_echantillon = min(nb_frames_echantillon, nb_total)

    voxels_statiques = _voxels_statiques(sequence_points)

    debut_max = nb_total - longueur_fenetre
    if nb_fenetres > 1 and debut_max > 0:
        debuts = np.linspace(0, debut_max, nb_fenetres).astype(int)
    else:
        debuts = np.zeros(nb_fenetres, dtype=int)

    if vecteurs:
        fn_descripteur = descripteur_pose_quadrants_vecteurs
    elif quadrants:
        fn_descripteur = descripteur_pose_quadrants
    else:
        fn_descripteur = descripteur_pose

    features_fenetres = []
    for debut in debuts:
        indices_frames = np.linspace(debut, debut + longueur_fenetre - 1, nb_frames_echantillon).astype(int)
        descripteurs = []
        for i in indices_frames:
            personne = _isoler_personne(sequence_points[i], voxels_statiques, nettoyage=nettoyage)
            descripteurs.append(fn_descripteur(personne))
        features_fenetres.append(_stats_temporelles(np.stack(descripteurs)))

    return features_fenetres
