#!/usr/bin/env python3
"""Squelettisation 3D temps reel d'une personne vue par le RoboSense Airy.

Methode : LiveHPS (Ren et al., CVPR 2024, code MIT,
https://github.com/4DVLab/LiveHPS), estimation de pose 3D par LiDAR seul,
temps reel, poids pre-entraines publics. Le code de LiveHPS n'est pas copie
dans ce depot : il est clone dans third_party/LiveHPS (ignore par git).
Le reseau prend les points de la personne seule (256 points, centres) sur une
fenetre de 32 images et donne 24 articulations SMPL par image. Il est
entierement en PyTorch : il tourne sur CPU, et sur GPU NVIDIA s'il y en a un.

Etapes, a chaque nuage /rslidar_points :

1. Soustraction de fond. Au demarrage, le script enregistre la scene vide
   (distance de fond par pixel laser x azimut, comme la projection 2D). Un
   point appartient au premier plan s'il est nettement plus proche que le fond
   dans sa direction.
2. Segmentation. Les points du premier plan sont regroupes par proximite ; le
   groupe de taille humaine le plus proche de la personne suivie (ou le plus
   gros) est garde.
3. Reseau. Les points de la personne sont centres puis reduits a 256 points
   (echantillonnage par point le plus eloigne, comme a l'entrainement), et
   ajoutes a la fenetre glissante des 32 dernieres images.
4. Sortie. Le squelette de l'image la plus recente est publie pour RViz
   (/squelette/marqueurs), avec les points de la personne
   (/squelette/personne), et enregistre a la fermeture (.npz).

Le sol est detecte dans la scene vide : il sert a aligner l'axe vertical
(le reseau attend z vers le haut) et donne la hauteur du lidar, utile pour le
recentrage z = z_brut - z_sol prevu dans
Tracking/mois/2-aout/5-representation_finale_projection_recentrage.txt.
"""

import argparse
from collections import deque
from datetime import datetime
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import rclpy
import torch

from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray


RACINE_DEPOT = Path(__file__).resolve().parents[3]

# ----------------------------------------------------------------------
# Reglages
# ----------------------------------------------------------------------

# Topic publie par rslidar_sdk.
TOPIC_NUAGE = "/rslidar_points"

# Code et poids de LiveHPS, a installer a la main (cf. Tracking/mois/
# 5-novembre/2-squelettisation_temps_reel.txt) :
#   git clone https://github.com/4DVLab/LiveHPS.git third_party/LiveHPS
#   poids livehps.t7 (Google Drive du projet LiveHPS) dans poids/
DOSSIER_LIVEHPS = RACINE_DEPOT / "third_party" / "LiveHPS"
POIDS_LIVEHPS = Path(__file__).resolve().parent / "poids" / "livehps.t7"

# Fenetre temporelle du reseau, en images (32 a l'entrainement, soit 3,2 s).
FENETRE = 32
# Nombre de points de la personne en entree du reseau (256 a l'entrainement).
NOMBRE_POINTS = 256

# Orientation de montage du lidar : rotations (roulis, tangage, lacet) en
# degres qui ramenent le repere du lidar a un repere ou z pointe vers le haut.
# (0, 0, 0) : Airy pose a plat. Le sol detecte corrige ensuite les petites
# inclinaisons (jusqu'a 25 degres).
ROTATION_CAPTEUR_DEG = (0.0, 0.0, 0.0)

# Grille de la soustraction de fond : laser x azimut, comme la projection 2D.
HAUTEUR = 96
LARGEUR = 900

# Scene vide : attente avant capture (le temps de sortir du champ), puis
# nombre d'images utilisees pour estimer le fond.
DELAI_AVANT_FOND = 5.0
IMAGES_FOND = 30
FICHIER_FOND = "fond_airy.npz"

# Un point est au premier plan s'il est plus proche que le fond d'au moins
# MARGE_FOND metres dans sa direction.
MARGE_FOND = 0.20
DISTANCE_MAX_PERSONNE = 8.0

# Segmentation : voxels de 5 cm, relies s'ils sont a moins de 20 cm.
TAILLE_VOXEL = 0.05
RAYON_GROUPE = 0.20
POINTS_MIN_PERSONNE = 60
HAUTEUR_PERSONNE = (0.3, 2.4)
LARGEUR_MAX_PERSONNE = 1.6
DISTANCE_SUIVI = 1.0
IMAGES_PERDUES_MAX = 5

# Intervalle entre deux lignes d'etat, en secondes.
PERIODE_ETAT = 1.0

# 24 articulations SMPL, dans l'ordre de sortie du reseau.
ARTICULATIONS = (
    "bassin", "hanche_g", "hanche_d", "colonne_1", "genou_g", "genou_d",
    "colonne_2", "cheville_g", "cheville_d", "colonne_3", "pied_g", "pied_d",
    "cou", "clavicule_g", "clavicule_d", "tete", "epaule_g", "epaule_d",
    "coude_g", "coude_d", "poignet_g", "poignet_d", "main_g", "main_d",
)
OS = (
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23),
)


# ----------------------------------------------------------------------
# Geometrie
# ----------------------------------------------------------------------

def matrice_rotation(roulis, tangage, lacet):
    """Rotation (degres) autour de x, puis y, puis z."""

    a, b, c = np.deg2rad([roulis, tangage, lacet])
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def rotation_vers_z(normale):
    """Rotation qui amene le vecteur unitaire 'normale' sur l'axe z."""

    z = np.array([0.0, 0.0, 1.0])
    axe = np.cross(normale, z)
    sinus = np.linalg.norm(axe)
    cosinus = float(np.dot(normale, z))
    if sinus < 1e-8:
        return np.eye(3)
    k = axe / sinus
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + sinus * kx + (1.0 - cosinus) * (kx @ kx)


def estimer_sol(points, rng, iterations=300, tolerance=0.03, angle_max_deg=25.0):
    """Cherche le plan du sol dans la scene vide (repere z vers le haut).

    RANSAC sur les plans presque horizontaux, en ne gardant que ceux qui ont
    presque tous les points au-dessus d'eux (le sol, pas un plafond ni une
    table). Retourne (normale, hauteur_du_sol, nombre_de_points_du_sol) ou None.
    """

    if len(points) > 20000:
        points = points[rng.choice(len(points), 20000, replace=False)]
    if len(points) < 100:
        return None

    cos_max = np.cos(np.deg2rad(angle_max_deg))
    meilleur = None
    for _ in range(iterations):
        p0, p1, p2 = points[rng.choice(len(points), 3, replace=False)]
        normale = np.cross(p1 - p0, p2 - p0)
        norme = np.linalg.norm(normale)
        if norme < 1e-9:
            continue
        normale = normale / norme
        if normale[2] < 0:
            normale = -normale
        if normale[2] < cos_max:
            continue
        hauteur = points @ normale - np.dot(normale, p0)
        sur_le_plan = np.abs(hauteur) < tolerance
        au_dessus = np.mean(hauteur > -tolerance)
        nombre = int(np.count_nonzero(sur_le_plan))
        if au_dessus > 0.95 and (meilleur is None or nombre > meilleur[0]):
            meilleur = (nombre, sur_le_plan)

    if meilleur is None or meilleur[0] < 200:
        return None

    # Ajustement final par moindres carres sur les points du sol.
    sol = points[meilleur[1]]
    centre = sol.mean(axis=0)
    _, _, vt = np.linalg.svd(sol - centre, full_matrices=False)
    normale = vt[2] if vt[2][2] > 0 else -vt[2]
    return normale, float(np.dot(normale, centre)), meilleur[0]


def indices_pixels(x, y, z, ring):
    """Pixel (ligne, colonne) de chaque point, dans la grille laser x azimut.

    Sans champ 'ring', la ligne est l'angle vertical par pas de 1 degre."""

    colonne = np.floor(0.5 * (-np.arctan2(y, x) / np.pi + 1.0) * LARGEUR)
    colonne = np.clip(colonne.astype(np.int64), 0, LARGEUR - 1)
    if ring is not None:
        ligne = np.clip(ring.astype(np.int64), 0, HAUTEUR - 1)
        hauteur = HAUTEUR
    else:
        distance = np.sqrt(x * x + y * y + z * z)
        pitch = np.degrees(np.arcsin(np.clip(z / np.maximum(distance, 1e-6), -1.0, 1.0)))
        ligne = np.clip(np.floor(pitch + 90.0).astype(np.int64), 0, 179)
        hauteur = 180
    return ligne * LARGEUR + colonne, hauteur


def distance_min_par_pixel(pixel, distance, hauteur):
    """Image (hauteur, LARGEUR) de la distance la plus courte par pixel (inf si vide)."""

    image = np.full(hauteur * LARGEUR, np.inf, dtype=np.float32)
    np.minimum.at(image, pixel, distance.astype(np.float32))
    return image.reshape(hauteur, LARGEUR)


def eroder_fond(fond):
    """Minimum du fond sur chaque pixel et ses 8 voisins.

    D'un tour a l'autre, un meme point du decor peut tomber dans la colonne
    voisine : comparer au minimum local evite les faux premiers plans sur les
    bords des objets."""

    resultat = fond.copy()
    for dl in (-1, 0, 1):
        decale = np.roll(fond, dl, axis=0)
        if dl == -1:
            decale[-1] = np.inf
        elif dl == 1:
            decale[0] = np.inf
        for dc in (-1, 0, 1):
            resultat = np.minimum(resultat, np.roll(decale, dc, axis=1))
    return resultat


def regrouper(points):
    """Etiquette de groupe de chaque point (voisins a moins de RAYON_GROUPE)."""

    cles = np.floor(points / TAILLE_VOXEL).astype(np.int64)
    _, inverse = np.unique(cles, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    nombre_voxels = int(inverse.max()) + 1

    # Centre de chaque voxel occupe.
    centres = np.zeros((nombre_voxels, 3))
    np.add.at(centres, inverse, points)
    centres /= np.bincount(inverse, minlength=nombre_voxels)[:, None]

    paires = cKDTree(centres).query_pairs(RAYON_GROUPE, output_type="ndarray")
    graphe = coo_matrix(
        (np.ones(len(paires)), (paires[:, 0], paires[:, 1])),
        shape=(nombre_voxels, nombre_voxels),
    )
    _, etiquettes = connected_components(graphe, directed=False)
    return etiquettes[inverse]


def choisir_personne(points, centre_precedent):
    """Garde le groupe de taille humaine le plus pertinent, ou None."""

    if len(points) < POINTS_MIN_PERSONNE:
        return None

    etiquettes = regrouper(points)
    candidats = []
    for etiquette in np.unique(etiquettes):
        groupe = points[etiquettes == etiquette]
        if len(groupe) < POINTS_MIN_PERSONNE:
            continue
        etendue = groupe.max(axis=0) - groupe.min(axis=0)
        if not HAUTEUR_PERSONNE[0] <= etendue[2] <= HAUTEUR_PERSONNE[1]:
            continue
        if np.hypot(etendue[0], etendue[1]) > LARGEUR_MAX_PERSONNE:
            continue
        candidats.append(groupe)

    if not candidats:
        return None

    # Suivi : le groupe le plus proche de la personne de l'image precedente.
    if centre_precedent is not None:
        distances = [np.linalg.norm(g.mean(axis=0) - centre_precedent) for g in candidats]
        plus_proche = int(np.argmin(distances))
        if distances[plus_proche] < DISTANCE_SUIVI:
            return candidats[plus_proche]

    return max(candidats, key=len)


def echantillonner_fps(points, nombre, rng):
    """Echantillonnage par point le plus eloigne, comme a l'entrainement de
    LiveHPS (dataset/Livehps_dataset.py). Moins de 'nombre' points : repetes."""

    if len(points) < nombre:
        repetitions = int(np.ceil(nombre / len(points)))
        return np.tile(points, (repetitions, 1))[:nombre]
    if len(points) > 8 * nombre:
        points = points[rng.choice(len(points), 8 * nombre, replace=False)]

    choisis = np.empty(nombre, dtype=np.int64)
    distance = np.full(len(points), np.inf)
    courant = int(rng.integers(len(points)))
    for i in range(nombre):
        choisis[i] = courant
        distance = np.minimum(distance, np.sum((points - points[courant]) ** 2, axis=1))
        courant = int(np.argmax(distance))
    return points[choisis]


# ----------------------------------------------------------------------
# Reseau LiveHPS
# ----------------------------------------------------------------------

class EstimateurLiveHPS:
    """LiveHPS sur une fenetre glissante, image par image.

    Les caracteristiques PointNet de chaque image sont calculees une seule
    fois et gardees dans la fenetre : a chaque nouvelle image, seul le nouveau
    nuage passe dans PointNet. Les etapes temporelles (GRU, transformers)
    sont recalculees sur toute la fenetre, comme dans le modele d'origine.
    """

    def __init__(self, dossier_livehps, chemin_poids, appareil):
        sys.path.insert(0, str(dossier_livehps))
        from Model.livehps import CPO, PBT, SMPL_trans

        k = len(ARTICULATIONS)
        modele = torch.nn.Module()
        modele.model1 = PBT(k=k * 3, channel=3)
        modele.model2 = SMPL_trans(in_c=3 + 1024)
        modele.model3 = CPO(in_c=1024 + 72, hidden=256 + 3 + 1024, out_c=3, kp_num=k)
        modele.model4 = CPO(in_c=1024 + 72, hidden=256 + 3 + 1024, out_c=6, kp_num=k)
        modele.model5 = CPO(in_c=1024 + 72, hidden=256 + 3 + 1024, out_c=6, kp_num=k, shape=True)

        # Meme filtrage des cles que load_GPUS() dans LiveHPS/models.py.
        etat = torch.load(chemin_poids, map_location="cpu")
        etat_filtre = {}
        for cle, valeur in etat.items():
            if "sa2" in cle.split("."):
                continue
            etat_filtre[cle[7:] if cle.startswith("module") else cle] = valeur
        modele.load_state_dict(etat_filtre)

        self.modele = modele.to(appareil).eval()
        self.appareil = appareil
        self.caracteristiques = deque(maxlen=FENETRE)
        self.nuages = deque(maxlen=FENETRE)

    def vider(self):
        self.caracteristiques.clear()
        self.nuages.clear()

    @torch.no_grad()
    def ajouter_image(self, nuage_centre):
        """nuage_centre : (NOMBRE_POINTS, 3), centre sur sa moyenne."""

        nuage = torch.from_numpy(nuage_centre.astype(np.float32)).to(self.appareil)
        caracteristique = self.modele.model1.encode(nuage.T.unsqueeze(0))  # (1, 1024)
        self.caracteristiques.append(caracteristique[0])
        self.nuages.append(nuage)

    @torch.no_grad()
    def estimer(self):
        """Articulations (24, 3) de la derniere image, dans le repere du nuage
        centre (bassin = translation estimee par le reseau)."""

        manque = FENETRE - len(self.caracteristiques)
        caracteristiques = [self.caracteristiques[0]] * manque + list(self.caracteristiques)
        nuages = [self.nuages[0]] * manque + list(self.nuages)
        x = torch.stack(caracteristiques).unsqueeze(0)  # (1, T, 1024)
        nuages = torch.stack(nuages).unsqueeze(0)  # (1, T, 256, 3)

        # Fin de PBT.forward (LiveHPS/Model/livehps.py), PointNet deja fait.
        pbt = self.modele.model1
        h0 = torch.zeros(2, 1, 512, device=self.appareil)
        sortie, _ = pbt.gru(x, h0)
        y = torch.relu(pbt.fc1(sortie))
        y = torch.relu(pbt.dropout(pbt.fc2(y)))
        articulations = pbt.fc3(y)  # (1, T, 72)

        # Suite de ModelNet_shape.forward (LiveHPS/models.py) : affinage des
        # articulations (CPO) puis position du bassin (SMPL_trans).
        nouvelles, _, _ = self.modele.model3(articulations, sortie)
        nouvelles = nouvelles.reshape(1, FENETRE, -1)
        translation = self.modele.model2(nuages, nouvelles, sortie.reshape(1, FENETRE, -1))

        derniere = nouvelles[0, -1].reshape(-1, 3)
        bassin = translation.reshape(FENETRE, 3)[-1]
        # Squelette relatif au bassin, place a la position estimee du bassin.
        return (derniere - derniere[0] + bassin).cpu().numpy()


# ----------------------------------------------------------------------
# Noeud ROS 2
# ----------------------------------------------------------------------

class SqueletteAiry(Node):

    def __init__(self, estimateur, nouveau_fond):
        super().__init__("squelette_robosense_airy")

        self.estimateur = estimateur
        self.rng = np.random.default_rng(0)
        self.rotation_grossiere = matrice_rotation(*ROTATION_CAPTEUR_DEG)
        self.rotation = self.rotation_grossiere
        self.hauteur_sol = None

        # Reception dans le fil ROS 2 : seul le dernier nuage est garde.
        self.verrou = threading.Lock()
        self.dernier_message = None
        self.nombre_recus = 0
        self.create_subscription(
            PointCloud2, TOPIC_NUAGE, self.recevoir_nuage, qos_profile_sensor_data
        )
        self.publieur_marqueurs = self.create_publisher(MarkerArray, "/squelette/marqueurs", 10)
        self.publieur_personne = self.create_publisher(PointCloud2, "/squelette/personne", 10)

        # Scene vide : chargee si deja enregistree, sinon capturee.
        self.fond = None
        self.fond_erode = None
        self.images_fond = []
        self.points_fond = None
        self.debut_attente = time.monotonic()
        if not nouveau_fond and Path(FICHIER_FOND).exists():
            self.charger_fond()
        else:
            print(
                f"\nCapture de la scene vide dans {DELAI_AVANT_FOND:.0f} s : "
                "sortez du champ du lidar."
            )

        # Suivi de la personne et enregistrement.
        self.centre_precedent = None
        self.images_perdues = 0
        self.enregistrement_temps = []
        self.enregistrement_articulations = []

        # Mesures pour la ligne d'etat.
        self.dernier_etat = time.monotonic()
        self.recus_au_dernier_etat = 0
        self.nombre_squelettes = 0
        self.points_personne = 0
        self.duree_segmentation = 0.0
        self.duree_reseau = 0.0

    # --- fil ROS 2 ---

    def recevoir_nuage(self, message):
        with self.verrou:
            self.dernier_message = message
            self.nombre_recus += 1

    def prendre_dernier_message(self):
        with self.verrou:
            message, self.dernier_message = self.dernier_message, None
        return message

    # --- scene vide ---

    def charger_fond(self):
        donnees = np.load(FICHIER_FOND)
        self.fond = donnees["fond"]
        self.rotation = donnees["rotation"]
        hauteur_sol = float(donnees["hauteur_sol"])
        self.hauteur_sol = None if np.isnan(hauteur_sol) else hauteur_sol
        self.fond_erode = eroder_fond(self.fond)
        print(f"\nScene vide chargee depuis {Path(FICHIER_FOND).resolve()}")
        print("(relancer avec --nouveau-fond si le lidar ou le decor a bouge)")
        self.decrire_sol()

    def capturer_fond(self, pixel, distance, hauteur, points):
        attente = time.monotonic() - self.debut_attente
        if attente < DELAI_AVANT_FOND:
            return
        if not self.images_fond:
            print(f"Capture de la scene vide ({IMAGES_FOND} images)...")
        self.images_fond.append(distance_min_par_pixel(pixel, distance, hauteur))
        self.points_fond = points
        if len(self.images_fond) < IMAGES_FOND:
            return

        # Distance de fond par pixel : mediane des images ou le pixel a un
        # retour (inf s'il n'en a jamais).
        pile = np.stack(self.images_fond)
        pile[np.isinf(pile)] = np.nan
        valides = np.isfinite(pile).any(axis=0)
        fond = np.full(pile.shape[1:], np.inf, dtype=np.float32)
        fond[valides] = np.nanmedian(pile[:, valides], axis=0)
        self.fond = fond
        self.fond_erode = eroder_fond(fond)
        self.images_fond = []

        # Sol : aligne l'axe vertical et donne la hauteur du lidar.
        points_haut = self.points_fond @ self.rotation_grossiere.T
        sol = estimer_sol(points_haut, self.rng)
        if sol is None:
            self.rotation = self.rotation_grossiere
            self.hauteur_sol = None
        else:
            normale, hauteur_sol, _ = sol
            correction = rotation_vers_z(normale)
            self.rotation = correction @ self.rotation_grossiere
            self.hauteur_sol = hauteur_sol

        np.savez(
            FICHIER_FOND,
            fond=self.fond,
            rotation=self.rotation,
            hauteur_sol=np.nan if self.hauteur_sol is None else self.hauteur_sol,
        )
        print(f"Scene vide enregistree dans {Path(FICHIER_FOND).resolve()}")
        self.decrire_sol()
        print("Vous pouvez entrer dans le champ du lidar.\n")

    def decrire_sol(self):
        if self.hauteur_sol is None:
            print(
                "Sol non trouve dans la scene vide : axe vertical = "
                f"ROTATION_CAPTEUR_DEG {ROTATION_CAPTEUR_DEG}."
            )
            return
        inclinaison = np.degrees(np.arccos(np.clip(
            (self.rotation @ np.linalg.inv(self.rotation_grossiere))[2, 2], -1.0, 1.0
        )))
        print(
            f"Sol trouve : lidar a {-self.hauteur_sol:.2f} m au-dessus du sol, "
            f"inclinaison corrigee {inclinaison:.1f} degres."
        )

    # --- traitement ---

    def traiter_nuage(self, message):
        debut = time.perf_counter()

        a_ring = any(champ.name == "ring" for champ in message.fields)
        champs = ["x", "y", "z"] + (["ring"] if a_ring else [])
        donnees = point_cloud2.read_points(message, field_names=champs, skip_nans=True)
        x = donnees["x"].astype(np.float32)
        y = donnees["y"].astype(np.float32)
        z = donnees["z"].astype(np.float32)
        distance = np.sqrt(x * x + y * y + z * z)
        utiles = np.isfinite(distance) & (distance > 0.05)
        x, y, z, distance = x[utiles], y[utiles], z[utiles], distance[utiles]
        ring = donnees["ring"][utiles] if a_ring else None

        pixel, hauteur = indices_pixels(x, y, z, ring)
        points_lidar = np.stack([x, y, z], axis=1)

        if self.fond is None:
            self.capturer_fond(pixel, distance, hauteur, points_lidar)
            return
        if self.fond.shape[0] != hauteur:
            print("La scene vide ne correspond pas au format du nuage : nouvelle capture.")
            self.fond = None
            self.debut_attente = time.monotonic()
            return

        # 1. Soustraction de fond.
        fond = self.fond_erode.reshape(-1)[pixel]
        premier_plan = (distance < fond - MARGE_FOND) & (distance < DISTANCE_MAX_PERSONNE)

        # 2. Segmentation, dans le repere z vers le haut.
        points = points_lidar[premier_plan] @ self.rotation.T
        personne = choisir_personne(points, self.centre_precedent)
        milieu = time.perf_counter()
        self.duree_segmentation = 0.8 * self.duree_segmentation + 0.2 * (milieu - debut)

        if personne is None:
            self.images_perdues += 1
            if self.images_perdues > IMAGES_PERDUES_MAX:
                self.centre_precedent = None
                self.estimateur.vider()
            self.points_personne = 0
            return
        self.images_perdues = 0
        self.points_personne = len(personne)

        # 3. Reseau : nuage centre, 256 points, fenetre glissante.
        centre = personne.mean(axis=0)
        self.centre_precedent = centre
        self.estimateur.ajouter_image(echantillonner_fps(personne - centre, NOMBRE_POINTS, self.rng))
        articulations = self.estimateur.estimer() + centre
        fin = time.perf_counter()
        self.duree_reseau = 0.8 * self.duree_reseau + 0.2 * (fin - milieu)
        self.nombre_squelettes += 1

        # 4. Sorties : enregistrement (repere z vers le haut) et RViz (repere lidar).
        self.enregistrement_temps.append(
            message.header.stamp.sec + 1e-9 * message.header.stamp.nanosec
        )
        self.enregistrement_articulations.append(articulations.astype(np.float32))
        self.publier(message.header, articulations @ self.rotation, personne @ self.rotation)

    def publier(self, entete, articulations, personne):
        """Squelette et points de la personne, dans le repere du lidar."""

        duree_vie = Duration(seconds=0.3).to_msg()

        sphere = Marker()
        sphere.header = entete
        sphere.ns = "squelette"
        sphere.id = 0
        sphere.type = Marker.SPHERE_LIST
        sphere.action = Marker.ADD
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.06
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 1.0, 0.35, 0.1, 1.0
        sphere.lifetime = duree_vie
        sphere.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in articulations]

        lignes = Marker()
        lignes.header = entete
        lignes.ns = "squelette"
        lignes.id = 1
        lignes.type = Marker.LINE_LIST
        lignes.action = Marker.ADD
        lignes.pose.orientation.w = 1.0
        lignes.scale.x = 0.025
        lignes.color.r, lignes.color.g, lignes.color.b, lignes.color.a = 0.1, 0.9, 0.3, 1.0
        lignes.lifetime = duree_vie
        for a, b in OS:
            lignes.points.append(sphere.points[a])
            lignes.points.append(sphere.points[b])

        self.publieur_marqueurs.publish(MarkerArray(markers=[sphere, lignes]))
        self.publieur_personne.publish(
            point_cloud2.create_cloud_xyz32(entete, personne.astype(np.float32))
        )

    def afficher_etat(self):
        maintenant = time.monotonic()
        duree = maintenant - self.dernier_etat
        if duree < PERIODE_ETAT or self.fond is None:
            return

        with self.verrou:
            nombre_recus = self.nombre_recus
        print(
            f"Recus : {(nombre_recus - self.recus_au_dernier_etat) / duree:4.1f} Hz | "
            f"Squelettes : {self.nombre_squelettes / duree:4.1f} Hz | "
            f"Personne : {self.points_personne:5d} pts | "
            f"Segmentation : {1000 * self.duree_segmentation:3.0f} ms | "
            f"Reseau : {1000 * self.duree_reseau:3.0f} ms"
        )
        self.dernier_etat = maintenant
        self.recus_au_dernier_etat = nombre_recus
        self.nombre_squelettes = 0

    def enregistrer(self):
        """Sequence de squelettes (repere z vers le haut, origine au lidar)."""

        if not self.enregistrement_articulations:
            return
        chemin = Path(datetime.now().strftime("squelettes_airy_%Y%m%d_%H%M%S.npz")).resolve()
        np.savez(
            chemin,
            articulations=np.stack(self.enregistrement_articulations),
            temps=np.array(self.enregistrement_temps),
            noms=np.array(ARTICULATIONS),
            os=np.array(OS),
            rotation=self.rotation,
            hauteur_sol=np.nan if self.hauteur_sol is None else self.hauteur_sol,
        )
        print(f"\n{len(self.enregistrement_articulations)} squelettes enregistres dans {chemin}")


def tourner(executeur):
    try:
        executeur.spin()
    except ExternalShutdownException:
        pass


def main():
    analyseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analyseur.add_argument("--nouveau-fond", action="store_true",
                           help="recapturer la scene vide meme si elle est deja enregistree")
    analyseur.add_argument("--poids", type=Path, default=POIDS_LIVEHPS,
                           help="chemin des poids LiveHPS (livehps.t7)")
    analyseur.add_argument("--livehps", type=Path, default=DOSSIER_LIVEHPS,
                           help="dossier du code LiveHPS (clone du depot 4DVLab/LiveHPS)")
    arguments, _ = analyseur.parse_known_args()

    if not (arguments.livehps / "Model" / "livehps.py").exists():
        sys.exit(
            f"Code LiveHPS introuvable dans {arguments.livehps}.\n"
            f"Lancer : git clone https://github.com/4DVLab/LiveHPS.git {DOSSIER_LIVEHPS}"
        )
    if not arguments.poids.exists():
        sys.exit(
            f"Poids LiveHPS introuvables : {arguments.poids}\n"
            "Telecharger livehps.t7 (cf. Tracking/mois/5-novembre/"
            "2-squelettisation_temps_reel.txt)."
        )

    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Chargement de LiveHPS sur {appareil}...")
    estimateur = EstimateurLiveHPS(arguments.livehps, arguments.poids, appareil)

    rclpy.init()
    noeud = SqueletteAiry(estimateur, arguments.nouveau_fond)

    executeur = SingleThreadedExecutor()
    executeur.add_node(noeud)
    fil_ros = threading.Thread(target=tourner, args=(executeur,), daemon=True)
    fil_ros.start()

    try:
        while rclpy.ok():
            message = noeud.prendre_dernier_message()
            if message is None:
                time.sleep(0.005)
            else:
                noeud.traiter_nuage(message)
            noeud.afficher_etat()
    except KeyboardInterrupt:
        pass

    noeud.enregistrer()
    if rclpy.ok():
        rclpy.shutdown()
    fil_ros.join(timeout=2.0)
    noeud.destroy_node()


if __name__ == "__main__":
    main()
