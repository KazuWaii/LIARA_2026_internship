#!/usr/bin/env python3
"""Projection 2D temps reel du nuage de points RoboSense Airy.

Meme principe que projection_unitree_l2_*.py (projection spherique, z-buffer,
affichage Matplotlib en direct, sauvegarde .npy), avec trois differences :

- L'Airy est un lidar mecanique : chaque message /rslidar_points contient deja
  un tour complet (360 degres). Il n'y a donc rien a accumuler, contrairement
  au Unitree L2 qui avait besoin de plusieurs nuages pour remplir l'image.

- Si rslidar_sdk publie le champ 'ring' (POINT_TYPE XYZIRT), la ligne de
  l'image est directement le numero du laser. La grille (canal laser, azimut)
  est alors exacte et ne depend plus du champ de vision vertical. Sans 'ring'
  (POINT_TYPE XYZI, valeur par defaut du SDK), la ligne est calculee a partir
  de l'angle vertical, comme pour le Unitree L2.

- L'image suit la representation retenue (cf. Tracking/mois/2-aout/
  5-representation_finale_projection_recentrage.txt) : plusieurs canaux par
  pixel (distance, x, y, z, intensite) et un canal de validite. Un pixel vide
  vaut 0 partout, et seul le canal de validite permet de le distinguer d'un
  point proche de l'origine. Les coordonnees x, y, z sont celles du repere du
  lidar : le recentrage (boite englobante du sujet, z reference au sol) se
  fait plus tard, une fois par fenetre temporelle.
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import rclpy

from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


# Topic publie par rslidar_sdk (ros_send_point_cloud_topic dans config.yaml).
TOPIC_NUAGE = "/rslidar_points"

# Taille de l'image.
# HAUTEUR = nombre de lasers de l'Airy : 96 (ou 48 / 192 selon le modele).
# LARGEUR = 900 correspond a la resolution native a 10 Hz : rs_driver decode
# un tir des 96 lasers toutes les 111,08 us, soit environ 900 tirs par tour,
# donc 0,4 degre par colonne. Une largeur plus faible regroupe les colonnes.
HAUTEUR = 96
LARGEUR = 900

# Champ de vision vertical, utilise uniquement si le champ 'ring' est absent.
# Le premier nuage recu affiche l'angle vertical reellement observe : ajuster
# ces deux valeurs en consequence.
FOV_HAUT_DEGRES = 90.0
FOV_BAS_DEGRES = 0.0

# Distances conservees, en metres (portee minimale de l'Airy : 0,1 m).
DISTANCE_MIN = 0.1
DISTANCE_MAX = 5.0

# Borne haute de l'echelle de couleur de l'intensite (affichage seulement).
INTENSITE_MAX = 100.0

# Nombre maximal de rafraichissements de la fenetre par seconde.
FREQUENCE_AFFICHAGE = 10.0

# Ordre des canaux de l'image produite, de forme (len(CANAUX), HAUTEUR, LARGEUR).
CANAUX = ("distance", "x", "y", "z", "intensite", "validite")
CANAL = {nom: indice for indice, nom in enumerate(CANAUX)}

# Fichier dans lequel la premiere image complete est enregistree.
FICHIER_SORTIE = "image_airy.npy"


def projeter_nuage(x, y, z, intensite, ring=None):
    """Projette un nuage dans une image multi-canaux.

    Retourne l'image, de forme (len(CANAUX), HAUTEUR, LARGEUR), et le nombre
    de points rejetes parce que leur numero de laser depasse HAUTEUR.
    """

    image = np.zeros((len(CANAUX), HAUTEUR, LARGEUR), dtype=np.float32)

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    intensite = np.asarray(intensite, dtype=np.float32)

    # Distance radiale de chaque point par rapport au lidar.
    distance = np.sqrt(x * x + y * y + z * z)

    # Suppression des points invalides ou hors de la plage choisie.
    valide = (
        np.isfinite(distance)
        & (distance >= DISTANCE_MIN)
        & (distance <= DISTANCE_MAX)
    )

    x = x[valide]
    y = y[valide]
    z = z[valide]
    intensite = intensite[valide]
    distance = distance[valide]
    if ring is not None:
        ring = np.asarray(ring)[valide]

    if distance.size == 0:
        return image, 0

    # Colonne : angle horizontal, avec la meme convention que pour le
    # Unitree L2 (l'avant du lidar, axe x, au centre de l'image).
    yaw = -np.arctan2(y, x)
    colonne = np.floor(0.5 * (yaw / np.pi + 1.0) * LARGEUR).astype(np.int64)
    colonne = np.clip(colonne, 0, LARGEUR - 1)

    if ring is not None:
        # rs_driver numerote les lasers par angle vertical croissant :
        # ring 0 est le laser le plus bas, il va donc sur la derniere ligne.
        ligne = HAUTEUR - 1 - ring.astype(np.int64)
        dans_image = (ligne >= 0) & (ligne < HAUTEUR)
        nombre_hors_image = int(np.count_nonzero(~dans_image))
    else:
        # Ligne : angle vertical, ramene dans le champ de vision choisi.
        pitch = np.arcsin(np.clip(z / distance, -1.0, 1.0))
        fov_haut = np.deg2rad(FOV_HAUT_DEGRES)
        fov_bas = np.deg2rad(FOV_BAS_DEGRES)

        dans_image = (pitch >= fov_bas) & (pitch <= fov_haut)
        ligne = np.floor(
            (1.0 - (pitch - fov_bas) / (fov_haut - fov_bas)) * HAUTEUR
        ).astype(np.int64)
        ligne = np.clip(ligne, 0, HAUTEUR - 1)
        nombre_hors_image = 0

    ligne = ligne[dans_image]
    colonne = colonne[dans_image]
    x = x[dans_image]
    y = y[dans_image]
    z = z[dans_image]
    intensite = intensite[dans_image]
    distance = distance[dans_image]

    if distance.size == 0:
        return image, nombre_hors_image

    # Z-buffer : si plusieurs points tombent dans le meme pixel (double
    # retour, colonnes regroupees), le plus proche est garde.
    # Tri par pixel puis par distance, et conservation du premier point de
    # chaque pixel : le resultat ne depend pas de l'ordre d'ecriture NumPy.
    pixel = ligne * LARGEUR + colonne
    ordre = np.lexsort((distance, pixel))
    pixel_trie = pixel[ordre]
    premier = np.ones(pixel_trie.size, dtype=bool)
    premier[1:] = pixel_trie[1:] != pixel_trie[:-1]
    garde = ordre[premier]

    lignes = ligne[garde]
    colonnes = colonne[garde]
    image[CANAL["distance"], lignes, colonnes] = distance[garde]
    image[CANAL["x"], lignes, colonnes] = x[garde]
    image[CANAL["y"], lignes, colonnes] = y[garde]
    image[CANAL["z"], lignes, colonnes] = z[garde]
    image[CANAL["intensite"], lignes, colonnes] = intensite[garde]
    image[CANAL["validite"], lignes, colonnes] = 1.0

    return image, nombre_hors_image


class ProjectionAiry(Node):

    def __init__(self):
        super().__init__("projection_robosense_airy")

        # Profil "capteur" (best effort) : on prefere perdre une image que
        # d'accumuler du retard. Compatible avec l'editeur fiable du SDK.
        self.create_subscription(
            PointCloud2,
            TOPIC_NUAGE,
            self.lire_nuage,
            qos_profile_sensor_data,
        )

        self.premier_nuage_recu = False
        self.alerte_hors_image = False
        self.premiere_image_sauvegardee = False
        self.derniere_mise_a_jour = 0.0

        # Creation de la fenetre d'affichage en direct.
        plt.ion()

        self.figure, (self.axes_distance, self.axes_intensite) = plt.subplots(
            2, 1, figsize=(12, 6), sharex=True
        )

        image_vide = np.ma.masked_all((HAUTEUR, LARGEUR), dtype=np.float32)
        self.vue_distance = self.creer_vue(
            self.axes_distance,
            image_vide,
            "turbo",
            DISTANCE_MIN,
            DISTANCE_MAX,
            "Distance (m)",
        )
        self.vue_intensite = self.creer_vue(
            self.axes_intensite,
            image_vide,
            "gray",
            0.0,
            INTENSITE_MAX,
            "Intensite",
        )

        self.axes_distance.set_title("Image de profondeur RoboSense Airy")
        self.axes_intensite.set_xlabel("Angle horizontal (colonne)")
        self.figure.tight_layout()
        plt.show(block=False)

        self.get_logger().info(
            f"En attente des nuages de points sur {TOPIC_NUAGE}..."
        )

    def creer_vue(self, axes, image, nom_couleurs, vmin, vmax, legende):
        """Ajoute une image a la fenetre, les pixels vides en noir."""

        couleurs = plt.get_cmap(nom_couleurs).copy()
        couleurs.set_bad("black")

        vue = axes.imshow(
            image,
            cmap=couleurs,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        axes.set_ylabel("Laser (ligne)")
        self.figure.colorbar(vue, ax=axes, label=legende)
        return vue

    def afficher_image(self, image, nombre_pixels):
        """Met a jour la fenetre Matplotlib sans bloquer ROS 2."""

        # Aucun travail n'est necessaire si l'utilisateur a ferme la fenetre.
        if not plt.fignum_exists(self.figure.number):
            return

        vide = image[CANAL["validite"]] == 0.0
        self.vue_distance.set_data(
            np.ma.masked_where(vide, image[CANAL["distance"]])
        )
        self.vue_intensite.set_data(
            np.ma.masked_where(vide, image[CANAL["intensite"]])
        )
        self.axes_distance.set_title(
            f"RoboSense Airy - {HAUTEUR}x{LARGEUR} - "
            f"{nombre_pixels} pixels valides"
        )

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def decrire_premier_nuage(self, message, a_ring, points):
        """Affiche une seule fois les informations utiles au reglage."""

        noms_champs = [champ.name for champ in message.fields]
        print("\nPremier nuage recu")
        print("Champs du message :", noms_champs)
        print("Taille du message :", message.height, "x", message.width)

        if a_ring:
            print("Lignes de l'image : numero du laser (champ 'ring').")
        else:
            print(
                "Champ 'ring' absent : lignes calculees avec l'angle vertical "
                f"({FOV_BAS_DEGRES} a {FOV_HAUT_DEGRES} degres)."
            )
            print(
                "Pour une grille exacte (laser, azimut), mettre "
                "set(POINT_TYPE XYZIRT) dans rslidar_sdk/CMakeLists.txt, "
                "puis recompiler."
            )

        x = points["x"].astype(np.float64)
        y = points["y"].astype(np.float64)
        z = points["z"].astype(np.float64)
        distance = np.sqrt(x * x + y * y + z * z)
        utiles = np.isfinite(distance) & (distance > 0.0)
        if np.any(utiles):
            pitch = np.degrees(np.arcsin(z[utiles] / distance[utiles]))
            print(
                "Angle vertical observe :",
                round(float(pitch.min()), 1),
                "a",
                round(float(pitch.max()), 1),
                "degres",
            )
        if a_ring and points.size > 0:
            print("Numeros de laser observes : 0 a", int(points["ring"].max()))

    def lire_nuage(self, message):
        """Cette fonction est appelee a chaque nouveau nuage (un tour complet)."""

        a_ring = any(champ.name == "ring" for champ in message.fields)
        champs = ["x", "y", "z", "intensity"] + (["ring"] if a_ring else [])

        # Decodage du message ROS 2 en tableau NumPy structure.
        # skip_nans retire les directions sans retour (dense_points: false).
        points = point_cloud2.read_points(
            message,
            field_names=champs,
            skip_nans=True,
        )

        if not self.premier_nuage_recu:
            self.decrire_premier_nuage(message, a_ring, points)
            self.premier_nuage_recu = True

        image, nombre_hors_image = projeter_nuage(
            points["x"],
            points["y"],
            points["z"],
            points["intensity"],
            points["ring"] if a_ring else None,
        )

        if nombre_hors_image > 0 and not self.alerte_hors_image:
            self.get_logger().warning(
                f"{nombre_hors_image} points ont un numero de laser >= "
                f"{HAUTEUR} : augmenter HAUTEUR (modele 192 lasers ?)."
            )
            self.alerte_hors_image = True

        # Calcul des statistiques.
        pixels_valides = image[CANAL["validite"]] > 0.0
        nombre_pixels = int(np.count_nonzero(pixels_valides))
        taux_remplissage = 100.0 * nombre_pixels / pixels_valides.size

        print("\nNouveau nuage")
        print("Points recus :", len(points))
        print("Pixels contenant une mesure :", nombre_pixels)
        print("Taux de remplissage :", round(taux_remplissage, 2), "%")

        if nombre_pixels > 0:
            distances = image[CANAL["distance"]][pixels_valides]
            print("Distance minimale :", float(distances.min()), "m")
            print("Distance maximale :", float(distances.max()), "m")

        # Le tenseur a utiliser pour le deep learning est la variable 'image',
        # de forme (len(CANAUX), HAUTEUR, LARGEUR) : un nouveau tour complet
        # est disponible a chaque message.

        # Mise a jour de la fenetre a la frequence choisie.
        maintenant = time.monotonic()
        if maintenant - self.derniere_mise_a_jour >= 1.0 / FREQUENCE_AFFICHAGE:
            self.afficher_image(image, nombre_pixels)
            self.derniere_mise_a_jour = maintenant

        # Enregistrement de la premiere image contenant des mesures.
        if not self.premiere_image_sauvegardee and nombre_pixels > 0:
            chemin = Path(FICHIER_SORTIE).resolve()
            np.save(chemin, image)
            self.premiere_image_sauvegardee = True
            print("Premiere image enregistree dans :", chemin)
            print("Forme :", image.shape, "- canaux :", CANAUX)


def main():
    rclpy.init()
    noeud = ProjectionAiry()

    try:
        rclpy.spin(noeud)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    plt.close("all")
    noeud.destroy_node()

    # Ctrl+C peut deja avoir arrete ROS 2 : un second shutdown leverait une erreur.
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
