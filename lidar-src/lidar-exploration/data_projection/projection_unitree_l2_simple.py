#!/usr/bin/env python3
"""Exemple simple : accumuler puis projeter les nuages du Unitree L2."""

from collections import deque
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


# Taille de l'image produite.
HAUTEUR = 36
LARGEUR = 144

# Nombre de nuages successifs utilises dans chaque projection.
NOMBRE_NUAGES = 6

# Champ de vision vertical utilise pour la projection.
FOV_HAUT_DEGRES = 90.0
FOV_BAS_DEGRES = 0.0

# Distances conservees, en metres.
DISTANCE_MIN = 0.05
DISTANCE_MAX = 5.0

# Nombre maximal de rafraichissements de la fenetre par seconde.
FREQUENCE_AFFICHAGE = 10.0


def creer_image_profondeur(x, y, z):
    """Projette les points XYZ dans une image contenant leur distance au lidar."""

    # Distance radiale de chaque point par rapport au lidar.
    distance = np.sqrt(x * x + y * y + z * z)

    # Suppression des points invalides ou hors de la plage choisie.
    valide = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & np.isfinite(distance)
        & (distance >= DISTANCE_MIN)
        & (distance <= DISTANCE_MAX)
    )

    x = x[valide]
    y = y[valide]
    z = z[valide]
    distance = distance[valide]

    # Image vide : NaN signifie qu'aucun point ne correspond au pixel.
    image = np.full((HAUTEUR, LARGEUR), np.nan, dtype=np.float32)

    if distance.size == 0:
        return image

    # Angles horizontal et vertical de chaque point.
    yaw = -np.arctan2(y, x)
    pitch = np.arcsin(np.clip(z / distance, -1.0, 1.0))

    fov_haut = np.deg2rad(FOV_HAUT_DEGRES)
    fov_bas = np.deg2rad(FOV_BAS_DEGRES)

    # Conservation des points situes dans le champ de vision vertical.
    dans_fov = (pitch >= fov_bas) & (pitch <= fov_haut)
    yaw = yaw[dans_fov]
    pitch = pitch[dans_fov]
    distance = distance[dans_fov]

    if distance.size == 0:
        return image

    # Conversion des angles en coordonnees de pixels.
    colonne = np.floor(0.5 * (yaw / np.pi + 1.0) * LARGEUR).astype(np.int32)
    ligne = np.floor(
        (1.0 - (pitch - fov_bas) / (fov_haut - fov_bas)) * HAUTEUR
    ).astype(np.int32)

    colonne = np.clip(colonne, 0, LARGEUR - 1)
    ligne = np.clip(ligne, 0, HAUTEUR - 1)

    # Si plusieurs points tombent dans le meme pixel, le plus proche est garde.
    # Les points lointains sont poses en premier, puis les proches les remplacent.
    ordre = np.argsort(distance)[::-1]
    image[ligne[ordre], colonne[ordre]] = distance[ordre]

    return image


class ProjectionLidar(Node):

    def __init__(self):
        super().__init__("projection_lidar_simple")

        self.create_subscription(
            PointCloud2,
            "/unilidar/cloud",
            self.lire_nuage,
            10,
        )

        # Ce tampon conserve automatiquement les NOMBRE_NUAGES dernieres images.
        self.images = deque(maxlen=NOMBRE_NUAGES)

        self.premiere_image_sauvegardee = False
        self.derniere_mise_a_jour = 0.0

        # Creation de la fenetre d'affichage en direct.
        plt.ion()

        couleurs = plt.cm.turbo.copy()
        couleurs.set_bad("black")

        self.figure, self.axes = plt.subplots()
        self.vue_image = self.axes.imshow(
            np.full((HAUTEUR, LARGEUR), np.nan, dtype=np.float32),
            cmap=couleurs,
            vmin=DISTANCE_MIN,
            vmax=DISTANCE_MAX,
            aspect="equal",
        )

        self.axes.set_title("Image de profondeur Unitree L2")
        self.axes.set_xlabel("Angle horizontal")
        self.axes.set_ylabel("Angle vertical")
        self.figure.colorbar(
            self.vue_image,
            ax=self.axes,
            label="Distance au lidar (m)",
        )
        self.figure.tight_layout()
        plt.show(block=False)

        self.get_logger().info("En attente des nuages de points...")

    def afficher_image(self, image, nombre_pixels, nombre_nuages):
        """Met a jour la fenetre Matplotlib sans bloquer ROS 2."""

        # Aucun travail n'est necessaire si l'utilisateur a ferme la fenetre.
        if not plt.fignum_exists(self.figure.number):
            return

        self.vue_image.set_data(np.ma.masked_invalid(image))
        self.axes.set_title(
            f"Unitree L2 - {nombre_nuages}/{NOMBRE_NUAGES} nuages - "
            f"{nombre_pixels} pixels valides"
        )

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def lire_nuage(self, message):
        """Cette fonction est appelee a chaque nouveau nuage."""

        # Decodage du message ROS 2 en tableaux NumPy.
        points = point_cloud2.read_points(
            message,
            field_names=["x", "y", "z"],
            skip_nans=True,
        )

        # Projection du nouveau nuage uniquement.
        image_nouvelle = creer_image_profondeur(
            points["x"],
            points["y"],
            points["z"],
        )

        # Ajout de l'image au tampon.
        # L'image la plus ancienne est automatiquement supprimee
        # lorsque le tampon contient deja NOMBRE_NUAGES images.
        self.images.append(image_nouvelle)

        # Fusion des images.
        # Pour chaque pixel, np.fmin conserve la distance la plus proche
        # tout en ignorant les valeurs NaN.
        pile_images = np.stack(list(self.images), axis=0)
        image = np.fmin.reduce(pile_images, axis=0)

        # Calcul des statistiques.
        pixels_valides = np.isfinite(image)
        nombre_pixels = int(np.count_nonzero(pixels_valides))
        taux_remplissage = 100.0 * nombre_pixels / image.size

        print("\nNouveau nuage")
        print("Points dans le nouveau nuage :", len(points))
        print("Images accumulees :", len(self.images), "/", NOMBRE_NUAGES)
        print("Taille de l'image :", image.shape)
        print("Pixels contenant une mesure :", nombre_pixels)
        print("Taux de remplissage :", round(taux_remplissage, 2), "%")

        if nombre_pixels > 0:
            print(
                "Distance minimale :",
                float(np.nanmin(image)),
                "m",
            )
            print(
                "Distance maximale :",
                float(np.nanmax(image)),
                "m",
            )

        # Mise a jour de la fenetre a la frequence choisie.
        maintenant = time.monotonic()

        if (
            maintenant - self.derniere_mise_a_jour
            >= 1.0 / FREQUENCE_AFFICHAGE
        ):
            self.afficher_image(
                image,
                nombre_pixels,
                len(self.images),
            )
            self.derniere_mise_a_jour = maintenant

        # Enregistrement de la premiere image construite avec
        # le nombre complet d'images demande.
        if (
            not self.premiere_image_sauvegardee
            and len(self.images) == NOMBRE_NUAGES
        ):
            chemin = Path("image_profondeur_l2.npy").resolve()
            np.save(chemin, image)

            self.premiere_image_sauvegardee = True

            print(
                "Premiere image accumulee enregistree dans :",
                chemin,
            )


def main():
    rclpy.init()
    noeud = ProjectionLidar()

    try:
        rclpy.spin(noeud)
    except KeyboardInterrupt:
        pass

    plt.close("all")
    noeud.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
