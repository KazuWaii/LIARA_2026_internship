#!/usr/bin/env python3
"""Projection temps reel du Unitree L2 avec derniere valeur valide par pixel."""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


# Taille de l'image de profondeur.
HAUTEUR = 36
LARGEUR = 144

# Une mesure reste disponible pendant ce nombre de nouveaux messages.
# Diminuer cette valeur augmente la reactivite dans une scene dynamique.
# L'augmenter remplit davantage l'image dans une scene immobile.
PERSISTANCE_IMAGES = 6

# Champ de vision vertical utilise pour la projection.
FOV_HAUT_DEGRES = 90.0
FOV_BAS_DEGRES = 0.0

# Distances conservees, en metres.
DISTANCE_MIN = 0.05
DISTANCE_MAX = 5.0

# Nombre maximal de rafraichissements de la fenetre par seconde.
FREQUENCE_AFFICHAGE = 10.0


def creer_image_profondeur(x, y, z):
    """Projette un seul nuage XYZ dans une image de distance."""

    distance = np.sqrt(x * x + y * y + z * z)

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

    image = np.full((HAUTEUR, LARGEUR), np.nan, dtype=np.float32)

    if distance.size == 0:
        return image

    yaw = -np.arctan2(y, x)
    pitch = np.arcsin(np.clip(z / distance, -1.0, 1.0))

    fov_haut = np.deg2rad(FOV_HAUT_DEGRES)
    fov_bas = np.deg2rad(FOV_BAS_DEGRES)

    dans_fov = (pitch >= fov_bas) & (pitch <= fov_haut)
    yaw = yaw[dans_fov]
    pitch = pitch[dans_fov]
    distance = distance[dans_fov]

    if distance.size == 0:
        return image

    colonne = np.floor(
        0.5 * (yaw / np.pi + 1.0) * LARGEUR
    ).astype(np.int32)
    ligne = np.floor(
        (1.0 - (pitch - fov_bas) / (fov_haut - fov_bas)) * HAUTEUR
    ).astype(np.int32)

    colonne = np.clip(colonne, 0, LARGEUR - 1)
    ligne = np.clip(ligne, 0, HAUTEUR - 1)

    # Dans le nuage actuel, plusieurs points peuvent tomber dans le meme pixel.
    # Le z-buffer garde alors le point le plus proche.
    image_plate = np.full(HAUTEUR * LARGEUR, np.inf, dtype=np.float32)
    indices = ligne * LARGEUR + colonne
    np.minimum.at(image_plate, indices, distance.astype(np.float32))

    image = image_plate.reshape(HAUTEUR, LARGEUR)
    image[np.isinf(image)] = np.nan
    return image


class ProjectionLidar(Node):

    def __init__(self):
        super().__init__("projection_lidar_derniere_valeur")

        self.create_subscription(
            PointCloud2,
            "/unilidar/cloud",
            self.lire_nuage,
            10,
        )

        # Une seule image et un age par pixel remplacent le tampon de nuages.
        self.image_memoire = np.full(
            (HAUTEUR, LARGEUR),
            np.nan,
            dtype=np.float32,
        )
        self.age_pixels = np.full(
            (HAUTEUR, LARGEUR),
            PERSISTANCE_IMAGES,
            dtype=np.uint16,
        )

        self.nombre_messages = 0
        self.premiere_image_sauvegardee = False
        self.derniere_mise_a_jour = 0.0

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

        self.get_logger().info(
            "En attente des nuages de points sur /unilidar/cloud..."
        )

    def actualiser_memoire(self, image_nouvelle):
        """Met a jour chaque pixel avec sa derniere valeur valide."""

        pixels_deja_valides = np.isfinite(self.image_memoire)

        # Les pixels qui possedent deja une mesure vieillissent d'un message.
        self.age_pixels[pixels_deja_valides] += 1

        # Une nouvelle mesure remplace toujours l'ancienne valeur du pixel.
        # Contrairement a une fusion par minimum, la nouveaute est donc visible
        # immediatement, meme si un objet s'est eloigne du lidar.
        nouveaux_pixels = np.isfinite(image_nouvelle)
        self.image_memoire[nouveaux_pixels] = image_nouvelle[nouveaux_pixels]
        self.age_pixels[nouveaux_pixels] = 0

        # Les valeurs qui n'ont pas ete observees depuis trop longtemps sont
        # supprimees afin de ne pas creer de traces fantomes permanentes.
        pixels_expires = self.age_pixels >= PERSISTANCE_IMAGES
        self.image_memoire[pixels_expires] = np.nan

        return nouveaux_pixels

    def afficher_image(self, image, nombre_pixels, nombre_nouveaux):
        """Met a jour la fenetre Matplotlib sans bloquer ROS 2."""

        if not plt.fignum_exists(self.figure.number):
            return

        self.vue_image.set_data(np.ma.masked_invalid(image))
        self.axes.set_title(
            f"Unitree L2 - {nombre_pixels} pixels valides "
            f"({nombre_nouveaux} actualises)"
            f"({PERSISTANCE_IMAGES} images persistantes)"
        )

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def lire_nuage(self, message):
        """Traite immediatement chaque nouveau nuage PointCloud2."""

        points = point_cloud2.read_points(
            message,
            field_names=["x", "y", "z"],
            skip_nans=True,
        )

        # Seul le nouveau nuage est projete : aucun nuage n'est concatene.
        image_nouvelle = creer_image_profondeur(
            points["x"],
            points["y"],
            points["z"],
        )

        nouveaux_pixels = self.actualiser_memoire(image_nouvelle)
        self.nombre_messages += 1

        # La copie donne une image stable, directement utilisable par un modele.
        image = self.image_memoire.copy()

        pixels_valides = np.isfinite(image)
        nombre_pixels = int(np.count_nonzero(pixels_valides))
        nombre_nouveaux = int(np.count_nonzero(nouveaux_pixels))
        nombre_memorises = nombre_pixels - int(
            np.count_nonzero(pixels_valides & nouveaux_pixels)
        )
        taux_remplissage = 100.0 * nombre_pixels / image.size

        print("\nNouveau nuage")
        print("Points recus :", len(points))
        print("Pixels actualises :", nombre_nouveaux)
        print("Pixels conserves de la memoire :", nombre_memorises)
        print("Pixels valides au total :", nombre_pixels)
        print("Taux de remplissage :", round(taux_remplissage, 2), "%")

        if nombre_pixels > 0:
            print("Distance minimale :", float(np.nanmin(image)), "m")
            print("Distance maximale :", float(np.nanmax(image)), "m")

        # Une nouvelle image est disponible pour le deep learning a chaque
        # callback, sans attendre que plusieurs nuages soient concatenes.
        # Le tenseur a utiliser est la variable 'image'.

        maintenant = time.monotonic()
        if maintenant - self.derniere_mise_a_jour >= 1.0 / FREQUENCE_AFFICHAGE:
            self.afficher_image(image, nombre_pixels, nombre_nouveaux)
            self.derniere_mise_a_jour = maintenant

        # Enregistrement d'un exemple une fois la memoire initialement remplie.
        if (
            not self.premiere_image_sauvegardee
            and self.nombre_messages >= PERSISTANCE_IMAGES
        ):
            chemin = Path(
                "image_profondeur_l2_derniere_valeur.npy"
            ).resolve()
            np.save(chemin, image)
            self.premiere_image_sauvegardee = True
            print("Premiere image memorisee enregistree dans :", chemin)


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
