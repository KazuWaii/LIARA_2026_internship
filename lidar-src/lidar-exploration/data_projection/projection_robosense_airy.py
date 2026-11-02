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

Fluidite de l'affichage :

- La reception ROS 2 tourne dans un fil separe qui ne garde que le dernier
  nuage recu. Si le traitement prend du retard, les nuages intermediaires sont
  sautes au lieu de s'accumuler : l'image affichee est toujours la plus
  recente, le decalage ne grandit pas.

- La fenetre ne redessine que les deux images (blitting), pas les axes ni les
  barres de couleur : environ 30 ms par image au lieu de 190 ms.

- Une ligne d'etat par seconde donne la frequence de reception, la frequence
  d'affichage et le temps de chaque etape, pour savoir ce qui ralentit.
"""

from pathlib import Path
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import rclpy

from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


# Topic publie par rslidar_sdk (ros_send_point_cloud_topic dans config.yaml).
TOPIC_NUAGE = "/rslidar_points"

# Frequence de rotation du lidar (10 Hz par defaut), utilisee pour detecter
# les nuages perdus avant d'arriver dans ce script.
FREQUENCE_LIDAR = 10.0

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

# Intervalle entre deux lignes d'etat dans le terminal, en secondes.
PERIODE_ETAT = 1.0

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
    # Chaque point recoit une seule cle de tri : le numero du pixel dans les
    # bits de poids fort, la distance dans les bits de poids faible. Pour un
    # float32 positif, ses bits lus comme un entier sont dans le meme ordre que
    # sa valeur : trier la cle trie donc par pixel, puis par distance. Le
    # premier point de chaque pixel est le plus proche. Un seul tri d'entiers
    # est environ 5 fois plus rapide que np.lexsort sur deux cles.
    pixel = ligne * LARGEUR + colonne
    cle = (pixel << 32) | distance.view(np.int32).astype(np.int64)
    ordre = np.argsort(cle)
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

        # Dernier nuage recu. Il est ecrit par le fil ROS 2 et lu par le fil
        # principal : le verrou evite qu'ils y accedent en meme temps.
        self.verrou = threading.Lock()
        self.dernier_message = None
        self.nombre_recus = 0

        # Profil "capteur" (best effort) : on prefere perdre une image que
        # d'accumuler du retard. Compatible avec l'editeur fiable du SDK.
        self.create_subscription(
            PointCloud2,
            TOPIC_NUAGE,
            self.recevoir_nuage,
            qos_profile_sensor_data,
        )

        self.premier_nuage_recu = False
        self.alerte_hors_image = False
        self.alerte_reception = False
        self.premiere_image_sauvegardee = False

        # Mesures pour la ligne d'etat.
        self.debut = time.monotonic()
        self.dernier_etat = self.debut
        self.recus_au_dernier_etat = 0
        self.nombre_affiches = 0
        self.duree_projection = 0.0
        self.duree_affichage = 0.0
        self.dernier_remplissage = 0.0

        self.creer_fenetre()

        self.get_logger().info(
            f"En attente des nuages de points sur {TOPIC_NUAGE}..."
        )

    # ------------------------------------------------------------------
    # Fil ROS 2 : reception seulement, aucun calcul.
    # ------------------------------------------------------------------

    def recevoir_nuage(self, message):
        """Garde le nuage le plus recent ; l'ancien, s'il n'a pas ete traite,
        est abandonne."""

        with self.verrou:
            self.dernier_message = message
            self.nombre_recus += 1

    def prendre_dernier_message(self):
        """Retourne le dernier nuage recu (ou None) et vide la place."""

        with self.verrou:
            message = self.dernier_message
            self.dernier_message = None
        return message

    # ------------------------------------------------------------------
    # Fenetre Matplotlib (fil principal).
    # ------------------------------------------------------------------

    def creer_fenetre(self):
        plt.ion()

        self.figure, (self.axes_distance, self.axes_intensite) = plt.subplots(
            2, 1, figsize=(12, 6), sharex=True
        )
        canvas = self.figure.canvas

        # Le blitting ne redessine que les images. Les images sont alors
        # "animees" : Matplotlib les exclut du dessin normal de la fenetre.
        self.blit_possible = bool(getattr(canvas, "supports_blit", False))
        self.fond = None

        image_vide = np.ma.masked_array(
            np.zeros((HAUTEUR, LARGEUR), dtype=np.float32),
            mask=True,
        )
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

        self.axes_distance.set_title(
            f"RoboSense Airy - image {HAUTEUR}x{LARGEUR} (laser x azimut)"
        )
        self.axes_intensite.set_xlabel("Angle horizontal (colonne)")
        self.figure.tight_layout()

        # A chaque dessin complet de la fenetre (ouverture, redimensionnement),
        # on memorise le fond sans les images, pour le blitting.
        canvas.mpl_connect("draw_event", self.sur_dessin_complet)

        plt.show(block=False)
        canvas.draw()
        canvas.flush_events()

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
            animated=self.blit_possible,
        )
        axes.set_ylabel("Laser (ligne)")
        self.figure.colorbar(vue, ax=axes, label=legende)
        return vue

    def sur_dessin_complet(self, evenement):
        """Memorise le fond de la fenetre, puis y dessine les images."""

        if not self.blit_possible:
            return
        self.fond = self.figure.canvas.copy_from_bbox(self.figure.bbox)
        self.dessiner_images()

    def dessiner_images(self):
        self.axes_distance.draw_artist(self.vue_distance)
        self.axes_intensite.draw_artist(self.vue_intensite)

    def fenetre_ouverte(self):
        return plt.fignum_exists(self.figure.number)

    def attendre_evenements(self, duree):
        """Laisse la fenetre reagir (deplacement, fermeture...) un instant."""

        self.figure.canvas.start_event_loop(duree)

    def afficher_image(self, image):
        """Met a jour les deux images de la fenetre."""

        if not self.fenetre_ouverte():
            return

        vide = image[CANAL["validite"]] == 0.0
        self.vue_distance.set_data(
            np.ma.masked_where(vide, image[CANAL["distance"]])
        )
        self.vue_intensite.set_data(
            np.ma.masked_where(vide, image[CANAL["intensite"]])
        )

        canvas = self.figure.canvas
        if self.blit_possible and self.fond is not None:
            # Fond memorise + images seulement : pas de redessin des axes.
            canvas.restore_region(self.fond)
            self.dessiner_images()
            canvas.blit(self.figure.bbox)
        else:
            canvas.draw_idle()
        canvas.flush_events()

    # ------------------------------------------------------------------
    # Traitement (fil principal).
    # ------------------------------------------------------------------

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
        print()

    def traiter_nuage(self, message):
        """Projette un nuage (un tour complet) et met a jour l'affichage."""

        debut = time.perf_counter()

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

        # Le tenseur a utiliser pour le deep learning est la variable 'image',
        # de forme (len(CANAUX), HAUTEUR, LARGEUR) : un tour complet du lidar.

        milieu = time.perf_counter()
        self.afficher_image(image)
        fin = time.perf_counter()

        # Moyennes glissantes des durees, en secondes.
        self.duree_projection = 0.8 * self.duree_projection + 0.2 * (milieu - debut)
        self.duree_affichage = 0.8 * self.duree_affichage + 0.2 * (fin - milieu)
        self.nombre_affiches += 1
        self.dernier_remplissage = 100.0 * float(
            np.count_nonzero(image[CANAL["validite"]])
        ) / (HAUTEUR * LARGEUR)

        if nombre_hors_image > 0 and not self.alerte_hors_image:
            self.get_logger().warning(
                f"{nombre_hors_image} points ont un numero de laser >= "
                f"{HAUTEUR} : augmenter HAUTEUR (modele 192 lasers ?)."
            )
            self.alerte_hors_image = True

        # Enregistrement de la premiere image contenant des mesures.
        if not self.premiere_image_sauvegardee and self.dernier_remplissage > 0:
            chemin = Path(FICHIER_SORTIE).resolve()
            np.save(chemin, image)
            self.premiere_image_sauvegardee = True
            print("Premiere image enregistree dans :", chemin)
            print("Forme :", image.shape, "- canaux :", CANAUX)

    def afficher_etat(self):
        """Une ligne par seconde : frequences et duree de chaque etape."""

        maintenant = time.monotonic()
        duree = maintenant - self.dernier_etat
        if duree < PERIODE_ETAT:
            return

        with self.verrou:
            nombre_recus = self.nombre_recus

        frequence_reception = (nombre_recus - self.recus_au_dernier_etat) / duree
        frequence_affichage = self.nombre_affiches / duree

        print(
            f"Recus : {frequence_reception:4.1f} Hz | "
            f"Affiches : {frequence_affichage:4.1f} Hz | "
            f"Projection : {1000 * self.duree_projection:3.0f} ms | "
            f"Affichage : {1000 * self.duree_affichage:3.0f} ms | "
            f"Remplissage : {self.dernier_remplissage:4.1f} %"
        )

        # Apres quelques secondes, une reception nettement sous la frequence
        # du lidar signifie que des nuages sont perdus avant ce script
        # (transport DDS), et non par le traitement.
        if (
            not self.alerte_reception
            and maintenant - self.debut > 3.0
            and 0.0 < frequence_reception < 0.8 * FREQUENCE_LIDAR
        ):
            self.get_logger().warning(
                "Reception inferieure a la frequence du lidar : des nuages "
                "sont perdus par ROS 2 avant ce script. Essayer "
                "'export FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA' dans les "
                "terminaux du driver et de ce script."
            )
            self.alerte_reception = True

        self.dernier_etat = maintenant
        self.recus_au_dernier_etat = nombre_recus
        self.nombre_affiches = 0


def tourner(executeur):
    """Boucle de reception ROS 2, executee dans un fil separe."""

    try:
        executeur.spin()
    except ExternalShutdownException:
        pass


def main():
    rclpy.init()
    noeud = ProjectionAiry()

    # La reception ROS 2 tourne dans un fil separe : elle continue pendant
    # que le fil principal projette et affiche. La fenetre Matplotlib, elle,
    # doit rester dans le fil principal.
    executeur = SingleThreadedExecutor()
    executeur.add_node(noeud)
    fil_ros = threading.Thread(target=tourner, args=(executeur,), daemon=True)
    fil_ros.start()

    try:
        while rclpy.ok() and noeud.fenetre_ouverte():
            message = noeud.prendre_dernier_message()
            if message is None:
                # Rien de nouveau : la fenetre traite ses evenements.
                noeud.attendre_evenements(0.005)
            else:
                noeud.traiter_nuage(message)
            noeud.afficher_etat()
    except KeyboardInterrupt:
        pass

    plt.close("all")

    # Ctrl+C peut deja avoir arrete ROS 2 : un second shutdown leverait une
    # erreur. L'arret de ROS 2 termine aussi la boucle du fil de reception.
    if rclpy.ok():
        rclpy.shutdown()
    fil_ros.join(timeout=2.0)
    noeud.destroy_node()


if __name__ == "__main__":
    main()
