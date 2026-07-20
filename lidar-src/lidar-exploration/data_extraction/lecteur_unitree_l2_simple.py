#!/usr/bin/env python3
"""Exemple minimal pour lire le nuage de points du Unitree L2."""

import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class LecteurLidar(Node):

    def __init__(self):
        # Creation du noeud ROS 2.
        super().__init__("lecteur_lidar_simple")

        # Abonnement au nuage de points du Unitree L2.
        self.create_subscription(
            PointCloud2,
            "/unilidar/cloud",
            self.lire_nuage,
            10,
        )

        self.get_logger().info("En attente des nuages de points...")

    def lire_nuage(self, message):
        """Cette fonction est appelee a chaque nouveau nuage."""

        # Decodage du message ROS 2 en tableau NumPy.
        points = point_cloud2.read_points(
            message,
            field_names=["x", "y", "z", "intensity", "ring", "time"],
            skip_nans=True,
        )

        # Acces aux differentes donnees.
        x = points["x"]
        y = points["y"]
        z = points["z"]
        intensity = points["intensity"]
        ring = points["ring"]
        temps = points["time"]

        # Creation d'une matrice contenant les coordonnees XYZ.
        xyz = np.column_stack((x, y, z))

        print("\nNouveau nuage")
        print("Nombre de points :", len(xyz))

        if len(xyz) > 0:
            print("Premier XYZ :", xyz[0])
            print("Intensite :", intensity[0])
            print("Anneau :", ring[0])
            print("Temps relatif :", temps[0])

        # Les tableaux utilisables sont maintenant :
        # xyz, intensity, ring et temps.


def main():
    # Demarrage de ROS 2.
    rclpy.init()

    # Creation du lecteur.
    noeud = LecteurLidar()

    try:
        # Attend les messages et appelle lire_nuage().
        rclpy.spin(noeud)
    except KeyboardInterrupt:
        pass

    # Arret propre.
    noeud.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
