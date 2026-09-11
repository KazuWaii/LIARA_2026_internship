"""Comme entrainer_laplacien_v4.py, mais consomme features_laplacien_v5_all.npz
(extraire_laplacien_v5.py -- nettoyage=True, RANSAC plans + bruit statistique
avant DBSCAN) et compare directement aux references v4 SANS nettoyage
(0,659 sujet / 0,653 environnement) -- teste si l'isolation de la personne
etait bien le facteur limitant (hypothese rang 3, recherche du 5 septembre).

Resultat : echec net (cf. README) -- le nettoyage detruit le signal."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import protocole as proto

if __name__ == "__main__":
    data = np.load("mmfi-src/features_laplacien_v5_all.npz")
    X, y, envs, sujets = data["X"], data["y"], data["envs"], data["sujets"]
    print(f"features totales (avec nettoyage RANSAC+outlier): X={X.shape}")

    resultats = proto.evaluer_deux_splits(
        X, y, envs, sujets,
        lambda *a: proto.evaluer_classifieurs(*a, regler=False),
        nom_experience="LAPLACIEN v5 (nettoyage RANSAC + bruit statistique)")

    proto.resumer("LAPLACIEN v5", resultats)
    print("\nreference v4 SANS nettoyage : split sujet SVM_tune=0.659 HGB_tune=0.649 | "
          "split environnement SVM_tune=0.643 HGB_tune=0.653")
    print("ENTRAINEMENT_LAPLACIEN_V5_OK")
