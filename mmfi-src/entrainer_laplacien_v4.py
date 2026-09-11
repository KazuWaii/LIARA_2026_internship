"""Evalue les features Laplacien v4 (quadrants, 8 fenetres/sequence, 1080
sequences -- extraire_laplacien_v4.py) sur les deux splits du projet.

Ablation d'origine (README, piste B) : le meilleur resultat Laplacien etait
0,617 avec 4 fenetres/sequence et le split sujet ; ce script isole l'effet
du fenetrage plus dense (8 fenetres) et celui du split.

Hyperparametres SVM/HGB FIXES (protocole.SVM_TUNE_PARAMS/HGB_TUNE_PARAMS,
trouves par GridSearchCV sur les features v2, cf. tuner_laplacien_v2.py) :
`regler=False` -- un grid search par split coutait trop cher avec le
one-vs-one 27 classes, et on veut justement isoler l'effet du split A
HYPERPARAMETRES CONSTANTS."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import protocole as proto

if __name__ == "__main__":
    data = np.load("mmfi-src/features_laplacien_v4_all.npz")
    X, y, envs, sujets = data["X"], data["y"], data["envs"], data["sujets"]
    print(f"features totales: X={X.shape}")

    resultats = proto.evaluer_deux_splits(
        X, y, envs, sujets,
        lambda *a: proto.evaluer_classifieurs(*a, regler=False),
        nom_experience="LAPLACIEN v4 (quadrants, 8 fenetres/sequence)")

    proto.resumer("LAPLACIEN v4", resultats)
    print("\nreference anterieure (4 fenetres, split sujet) : "
          "SVM_defaut=0.617 SVM_tune=0.638 HGB_tune=0.649")
    print("ENTRAINEMENT_LAPLACIEN_V4_OK")
