"""Classifieurs CLASSIQUES (SVM/RandomForest/HistGradientBoosting) sur les
features Point-MAE DEJA EXTRAITES (features_pointmae.npz -- encodeur
ShapeNet-pretrained FIGE, 768 dims/frame, 8 frames/fenetre), a la place de
la tete lineaire+pooling appris (entrainer_pointmae_seeds.py). Meme pooling
mean+max TEMPOREL (8 frames -> 1536 dims), mais classifieur classique au
lieu d'une descente de gradient -- teste si un classifieur a marge (SVM) ou
un ensemble d'arbres extrait davantage du signal deja tres riche de ces
features que la tete lineaire simple.

Reglage du SVM par GridSearchCV + GroupKFold PAR SUJET sur le TRAIN SEUL
(cf. protocole.regler_svm) : chaque sequence produit plusieurs fenetres
correlees, qu'un K-Fold aveugle repartirait entre folds.

Evalue sur LES DEUX splits (sujet ET environnement E01-E03/E04) -- avant le
11 septembre ce script ne calculait que le split sujet (0,949), alors que le
README rapportait aussi 0,926 en split environnement, chiffre issu d'une
edition ad hoc jamais commitee."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import protocole as proto

if __name__ == "__main__":
    data = np.load("mmfi-src/features_pointmae.npz")
    X, y, envs, sujets = data["X"], data["y"], data["envs"], data["sujets"]
    X = proto.pool_temporel(X)
    print(f"features Point-MAE poolees temporellement: X={X.shape}")

    resultats = proto.evaluer_deux_splits(
        X, y, envs, sujets, proto.evaluer_classifieurs,
        nom_experience="POINT-MAE + CLASSIFIEURS CLASSIQUES")

    proto.resumer("POINT-MAE + CLASSIFIEURS CLASSIQUES", resultats)
    print("\nreference tete lineaire+pooling appris : 0.934 +/- 0.006 (sujet)")
    print("TUNER_POINTMAE_CLASSIQUE_OK")
