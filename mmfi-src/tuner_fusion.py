"""Fusion des features du spectre Laplacien (classique) et des features
Point-MAE (ShapeNet) -- teste si les deux sont COMPLEMENTAIRES : le spectre
Laplacien capture la geometrie de la pose via les valeurs propres du graphe
de proximite, Point-MAE capture des features apprises generiques sur ~50k
formes 3D. Deux points de vue differents sur le meme nuage de points isole.

Les deux caches (features_laplacien_v4_all.npz : 8 fenetres/sequence ;
features_pointmae.npz : 4 fenetres/sequence) n'ont PAS le meme nombre de
fenetres par sequence -- fusion au niveau de la SEQUENCE (pas de la
fenetre) : moyenne des fenetres de chaque sequence pour les deux jeux de
features (cle unique (env,sujet,y) -- action encodee dans y, cf.
mmfi_dataset.ACTIONS), puis concatenation des deux vecteurs moyennes.

Evalue sur LES DEUX splits (sujet ET environnement). Deux changements par
rapport a la version d'avant le 11 septembre :
  - le split environnement est reellement calcule (il etait rapporte a 0,937
    dans le README sans code ni log correspondant) ;
  - le reglage du SVM utilise la grille complete du projet
    (protocole.GRILLE_SVM) au lieu du sous-ensemble utilise ici auparavant.
    La grille precedente en est un sous-ensemble strict, donc le cv_acc
    retenu ne peut que monter ou rester egal ; le score de validation, lui,
    peut bouger legerement."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import protocole as proto

if __name__ == "__main__":
    dlap = np.load("mmfi-src/features_laplacien_v4_all.npz")
    dpm = np.load("mmfi-src/features_pointmae.npz")

    Xlap, ylap, elap, slap = proto.moyenne_par_sequence(
        dlap["X"], dlap["y"], dlap["envs"], dlap["sujets"])
    Xpm, ypm, epm, spm = proto.moyenne_par_sequence(
        proto.pool_temporel(dpm["X"]), dpm["y"], dpm["envs"], dpm["sujets"])

    print(f"sequences Laplacien: {Xlap.shape}  sequences Point-MAE: {Xpm.shape}")
    assert np.array_equal(ylap, ypm) and np.array_equal(elap, epm) and np.array_equal(slap, spm), \
        "les deux jeux de features doivent couvrir exactement les memes sequences, dans le meme ordre"

    X = np.concatenate([Xlap, Xpm], axis=1)
    print(f"X_fusion: {X.shape}")

    resultats = proto.evaluer_deux_splits(
        X, ylap, elap, slap, proto.evaluer_classifieurs,
        nom_experience="FUSION LAPLACIEN + POINT-MAE")

    proto.resumer("FUSION LAPLACIEN + POINT-MAE", resultats)
    print("\nreferences : Laplacien seul 0.659 (sujet) | Point-MAE seul 0.949 (sujet) / 0.926 (env)")
    print("TUNER_FUSION_OK")
