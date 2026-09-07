"""Comme entrainer_laplacien_v4.py, mais consomme features_laplacien_v5_all.npz
(extraire_laplacien_v5.py -- nettoyage=True, RANSAC plans + bruit statistique
avant DBSCAN) et compare directement aux references v4 SANS nettoyage
(0,659 sujet / 0,653 environnement) -- teste si l'isolation de la personne
etait bien le facteur limitant (hypothese rang 3, recherche du 5 septembre)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entrainer_laplacien_v4 import _masque, _evaluer
from dataset_jepa import decouper_train_val, decouper_train_val_environnement

import numpy as np

if __name__ == "__main__":
    data = np.load("mmfi-src/features_laplacien_v5_all.npz")
    X, y, envs, sujets = data["X"], data["y"], data["envs"], data["sujets"]
    print(f"features totales (avec nettoyage RANSAC+outlier): X={X.shape}")

    resultats = {}

    seq_train, seq_val = decouper_train_val()
    m_train, m_val = _masque(seq_train, envs, sujets), _masque(seq_val, envs, sujets)
    resultats["split_sujet_nettoyage"] = _evaluer(
        "SPLIT SUJET, 8 fenetres, AVEC nettoyage RANSAC+outlier",
        X[m_train], y[m_train], X[m_val], y[m_val], sujets[m_train])

    seq_train_e, seq_val_e = decouper_train_val_environnement()
    m_train_e, m_val_e = _masque(seq_train_e, envs, sujets), _masque(seq_val_e, envs, sujets)
    resultats["split_environnement_nettoyage"] = _evaluer(
        "SPLIT ENVIRONNEMENT (papier), 8 fenetres, AVEC nettoyage RANSAC+outlier",
        X[m_train_e], y[m_train_e], X[m_val_e], y[m_val_e], sujets[m_train_e])

    print("\n\n=== RESUME FINAL (avec vs sans nettoyage) ===")
    print("SANS nettoyage (v4, reference) : split_sujet SVM_tune=0.659 HGB_tune=0.649 | "
          "split_environnement SVM_tune=0.643 HGB_tune=0.653")
    for nom, r in resultats.items():
        print(f"{nom}: SVM_defaut={r['svm_defaut']:.3f}  RF_defaut={r['rf_defaut']:.3f}  "
              f"SVM_tune={r['svm_tune']:.3f} (f1={r['svm_tune_f1']:.3f})  "
              f"HGB_tune={r['hgb_tune']:.3f} (f1={r['hgb_tune_f1']:.3f})")
    print("ENTRAINEMENT_LAPLACIEN_V5_OK")
