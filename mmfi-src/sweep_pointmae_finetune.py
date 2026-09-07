"""Essai rapide (1 graine, 800 pas, comme les autres sweeps du projet)
avant de committer au protocole complet 3 graines (~3h/graine mesure,
cf. Tracking) -- verifie si le fine-tuning complet de l'encodeur Point-MAE
depasse le linear probing (0,934+-0,006) avant d'investir ~9h de plus."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from dataset_jepa import decouper_train_val
from entrainer_laplacien_v4 import _masque
from entrainer_pointmae_finetune import entrainer_pointmae_finetune

if __name__ == "__main__":
    data = np.load("mmfi-src/groupes_pointmae.npz")
    groupes, centres, y, envs, sujets = data["groupes"], data["centres"], data["y"], data["envs"], data["sujets"]
    seq_train, seq_val = decouper_train_val()
    m_train, m_val = _masque(seq_train, envs, sujets), _masque(seq_val, envs, sujets)
    gt, ct, yt = groupes[m_train], centres[m_train], y[m_train]
    gv, cv, yv = groupes[m_val], centres[m_val], y[m_val]
    print(f"train: {gt.shape}  val: {gv.shape}")

    precision, f1 = entrainer_pointmae_finetune(gt, ct, yt, gv, cv, yv, nb_etapes=800, taille_lot=16,
                                                 graine=0, verbeux_tous=50)
    print(f"\nRESULTAT_SWEEP_FINETUNE precision={precision:.3f} f1={f1:.3f}")
    print("reference (linear probing, encodeur fige) : 0.934+-0.006")
    print("SWEEP_POINTMAE_FINETUNE_OK")
