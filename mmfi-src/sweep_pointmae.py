"""Sweep rapide (1 graine, moins de pas) avant de committer au protocole
complet 3 graines -- meme logique que sweep_lambda_squelette.py/
sweep_distillation.py : geler_encodeur=True (linear probing, recommande par
PointGST a cette echelle) vs False (fine-tuning complet), 800 pas."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entrainer_pointmae import entrainer_pointmae

if __name__ == "__main__":
    for geler in (True, False):
        print(f"\n=== sweep pointmae geler_encodeur={geler} ===")
        precision, f1 = entrainer_pointmae(
            nb_etapes=800, taille_lot=32, geler_encodeur=geler,
            graine=0, nb_workers=6, verbeux_tous=50)
        print(f"RESULTAT_PARTIEL geler={geler} precision={precision:.3f} f1={f1:.3f}")
    print("SWEEP_POINTMAE_OK")
