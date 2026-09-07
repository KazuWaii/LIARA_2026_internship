"""Reprise apres MemoryError transitoire (DataLoader worker, 4 septembre
matin) -- graine2 seule, memes conditions que
eval_finetune_jrdb2x_seeds.py (graine0=0.500/0.519, graine1=0.523/0.529
deja obtenus)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import probe_lineaire

if __name__ == "__main__":
    precision, f1, hist = probe_lineaire(
        "grille", "mmfi-src/checkpoints_mmfi_jrdb2x/final.pt",
        nb_etapes=2000, taille_lot=64, geler_encodeur=False, graine=2,
    )
    print(f"RESULTAT_PARTIEL jrdb2x_graine2 precision={precision:.3f} f1={f1:.3f}")
    print("EVAL_FINETUNE_JRDB2X_GRAINE2_OK")
