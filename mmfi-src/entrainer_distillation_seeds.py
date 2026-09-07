"""Run complet 3 graines du meilleur config trouve par sweep_distillation.py
(T=4.0, lambda3=1.0 -- 0,537 a 800 pas, deja au-dessus de la reference
from-scratch 0,509+-0,044 a MOINS de pas) -- 2000 pas, memes graines 0/1/2
que toutes les autres comparaisons du projet, pour une comparaison directe
et rigoureuse (pas de conclusion sur une seule graine, lecon retenue de la
nuit du 3-4 septembre)."""

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entrainer_distillation import entrainer_distillation

CHECKPOINT_ENSEIGNANT = "mmfi-src/checkpoints_stgcn_squelette_graine0.pt"

if __name__ == "__main__":
    resultats = {}
    for graine in (0, 1, 2):
        print(f"\n=== distillation_T4_l3-1 graine={graine} ===")
        precision, f1 = entrainer_distillation(
            chemin_checkpoint_enseignant=CHECKPOINT_ENSEIGNANT,
            nb_etapes=2000, graine=graine, temperature=4.0, lambda3=1.0,
            suffixe="distillation_T4_l3", verbeux_tous=100)
        resultats[graine] = (precision, f1)
        print(f"RESULTAT_PARTIEL distillation_graine{graine} precision={precision:.3f} f1={f1:.3f}")

    print("\n\n=== RESUME MULTI-GRAINES (distillation squelette->LiDAR, T=4.0, lambda3=1.0) ===")
    for g, (p, f) in resultats.items():
        print(f"graine={g}  precision={p:.3f}  f1={f:.3f}")
    ps = [resultats[g][0] for g in (0, 1, 2)]
    fs = [resultats[g][1] for g in (0, 1, 2)]
    print(f"distillation: precision mean={statistics.mean(ps):.3f} stdev={statistics.stdev(ps):.3f}  "
          f"f1 mean={statistics.mean(fs):.3f} stdev={statistics.stdev(fs):.3f}")
    print("references (meme protocole exact, 2000 pas, memes graines) : "
          "from-scratch LiDAR=0.509+-0.044 | squelette-auxiliaire (naif)=0.549+-0.009 | "
          "laplacien+SVM regle (meilleur LiDAR-only, classique)=0.659 | "
          "squelette-plafond (verite-terrain, PAS LiDAR-only)=0.994+-0.007")
    print("ENTRAINEMENT_DISTILLATION_SEEDS_OK")
