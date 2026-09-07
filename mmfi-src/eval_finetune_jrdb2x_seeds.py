"""Multi-graines (4 septembre matin) sur le checkpoint MM-Fi+JRDB(x2)
(`checkpoints_mmfi_jrdb2x/final.pt`, cf. entrainer_combine_jrdb2x.py) --
teste si DOUBLER le volume JRDB (reutilisation du split "test", ~50
sequences train au lieu de 25) rapproche le pre-entrainement de
from-scratch, suite au resultat multi-graines precedent (JRDB x1 :
0,498+-0,044 vs from-scratch 0,509+-0,044, statistiquement indiscernables).
Meme protocole exact (2000 pas, lot=64, geler_encodeur=False), 3 graines
(0,1,2) -- reutilise la reference from-scratch deja etablie sur 3 graines
(pas de nouvelle donnee : from-scratch ne depend pas du checkpoint)."""

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import probe_lineaire

REFERENCE_SCRATCH = {
    0: (0.514, 0.508),
    1: (0.551, 0.566),
    2: (0.463, 0.484),
}

if __name__ == "__main__":
    resultats = {}
    for graine in (0, 1, 2):
        cle = f"jrdb2x_graine{graine}"
        print(f"\n=== {cle} ===")
        precision, f1, hist = probe_lineaire(
            "grille", "mmfi-src/checkpoints_mmfi_jrdb2x/final.pt",
            nb_etapes=2000, taille_lot=64, geler_encodeur=False, graine=graine,
        )
        resultats[cle] = (precision, f1)
        print(f"RESULTAT_PARTIEL {cle} precision={precision:.3f} f1={f1:.3f}")

    print("\n\n=== RESUME JRDB2X MULTI-GRAINES ===")
    print(f"{'condition':20s} {'precision':>10s} {'f1_macro':>10s}")
    for nom, (p, f) in resultats.items():
        print(f"{nom:20s} {p:10.3f} {f:10.3f}")
    for g, (p, f) in REFERENCE_SCRATCH.items():
        print(f"{'scratch_graine' + str(g):20s} {p:10.3f} {f:10.3f}  (reference, deja mesure)")

    ps = [resultats[f"jrdb2x_graine{g}"][0] for g in (0, 1, 2)]
    fs = [resultats[f"jrdb2x_graine{g}"][1] for g in (0, 1, 2)]
    ps_s = [REFERENCE_SCRATCH[g][0] for g in (0, 1, 2)]
    fs_s = [REFERENCE_SCRATCH[g][1] for g in (0, 1, 2)]
    print(f"jrdb2x: precision mean={statistics.mean(ps):.3f} stdev={statistics.stdev(ps):.3f}"
          f"  f1 mean={statistics.mean(fs):.3f} stdev={statistics.stdev(fs):.3f}")
    print(f"scratch (reference): precision mean={statistics.mean(ps_s):.3f} stdev={statistics.stdev(ps_s):.3f}"
          f"  f1 mean={statistics.mean(fs_s):.3f} stdev={statistics.stdev(fs_s):.3f}")

    print("EVAL_FINETUNE_JRDB2X_SEEDS_OK")
