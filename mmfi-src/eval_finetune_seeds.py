"""Repetition multi-graines (4 septembre matin) de la comparaison fine-tuning
combine-MM-Fi+JRDB vs from-scratch -- suite au resultat a graine=0 (egalite
stricte en precision, 0,514 vs 0,514, f1 leger mieux 0,523 vs 0,508) obtenu
via eval_finetune_combine.py, mais la trajectoire de validation etait bruitee
(0,458-0,565 sur les 500 derniers pas) donc un seul run ne tranche pas.
Repete graines 1 et 2 (graine=0 deja documentee dans README/tracking) pour
les deux conditions -- meme protocole exact (2000 pas, lot=64,
geler_encodeur=False)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import probe_lineaire

if __name__ == "__main__":
    resultats = {
        "combine_graine0": (0.514, 0.523),
        "scratch_graine0": (0.514, 0.508),
    }
    conditions = [
        ("combine", "mmfi-src/checkpoints_mmfi_jrdb/final.pt"),
        ("scratch", None),
    ]
    for graine in (1, 2):
        for nom, chemin in conditions:
            cle = f"{nom}_graine{graine}"
            print(f"\n=== {cle} ===")
            precision, f1, hist = probe_lineaire(
                "grille", chemin, nb_etapes=2000, taille_lot=64,
                geler_encodeur=False, graine=graine,
            )
            resultats[cle] = (precision, f1)
            print(f"RESULTAT_PARTIEL {cle} precision={precision:.3f} f1={f1:.3f}")

    print("\n\n=== RESUME MULTI-GRAINES ===")
    print(f"{'condition':25s} {'precision':>10s} {'f1_macro':>10s}")
    for nom, (p, f) in resultats.items():
        print(f"{nom:25s} {p:10.3f} {f:10.3f}")

    import statistics
    for prefix in ("combine", "scratch"):
        ps = [resultats[f"{prefix}_graine{g}"][0] for g in (0, 1, 2)]
        fs = [resultats[f"{prefix}_graine{g}"][1] for g in (0, 1, 2)]
        print(f"{prefix}: precision mean={statistics.mean(ps):.3f} stdev={statistics.stdev(ps):.3f}"
              f"  f1 mean={statistics.mean(fs):.3f} stdev={statistics.stdev(fs):.3f}")

    print("EVAL_FINETUNE_SEEDS_OK")
