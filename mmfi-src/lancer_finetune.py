"""Lanceur unique des evaluations en aval (linear probing et fine-tuning)
sur les checkpoints JEPA -- remplace six scripts qui n'etaient chacun qu'une
boucle autour de `evaluation.probe_lineaire` avec d'autres checkpoints,
graines et flag `geler_encodeur` : eval_vjepa, eval_finetune_vjepa,
eval_finetune_combine, eval_finetune_seeds, eval_finetune_jrdb2x_seeds,
eval_finetune_jrdb2x_graine2 (ce dernier n'existait que pour reprendre une
graine apres un MemoryError transitoire -- c'est maintenant `--graines 2`).

    python mmfi-src/lancer_finetune.py --liste
    python mmfi-src/lancer_finetune.py jrdb2x
    python mmfi-src/lancer_finetune.py jrdb2x --graines 2     # reprise d'une graine
"""

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# `None` comme chemin de checkpoint = encodeur non pre-entraine (aleatoire).
# C'est le controle indispensable : sans lui, un bon score en aval ne dit pas
# si le pre-entrainement sert a quelque chose (cf. decouverte du 2 septembre,
# ou les encodeurs pre-entraines etaient BATTUS par des aleatoires).
SCRATCH = None

PRESETS = {
    "vjepa-probe": dict(
        description="checkpoint V-JEPA MM-Fi, encodeur GELE (linear probing) vs aleatoire",
        conditions=[("vjepa_preentraine", "mmfi-src/checkpoints_grille_vjepa/final.pt"),
                    ("vjepa_aleatoire", SCRATCH)],
        graines=(0,), geler_encodeur=True),
    "vjepa-finetune": dict(
        description="meme checkpoint, encodeur DEGELE (fine-tuning) vs from-scratch",
        conditions=[("vjepa_preentraine", "mmfi-src/checkpoints_grille_vjepa/final.pt"),
                    ("vjepa_scratch", SCRATCH)],
        graines=(0,), geler_encodeur=False),
    "combine": dict(
        description="pre-entrainement MM-Fi+JRDB x1 vs from-scratch, fine-tune, 3 graines",
        conditions=[("combine", "mmfi-src/checkpoints_mmfi_jrdb/final.pt"),
                    ("scratch", SCRATCH)],
        graines=(0, 1, 2), geler_encodeur=False),
    "jrdb2x": dict(
        description="pre-entrainement MM-Fi+JRDB x2 vs from-scratch, fine-tune, 3 graines",
        conditions=[("jrdb2x", "mmfi-src/checkpoints_mmfi_jrdb2x/final.pt"),
                    ("scratch", SCRATCH)],
        graines=(0, 1, 2), geler_encodeur=False),
}


def main(argv=None):
    parseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parseur.add_argument("preset", nargs="?", choices=sorted(PRESETS))
    parseur.add_argument("--graines", type=int, nargs="+", default=None,
                         help="surcharge les graines du preset (reprise apres interruption)")
    parseur.add_argument("--nb-etapes", type=int, default=2000)
    parseur.add_argument("--liste", action="store_true")
    opts = parseur.parse_args(argv)

    if opts.liste or not opts.preset:
        print("presets disponibles :\n")
        for nom in sorted(PRESETS):
            p = PRESETS[nom]
            conditions = ", ".join(c for c, _ in p["conditions"])
            print(f"  {nom:16s} {p['description']}")
            print(f"  {'':16s} conditions=[{conditions}] graines={p['graines']} "
                  f"geler_encodeur={p['geler_encodeur']}\n")
        return 0

    from evaluation import probe_lineaire  # torch au demarrage seulement si on lance
    preset = PRESETS[opts.preset]
    graines = tuple(opts.graines) if opts.graines else preset["graines"]
    print(f"=== {opts.preset} : {preset['description']} ===")
    print(f"graines={graines}  geler_encodeur={preset['geler_encodeur']}  nb_etapes={opts.nb_etapes}")

    resultats = {}
    for nom, chemin in preset["conditions"]:
        precisions, f1s = [], []
        for graine in graines:
            print(f"\n=== {nom} graine={graine} ===")
            precision, f1, _hist = probe_lineaire(
                "grille", chemin, nb_etapes=opts.nb_etapes, taille_lot=64,
                geler_encodeur=preset["geler_encodeur"], graine=graine)
            print(f"RESULTAT_PARTIEL {nom}_graine{graine} precision={precision:.3f} f1={f1:.3f}")
            precisions.append(precision)
            f1s.append(f1)
        resultats[nom] = (precisions, f1s)

    print("\n\n=== RESUME ===")
    print(f"{'condition':28s} {'precision':>18s} {'f1_macro':>18s}")
    for nom, (precisions, f1s) in resultats.items():
        if len(precisions) > 1:
            p = f"{statistics.mean(precisions):.3f} +/- {statistics.stdev(precisions):.3f}"
            f = f"{statistics.mean(f1s):.3f} +/- {statistics.stdev(f1s):.3f}"
        else:
            p, f = f"{precisions[0]:.3f}", f"{f1s[0]:.3f}"
        print(f"{nom:28s} {p:>18s} {f:>18s}")
        print(f"{'':28s} graines: {', '.join(f'{v:.3f}' for v in precisions)}")
    print(f"LANCER_FINETUNE_OK ({opts.preset})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
