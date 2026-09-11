"""Lanceur unique des pre-entrainements JEPA (remplace sept scripts qui
n'etaient chacun qu'un appel a `entrainement.entrainer` avec d'autres
arguments : diag_combine, diag_jrdb, diag_vjepa_mirror, diag_vjepa_mirror_reg,
entrainer_combine, entrainer_combine_jrdb2x, entrainer_vjepa).

Ces scripts s'executaient A L'IMPORT (aucune garde `__main__`), donc le
simple fait d'importer le module lancait un entrainement complet. Les
presets ci-dessous rendent chaque run explicite, comparable d'un coup d'oeil,
et inoffensif a l'import.

    python mmfi-src/lancer_jepa.py --liste
    python mmfi-src/lancer_jepa.py vjepa
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Regularisation anti-effondrement du pooling : sans elle, l'encodeur
# pre-entraine se fait battre par un encodeur aleatoire en linear probing
# (decouverte du 2 septembre, cf. README / evaluation.py).
REG_POOLEE = dict(poids_uniformite_poolee=1.0, temperature_uniformite=2.0)

PRESETS = {
    # --- diagnostics courts (500-600 pas) : valident qu'un run est sain
    #     (pas de NaN, perte qui descend, pas d'effondrement) avant de payer
    #     les 3000 pas complets.
    "diag-vjepa": dict(
        description="mirroir V-JEPA pur, reg_coeff=0.0 comme en production chez eux",
        args=dict(methode="grille", source="mmfi", nb_etapes=500, suffixe="grille_vjepa_diag")),
    "diag-vjepa-reg": dict(
        description="mirroir V-JEPA + uniformite poolee (le pur ne suffit pas sur notre donnee)",
        args=dict(methode="grille", source="mmfi", nb_etapes=500, suffixe="grille_vjepa_diag_reg",
                  **REG_POOLEE)),
    "diag-jrdb": dict(
        description="JRDB standalone, PROFIL_JRDB natif",
        args=dict(methode="grille", source="jrdb", nb_etapes=500, suffixe="jrdb_diag", **REG_POOLEE)),
    "diag-combine": dict(
        description="MM-Fi + JRDB (JRDB projete via PROFIL_MMFI)",
        args=dict(methode="grille", source="mmfi_jrdb", nb_etapes=500, suffixe="mmfi_jrdb_diag",
                  **REG_POOLEE)),

    # --- runs complets (3000 pas)
    "vjepa": dict(
        description="MM-Fi seul, mirroir V-JEPA + uniformite poolee (run de reference)",
        args=dict(methode="grille", source="mmfi", nb_etapes=3000, suffixe="grille_vjepa", **REG_POOLEE)),
    "combine": dict(
        description="MM-Fi + JRDB x1 (~25 sequences JRDB, <3% du corpus)",
        args=dict(methode="grille", source="mmfi_jrdb", nb_etapes=3000, suffixe="mmfi_jrdb", **REG_POOLEE)),
    "combine-jrdb2x": dict(
        description="MM-Fi + JRDB x2 (~50 sequences, reutilise le split test de JRDB)",
        args=dict(methode="grille", source="mmfi_jrdb2x", nb_etapes=3000, suffixe="mmfi_jrdb2x",
                  **REG_POOLEE)),
}


def main(argv=None):
    parseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parseur.add_argument("preset", nargs="?", choices=sorted(PRESETS),
                         help="nom du run a lancer")
    parseur.add_argument("--liste", action="store_true", help="liste les presets et sort")
    opts = parseur.parse_args(argv)

    if opts.liste or not opts.preset:
        print("presets disponibles :\n")
        for nom in sorted(PRESETS):
            p = PRESETS[nom]
            print(f"  {nom:16s} {p['description']}")
            print(f"  {'':16s} {p['args']}\n")
        return 0

    from entrainement import entrainer  # importe ici : torch au demarrage seulement si on lance
    preset = PRESETS[opts.preset]
    print(f"=== {opts.preset} : {preset['description']} ===")
    entrainer(**preset["args"])
    print(f"LANCER_JEPA_OK ({opts.preset})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
