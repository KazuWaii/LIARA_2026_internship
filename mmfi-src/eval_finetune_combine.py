"""Fine-tuning du checkpoint pre-entraine sur MM-Fi+JRDB combine (cf.
entrainer_combine.py, checkpoints_mmfi_jrdb/final.pt) -- teste si l'ajout de
JRDB ferme l'ecart trouve avec MM-Fi seul (0,486 pre-entraine vs 0,514
from-scratch, cf. README section fine-tuning). Meme protocole exact que
eval_finetune_vjepa.py (2000 pas, lot=64, geler_encodeur=False, graine=0) --
la condition from-scratch n'est pas re-executee ici : meme appel
deterministe (graine=0) deja documente (0,514 precision / 0,508 f1 macro)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import probe_lineaire

if __name__ == "__main__":
    precision, f1, hist = probe_lineaire(
        "grille", "mmfi-src/checkpoints_mmfi_jrdb/final.pt",
        nb_etapes=2000, taille_lot=64, geler_encodeur=False,
    )
    print("\n\n=== RESUME ===")
    print(f"{'condition':35s} {'precision':>10s} {'f1_macro':>10s}")
    print(f"{'grille_mmfi_jrdb_finetune_preentraine':35s} {precision:10.3f} {f1:10.3f}")
    print(f"{'grille_vjepa_finetune_scratch (reference)':35s} {0.514:10.3f} {0.508:10.3f}")
    print("EVAL_FINETUNE_COMBINE_OK")
