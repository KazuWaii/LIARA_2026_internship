"""Fine-tuning (pas seulement linear probing) du checkpoint mirroir V-JEPA
contre from-scratch -- suite a la lecture de DeSPITE (ICCV 2025) le 3
septembre soir, dont les propres resultats montrent que le linear probing
sous-estime l'apport du pre-entrainement (gains fine-tunes +1.4 a +4.7 points
contre un ecart plus faible en probing). Meme protocole que
log_eval_vjepa.txt (2000 pas, lot=64) mais geler_encodeur=False."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import probe_lineaire

if __name__ == "__main__":
    resultats = {}
    for chemin in ["mmfi-src/checkpoints_grille_vjepa/final.pt", None]:
        nom = f"grille_vjepa_finetune_{'preentraine' if chemin else 'scratch'}"
        print(f"\n=== {nom} ===")
        precision, f1, hist = probe_lineaire(
            "grille", chemin, nb_etapes=2000, taille_lot=64, geler_encodeur=False,
        )
        resultats[nom] = (precision, f1)

    print("\n\n=== RESUME ===")
    print(f"{'condition':35s} {'precision':>10s} {'f1_macro':>10s}")
    for nom, (p, f) in resultats.items():
        print(f"{nom:35s} {p:10.3f} {f:10.3f}")
