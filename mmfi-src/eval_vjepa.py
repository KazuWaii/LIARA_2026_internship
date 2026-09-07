"""Linear probing du checkpoint mirroir V-JEPA (grille, regularisation
reactivee) contre aleatoire -- meme methodologie que log_evaluation_v2.txt
(3 septembre), pour verifier si la combinaison masquage V-JEPA + perte L1
+ scheduleurs + uniformite poolee produit une representation utile en aval,
pas seulement un entrainement sain (sim_poolee bas ne garantit pas a lui
seul l'utilite downstream, cf. points_preentraine la veille : effondrement
resolu pendant l'entrainement mais toujours inutile en probing)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import probe_lineaire

if __name__ == "__main__":
    resultats = {}
    for chemin in ["mmfi-src/checkpoints_grille_vjepa/final.pt", None]:
        nom = f"grille_vjepa_{'preentraine' if chemin else 'aleatoire'}"
        print(f"\n=== {nom} ===")
        precision, f1, hist = probe_lineaire("grille", chemin, nb_etapes=2000, taille_lot=64)
        resultats[nom] = (precision, f1)

    print("\n\n=== RESUME ===")
    print(f"{'condition':30s} {'precision':>10s} {'f1_macro':>10s}")
    for nom, (p, f) in resultats.items():
        print(f"{nom:30s} {p:10.3f} {f:10.3f}")
