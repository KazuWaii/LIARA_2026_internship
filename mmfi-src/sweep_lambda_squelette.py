"""Sweep rapide (4 septembre matin) de lambda_squelette (poids de la perte
auxiliaire de regression du squelette dans la perte conjointe, cf.
entrainer_squelette.py) -- 800 pas, 1 graine, avant de committer un run
complet 2000 pas x 3 graines sur la meilleure valeur. Valeurs testees
suggerees par la recherche (synthese workflow "har-lidar-research", 4
septembre matin) : {0.1, 1, 10} -- pas de ponderation automatique
(incertitude/GradNorm), poids fixe choisi manuellement d'abord."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entrainer_squelette import entrainer_squelette

if __name__ == "__main__":
    resultats = {}
    for lam in (0.1, 1.0, 10.0):
        print(f"\n=== lambda={lam} ===")
        precision, f1, mse = entrainer_squelette(nb_etapes=800, graine=0, lambda_squelette=lam, nb_workers=4)
        resultats[lam] = (precision, f1, mse)
        print(f"RESULTAT_SWEEP lambda={lam} precision={precision:.3f} f1={f1:.3f} mse={mse:.4f}")

    print("\n\n=== RESUME SWEEP ===")
    for lam, (p, f, m) in resultats.items():
        print(f"lambda={lam:6.2f}  precision={p:.3f}  f1={f:.3f}  mse={m:.4f}")
    print("SWEEP_LAMBDA_OK")
