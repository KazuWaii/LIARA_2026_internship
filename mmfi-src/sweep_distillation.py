"""Sweep de la temperature T et du poids de perte de features lambda3 pour
la distillation squelette->LiDAR (entrainer_distillation.py), avant le run
complet 3 graines -- meme logique que sweep_lambda_squelette.py (4 septembre).
lambda1/lambda2 restent aux valeurs par defaut de Garcia et al. (0.5/0.5),
seuls T et lambda3 sont balayes ici (les deux hyperparametres les plus
susceptibles de ne pas transferer tels quels de leur echelle de donnees --
NTU/UCF101, dizaines de milliers d'exemples -- a la notre, ~1000 sequences).

800 pas, 1 graine, comme le sweep squelette-auxiliaire du 4 septembre."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entrainer_distillation import entrainer_distillation

CHECKPOINT_ENSEIGNANT = "mmfi-src/checkpoints_stgcn_squelette_graine0.pt"

if __name__ == "__main__":
    configs = [
        dict(temperature=2.0, lambda3=0.01),
        dict(temperature=4.0, lambda3=0.01),
        dict(temperature=10.0, lambda3=0.01),
        dict(temperature=4.0, lambda3=0.1),
        dict(temperature=4.0, lambda3=1.0),
    ]
    resultats = []
    for cfg in configs:
        print(f"\n=== sweep distillation {cfg} ===")
        precision, f1 = entrainer_distillation(
            chemin_checkpoint_enseignant=CHECKPOINT_ENSEIGNANT,
            nb_etapes=800, graine=0, verbeux_tous=100, **cfg)
        resultats.append((cfg, precision, f1))
        print(f"RESULTAT_PARTIEL {cfg} precision={precision:.3f} f1={f1:.3f}")

    print("\n\n=== RESUME SWEEP DISTILLATION (800 pas, graine=0) ===")
    for cfg, p, f in resultats:
        print(f"{cfg}: precision={p:.3f} f1={f:.3f}")
    print("references (from-scratch classification seule, meme protocole, a 800 pas non mesure "
          "specifiquement -- reference finale a 2000 pas): 0.509+-0.044")
    print("SWEEP_DISTILLATION_OK")
