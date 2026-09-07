"""Run complet multi-graines (4 septembre matin) du meilleur lambda trouve
par le sweep (lambda=1.0 : 0,523/0,545 a 800 pas, contre 0,421/0,407 pour
lambda=0.1 et 0,472/0,453 pour lambda=10, cf. log_sweep_lambda.txt) --
2000 pas, memes graines 0/1/2 que toutes nos autres comparaisons
(from-scratch classification-seule : 0,509+-0,044 ; JEPA MM-Fi+JRDB x1/x2)
pour une comparaison directe et rigoureuse (pas de conclusion sur une seule
graine, lecon retenue de la nuit precedente)."""

import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entrainer_squelette import entrainer_squelette

if __name__ == "__main__":
    resultats = {}
    for graine in (0, 1, 2):
        print(f"\n=== squelette_lambda1_graine{graine} ===")
        precision, f1, mse = entrainer_squelette(nb_etapes=2000, graine=graine, lambda_squelette=1.0,
                                                   nb_workers=4, suffixe="squelette_lambda1")
        resultats[graine] = (precision, f1, mse)
        print(f"RESULTAT_PARTIEL squelette_lambda1_graine{graine} precision={precision:.3f} f1={f1:.3f} mse={mse:.4f}")

    print("\n\n=== RESUME MULTI-GRAINES (squelette, lambda=1.0) ===")
    for g, (p, f, m) in resultats.items():
        print(f"graine={g}  precision={p:.3f}  f1={f:.3f}  mse={m:.4f}")

    ps = [resultats[g][0] for g in (0, 1, 2)]
    fs = [resultats[g][1] for g in (0, 1, 2)]
    print(f"squelette (lambda=1.0): precision mean={statistics.mean(ps):.3f} stdev={statistics.stdev(ps):.3f}"
          f"  f1 mean={statistics.mean(fs):.3f} stdev={statistics.stdev(fs):.3f}")
    print("reference from-scratch (classification seule, 3 graines, deja mesure): precision mean=0.509 stdev=0.044  f1 mean=0.519 stdev=0.042")
    print("ENTRAINEMENT_SQUELETTE_SEEDS_OK")
