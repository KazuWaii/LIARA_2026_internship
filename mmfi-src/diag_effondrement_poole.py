"""Diagnostic d'effondrement de la representation POOLEE -- remplace
diag_variance_poolee.py et diag_uniformite_poolee.py, devenus deux fichiers
au code identique a une chaine de print pres.

Contexte (2-3 septembre) : les diagnostics d'entrainement ne regardaient que
les patches individuels et ne voyaient donc pas que la representation POOLEE
(mean+max, celle que consomme le linear probe) s'effondrait -- au point que
les encodeurs pre-entraines etaient BATTUS par des encodeurs aleatoires en
probing. `perte_variance_poolee` (2 septembre) n'a rien change (0.9933 de
similarite apres 600 pas) ; `perte_uniformite_poolee` (3 septembre) l'a
corrige.

Ce script entraine un run court puis mesure directement la similarite
cosinus des features poolees sur un lot frais. Le poids de regularisation
est un ARGUMENT EXPLICITE : les deux anciens scripts laissaient la valeur
par defaut (0.0), si bien qu'ils mesuraient tous les deux un run SANS
regularisation malgre leurs docstrings.

    python mmfi-src/diag_effondrement_poole.py --poids 0     # sans regularisation
    python mmfi-src/diag_effondrement_poole.py --poids 1.0   # avec uniformite poolee
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

from entrainement import entrainer
from dataset_jepa import decouper_train_val, MMFiDatasetJEPA
from evaluation import extraire_patches, pooler_mean_max

def main():
    parseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parseur.add_argument("--poids", type=float, default=1.0,
                         help="poids_uniformite_poolee (0 = aucune regularisation)")
    parseur.add_argument("--temperature", type=float, default=2.0)
    parseur.add_argument("--nb-etapes", type=int, default=600)
    opts = parseur.parse_args()

    print(f"=== diagnostic effondrement poole : poids_uniformite_poolee={opts.poids} "
          f"temperature={opts.temperature} nb_etapes={opts.nb_etapes} ===")

    modele = entrainer(methode="grille", source="mmfi", nb_etapes=opts.nb_etapes,
                       intervalle_checkpoint=opts.nb_etapes,
                       poids_uniformite_poolee=opts.poids,
                       temperature_uniformite=opts.temperature)
    modele.eval()

    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    _, sequences_val = decouper_train_val()
    chargeur_val = DataLoader(MMFiDatasetJEPA(sequences_val), batch_size=16, shuffle=True, num_workers=0)
    lot = next(iter(chargeur_val))

    with torch.no_grad():
        z = extraire_patches(modele, lot, "grille", appareil)
        features = pooler_mean_max(z)

    sims = torch.nn.functional.cosine_similarity(features[0:1], features[1:], dim=-1)
    print(f"similarite cosinus (features poolees, echantillon 0 vs les {features.shape[0]-1} autres) : "
          f"{[round(v, 4) for v in sims.tolist()]}")
    print(f"moyenne: {sims.mean().item():.4f}")
    print("references : sans regularisation ~0.999 | perte_variance_poolee 0.9933 (inefficace) "
          "| perte_uniformite_poolee ~0.0-0.1 (corrige)")
    print("DIAG_EFFONDREMENT_POOLE_OK")


if __name__ == "__main__":
    main()
