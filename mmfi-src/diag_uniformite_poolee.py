"""Diagnostic (3 septembre) : valide si perte_uniformite_poolee (perte.py,
remplace perte_variance_poolee du 2 septembre qui s'est avere inefficace --
0.9933 de similarite apres 600 pas, quasi inchange) corrige l'effondrement du
pooling mesure en linear probing. Entraine 600 pas (grille, MM-Fi) avec la
perte a jour, puis mesure directement la similarite cosinus des features
poolees (mean+max, comme le linear probe) sur un lot frais -- doit etre
nettement sous ~0.99 si la regularisation fonctionne."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

from entrainement import entrainer
from dataset_jepa import decouper_train_val, MMFiDatasetJEPA
from evaluation import extraire_patches, pooler_mean_max

modele = entrainer(methode="grille", source="mmfi", nb_etapes=600, intervalle_checkpoint=600)
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
print(f"moyenne: {sims.mean().item():.4f}  (reference perte_variance_poolee: 0.9933, sans regularisation: ~0.999)")
