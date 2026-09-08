"""Comme extraire_pointmae.py, mais charge l'encodeur ADAPTE par
entrainer_pointmae_jepa.py (continued JEPA pretraining sur MM-Fi, parti des
poids ShapeNet) au lieu du checkpoint ShapeNet brut -- reutilise les groupes
BRUTS deja precalcules (groupes_pointmae.npz) pour TOUTES les sequences
(train+val), evite de refaire l'isolation+FPS/kNN. Sortie directement
comparable a features_pointmae.npz (meme structure) pour evaluer avec
tuner_pointmae_classique.py/entrainer_pointmae_seeds.py sans les modifier."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from modele_pointmae import EncodeurPointMAE

CHECKPOINT_ADAPTE = "mmfi-src/checkpoints_pointmae_jepa.pt"

if __name__ == "__main__":
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load("mmfi-src/groupes_pointmae.npz")
    groupes, centres, y, envs, sujets = data["groupes"], data["centres"], data["y"], data["envs"], data["sujets"]
    print(f"groupes totaux: {groupes.shape}")

    print(f"chargement encodeur adapte ({CHECKPOINT_ADAPTE})...")
    encodeur = EncodeurPointMAE().to(appareil)
    encodeur.load_state_dict(torch.load(CHECKPOINT_ADAPTE, map_location=appareil))
    encodeur.eval()

    N, T, G, K, _ = groupes.shape
    print("forward GPU (encodage + pooling mean+max par frame)...")
    t0 = time.time()
    toutes_features = []
    taille_lot = 64
    with torch.no_grad():
        for debut in range(0, N, taille_lot):
            g = torch.from_numpy(groupes[debut:debut + taille_lot]).float().to(appareil)  # (b,T,G,K,3)
            c = torch.from_numpy(centres[debut:debut + taille_lot]).float().to(appareil)
            b = g.shape[0]
            tokens = encodeur(g.reshape(b * T, G, K, 3), c.reshape(b * T, G, 3))  # (b*T,G,384)
            pool = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)  # (b*T,768)
            toutes_features.append(pool.reshape(b, T, -1).cpu().numpy())
            if (debut // taille_lot) % 10 == 0:
                print(f"  {debut}/{N}  [{time.time()-t0:.0f}s]", flush=True)
    print(f"forward GPU termine en {time.time()-t0:.1f}s")

    toutes_features = np.concatenate(toutes_features, axis=0)
    print(f"features: {toutes_features.shape}")

    np.savez("mmfi-src/features_pointmae_jepa.npz",
             X=toutes_features, y=y, envs=envs, sujets=sujets)
    print("features sauvegardees -> mmfi-src/features_pointmae_jepa.npz")
    print("EXTRACTION_POINTMAE_JEPA_FEATURES_OK")
