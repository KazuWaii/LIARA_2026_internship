"""Entrainement RAPIDE sur les features Point-MAE precalculees
(extraire_pointmae.py -- encodeur ShapeNet-pretrained FIGE, features
(N_fenetres,8,768) deja poolees mean+max par frame sur les groupes).
Le "modele" entraine ici est volontairement minimal : pooling mean+max
temporel (8 frames -> 1536 dims) + UNE tete lineaire -- equivalent a un
LINEAR PROBING complet sur les features gelees (PointGST/litterature :
recommande a cette echelle plutot que le fine-tuning complet). Meme
protocole que toutes les autres comparaisons du projet (2000 pas, lot=64,
AdamW 1e-3, graines 0/1/2, decouper_train_val) pour comparabilite directe."""

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch import nn

from dataset_jepa import decouper_train_val
from entrainer_laplacien_v4 import _masque

DIM_FEAT = 768


class TeteTemporelle(nn.Module):
    def __init__(self, nb_classes=27):
        super().__init__()
        self.tete = nn.Linear(2 * DIM_FEAT, nb_classes)

    def forward(self, x):
        """x: (B,T,768) -> logits (B,nb_classes)."""
        pool = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        return self.tete(pool)


def _f1_macro(preds, labels, nb_classes=27):
    f1s = []
    for c in range(nb_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return sum(f1s) / len(f1s) if f1s else float("nan")


def entrainer_pointmae_probe(X_train, y_train, X_val, y_val, nb_etapes=2000, taille_lot=64,
                              taux_apprentissage=1e-3, graine=0, appareil="cuda"):
    torch.manual_seed(graine)
    modele = TeteTemporelle().to(appareil)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=taux_apprentissage)

    n_train = X_train.shape[0]
    X_val_t, y_val_t = X_val.to(appareil), y_val.to(appareil)

    def _evaluer():
        modele.eval()
        with torch.no_grad():
            logits = modele(X_val_t)
            preds = logits.argmax(dim=-1)
            precision = (preds == y_val_t).float().mean().item()
            f1 = _f1_macro(preds.cpu(), y_val_t.cpu())
        modele.train()
        return precision, f1

    modele.train()
    for etape in range(nb_etapes):
        idx = torch.randint(0, n_train, (taille_lot,))
        x, y = X_train[idx].to(appareil), y_train[idx].to(appareil)
        logits = modele(x)
        perte = nn.functional.cross_entropy(logits, y)
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
        if etape % 200 == 0:
            precision, f1 = _evaluer()
            print(f"  [pointmae_probe graine={graine}] etape {etape:4d}/{nb_etapes}  "
                  f"ce={perte.item():.3f} precision_val={precision:.3f} f1_macro_val={f1:.3f}", flush=True)

    return _evaluer()


if __name__ == "__main__":
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load("mmfi-src/features_pointmae.npz")
    X, y, envs, sujets = data["X"], data["y"], data["envs"], data["sujets"]
    print(f"features totales: X={X.shape}")

    seq_train, seq_val = decouper_train_val()
    m_train, m_val = _masque(seq_train, envs, sujets), _masque(seq_val, envs, sujets)
    X_train, y_train = torch.from_numpy(X[m_train]).float(), torch.from_numpy(y[m_train]).long()
    X_val, y_val = torch.from_numpy(X[m_val]).float(), torch.from_numpy(y[m_val]).long()
    print(f"train: {X_train.shape}  val: {X_val.shape}")

    resultats = {}
    for graine in (0, 1, 2):
        print(f"\n=== pointmae_probe graine={graine} ===")
        precision, f1 = entrainer_pointmae_probe(X_train, y_train, X_val, y_val, graine=graine, appareil=appareil)
        resultats[graine] = (precision, f1)
        print(f"RESULTAT_PARTIEL pointmae_probe_graine{graine} precision={precision:.3f} f1={f1:.3f}")

    print("\n\n=== RESUME MULTI-GRAINES (Point-MAE linear probe, encodeur ShapeNet fige) ===")
    for g, (p, f) in resultats.items():
        print(f"graine={g}  precision={p:.3f}  f1={f:.3f}")
    ps = [resultats[g][0] for g in (0, 1, 2)]
    fs = [resultats[g][1] for g in (0, 1, 2)]
    print(f"pointmae_probe: precision mean={statistics.mean(ps):.3f} stdev={statistics.stdev(ps):.3f}  "
          f"f1 mean={statistics.mean(fs):.3f} stdev={statistics.stdev(fs):.3f}")
    print("references (meme protocole exact) : from-scratch LiDAR=0.509+-0.044 | "
          "squelette-auxiliaire=0.549+-0.009 | distillation squelette->LiDAR=0.523+-0.037 | "
          "laplacien+SVM regle (meilleur LiDAR-only)=0.659")
    print("ENTRAINEMENT_POINTMAE_SEEDS_OK")
