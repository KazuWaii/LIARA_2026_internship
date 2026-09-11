"""Entrainement RAPIDE sur les features Point-MAE precalculees
(extraire_pointmae.py -- encodeur ShapeNet-pretrained FIGE, features
(N_fenetres,8,768) deja poolees mean+max par frame sur les groupes).
Le "modele" entraine ici est volontairement minimal : pooling mean+max
temporel (8 frames -> 1536 dims) + UNE tete lineaire -- equivalent a un
LINEAR PROBING complet sur les features gelees (PointGST/litterature :
recommande a cette echelle plutot que le fine-tuning complet). Meme
protocole que toutes les autres comparaisons du projet (2000 pas, lot=64,
AdamW 1e-3, graines 0/1/2) pour comparabilite directe.

Evalue sur LES DEUX splits (protocole.evaluer_deux_splits) : split sujet
(comparaisons internes du projet) ET split environnement E01-E03/E04, seul
protocole homogene avec la litterature de reference (arXiv:2509.23255).
Avant le 11 septembre ce script ne calculait que le split sujet, alors que
le README rapportait aussi 0,908 +/- 0,009 en split environnement -- chiffre
obtenu par une edition ad hoc jamais commitee, donc non reproductible."""

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch import nn

import protocole as proto

DIM_FEAT = 768
GRAINES = (0, 1, 2)


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


def probe_multi_graines(X_train, y_train, X_val, y_val, _groupes_train,
                        graines=GRAINES, appareil=None):
    """Signature compatible protocole.evaluer_deux_splits (les groupes sujet
    ne servent pas ici : pas de validation croisee, la tete lineaire n'a pas
    d'hyperparametre a regler). Repete l'entrainement sur `graines` et
    retourne moyenne/ecart-type."""
    appareil = appareil or ("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = torch.from_numpy(X_train).float()
    ytr = torch.from_numpy(y_train).long()
    Xva = torch.from_numpy(X_val).float()
    yva = torch.from_numpy(y_val).long()

    precisions, f1s = [], []
    for graine in graines:
        print(f"\n  === graine={graine} ===")
        precision, f1 = entrainer_pointmae_probe(Xtr, ytr, Xva, yva, graine=graine, appareil=appareil)
        print(f"  RESULTAT_PARTIEL graine{graine} precision={precision:.3f} f1={f1:.3f}")
        precisions.append(precision)
        f1s.append(f1)

    resultat = dict(
        precision=statistics.mean(precisions),
        precision_std=statistics.stdev(precisions) if len(precisions) > 1 else 0.0,
        f1=statistics.mean(f1s),
        f1_std=statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
        precisions=precisions,
    )
    print(f"  -> precision {resultat['precision']:.3f} +/- {resultat['precision_std']:.3f}  "
          f"(graines: {', '.join(f'{p:.3f}' for p in precisions)})")
    return resultat


if __name__ == "__main__":
    data = np.load("mmfi-src/features_pointmae.npz")
    X, y, envs, sujets = data["X"], data["y"], data["envs"], data["sujets"]
    print(f"features totales: X={X.shape}")

    resultats = proto.evaluer_deux_splits(
        X, y, envs, sujets, probe_multi_graines,
        nom_experience="POINT-MAE LINEAR PROBE (encodeur ShapeNet fige)")

    proto.resumer("POINT-MAE LINEAR PROBE", resultats, cles=("precision", "precision_std", "f1"))
    print("\nreferences (meme protocole exact, split sujet) : from-scratch LiDAR=0.509+-0.044 | "
          "squelette-auxiliaire=0.549+-0.009 | distillation squelette->LiDAR=0.523+-0.037 | "
          "laplacien+SVM regle=0.659")
    print("ENTRAINEMENT_POINTMAE_SEEDS_OK")
