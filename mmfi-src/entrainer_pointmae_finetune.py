"""Fine-tuning COMPLET de l'encodeur Point-MAE (contrairement a
entrainer_pointmae_seeds.py, qui gele l'encodeur -- linear probing, 0,934).
Utilise les groupes BRUTS precalcules (extraire_pointmae_groupes.py) pour
eviter de refaire l'isolation+FPS/kNN a chaque pas (~17s/pas mesure "a la
volee", cf. Tracking) -- seul le forward/backward de l'encodeur (rapide sur
GPU) est refait a chaque pas, necessaire puisque ses poids changent.

Taux d'apprentissage separe pour l'encodeur (deja pre-entraine, taux plus
faible) et la tete (initialisee aleatoirement, taux normal) -- meme
principe que le fine-tuning JEPA existant du projet (evaluation.py)."""

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch import nn

from dataset_jepa import decouper_train_val
from entrainer_laplacien_v4 import _masque
from modele_pointmae import ModeleClassificationPointMAE, charger_poids_pretrain

CHECKPOINT_PRETRAIN = r"D:\pretrained_checkpoints\point_mae\pretrain.pth"


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


def entrainer_pointmae_finetune(groupes_train, centres_train, y_train, groupes_val, centres_val, y_val,
                                 nb_etapes=2000, taille_lot=16, taux_encodeur=1e-5, taux_tete=1e-3,
                                 graine=0, appareil="cuda", verbeux_tous=100):
    torch.manual_seed(graine)
    modele = ModeleClassificationPointMAE(geler_encodeur=False).to(appareil)
    charger_poids_pretrain(modele.encodeur, CHECKPOINT_PRETRAIN, appareil)

    optimiseur = torch.optim.AdamW([
        dict(params=modele.encodeur.parameters(), lr=taux_encodeur),
        dict(params=modele.tete.parameters(), lr=taux_tete),
    ])

    n_train = groupes_train.shape[0]
    n_val = groupes_val.shape[0]

    def _lot(groupes, centres, y, idx):
        g = torch.from_numpy(groupes[idx]).float().to(appareil)
        c = torch.from_numpy(centres[idx]).float().to(appareil)
        yb = torch.from_numpy(y[idx]).long().to(appareil)
        return g, c, yb

    @torch.no_grad()
    def _evaluer():
        modele.eval()
        correct, total, preds_tous, y_tous = 0, 0, [], []
        for debut in range(0, min(n_val, 20 * taille_lot), taille_lot):
            idx = np.arange(debut, min(debut + taille_lot, n_val))
            g, c, yb = _lot(groupes_val, centres_val, y_val, idx)
            logits = modele(g, c)
            preds = logits.argmax(-1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
            preds_tous.append(preds.cpu())
            y_tous.append(yb.cpu())
        modele.train()
        precision = correct / total if total else float("nan")
        f1 = _f1_macro(torch.cat(preds_tous), torch.cat(y_tous))
        return precision, f1

    modele.train()
    rng = np.random.default_rng(graine)
    for etape in range(nb_etapes):
        idx = rng.integers(0, n_train, size=taille_lot)
        g, c, yb = _lot(groupes_train, centres_train, y_train, idx)
        logits = modele(g, c)
        perte = nn.functional.cross_entropy(logits, yb)
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
        if etape % verbeux_tous == 0:
            precision, f1 = _evaluer()
            print(f"  [pointmae_finetune graine={graine}] etape {etape:4d}/{nb_etapes}  "
                  f"ce={perte.item():.3f} precision_val={precision:.3f} f1_macro_val={f1:.3f}", flush=True)

    return _evaluer()


if __name__ == "__main__":
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load("mmfi-src/groupes_pointmae.npz")
    groupes, centres, y, envs, sujets = data["groupes"], data["centres"], data["y"], data["envs"], data["sujets"]
    print(f"groupes totaux: {groupes.shape}")

    seq_train, seq_val = decouper_train_val()
    m_train, m_val = _masque(seq_train, envs, sujets), _masque(seq_val, envs, sujets)
    groupes_train, centres_train, y_train = groupes[m_train], centres[m_train], y[m_train]
    groupes_val, centres_val, y_val = groupes[m_val], centres[m_val], y[m_val]
    print(f"train: {groupes_train.shape}  val: {groupes_val.shape}")

    resultats = {}
    for graine in (0, 1, 2):
        print(f"\n=== pointmae_finetune graine={graine} ===")
        precision, f1 = entrainer_pointmae_finetune(
            groupes_train, centres_train, y_train, groupes_val, centres_val, y_val,
            graine=graine, appareil=appareil)
        resultats[graine] = (precision, f1)
        print(f"RESULTAT_PARTIEL pointmae_finetune_graine{graine} precision={precision:.3f} f1={f1:.3f}")

    print("\n\n=== RESUME MULTI-GRAINES (Point-MAE fine-tuning complet) ===")
    for g, (p, f) in resultats.items():
        print(f"graine={g}  precision={p:.3f}  f1={f:.3f}")
    ps = [resultats[g][0] for g in (0, 1, 2)]
    fs = [resultats[g][1] for g in (0, 1, 2)]
    print(f"pointmae_finetune: precision mean={statistics.mean(ps):.3f} stdev={statistics.stdev(ps):.3f}  "
          f"f1 mean={statistics.mean(fs):.3f} stdev={statistics.stdev(fs):.3f}")
    print("reference (linear probing, encodeur fige) : 0.934+-0.006")
    print("ENTRAINEMENT_POINTMAE_FINETUNE_OK")
