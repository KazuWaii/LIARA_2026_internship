"""Entrainement ModeleSTGCN (squelette verite-terrain SEUL, cf. modele_stgcn.py
/ squelette_dataset.MMFiDatasetSqueletteSeul) -- classification supervisee
directe, from scratch, MEME PROTOCOLE que toutes les autres comparaisons du
projet (2000 pas, lot=64, AdamW lr=1e-3, graines 0/1/2, decouper_train_val)
pour une comparabilite directe avec :
  - from-scratch classification seule (Transformer+LiDAR)      : 0.509 +- 0.044
  - squelette auxiliaire multi-tache (Transformer+LiDAR+MSE)   : 0.549 +- 0.009
  - spectre Laplacien + SVM regle (classique, LiDAR seul)      : 0.659 (meilleur LiDAR-only)
Objectif : quantifier le "plafond squelette" (cf. modele_stgcn.py docstring)."""

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset_jepa import decouper_train_val
from squelette_dataset import MMFiDatasetSqueletteSeul
from modele_stgcn import ModeleSTGCN


def entrainer_stgcn(nb_etapes=2000, taille_lot=64, taux_apprentissage=1e-3,
                     graine=0, nb_workers=4, suffixe=None):
    torch.manual_seed(graine)
    appareil = "cuda" if torch.cuda.is_available() else "cpu"

    sequences_train, sequences_val = decouper_train_val()
    chargeur_train = DataLoader(MMFiDatasetSqueletteSeul(sequences_train), batch_size=taille_lot,
                                 shuffle=True, num_workers=nb_workers, persistent_workers=nb_workers > 0)
    chargeur_val = DataLoader(MMFiDatasetSqueletteSeul(sequences_val), batch_size=taille_lot,
                               shuffle=True, num_workers=nb_workers, persistent_workers=nb_workers > 0)

    modele = ModeleSTGCN().to(appareil)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=taux_apprentissage, weight_decay=1e-4)

    def _pas(lot):
        squelette, labels = lot
        squelette, labels = squelette.to(appareil), labels.to(appareil)
        logits = modele(squelette)
        perte = nn.functional.cross_entropy(logits, labels)
        return perte, logits, labels

    @torch.no_grad()
    def _evaluer():
        modele.eval()
        correct, total, preds_tous, labels_tous = 0, 0, [], []
        for i, lot in enumerate(chargeur_val):
            if i >= 20:
                break
            _, logits, labels = _pas(lot)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            preds_tous.append(preds.cpu())
            labels_tous.append(labels.cpu())
        modele.train()
        precision = correct / total if total else float("nan")
        preds_tous, labels_tous = torch.cat(preds_tous), torch.cat(labels_tous)
        f1s = []
        for c in range(27):
            tp = ((preds_tous == c) & (labels_tous == c)).sum().item()
            fp = ((preds_tous == c) & (labels_tous != c)).sum().item()
            fn = ((preds_tous != c) & (labels_tous == c)).sum().item()
            if tp + fp + fn == 0:
                continue
            precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rappel_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1s.append(2 * precision_c * rappel_c / (precision_c + rappel_c) if (precision_c + rappel_c) > 0 else 0.0)
        f1_macro = sum(f1s) / len(f1s) if f1s else float("nan")
        return precision, f1_macro

    modele.train()
    etape = 0
    while etape < nb_etapes:
        for lot in chargeur_train:
            if etape >= nb_etapes:
                break
            perte, _, _ = _pas(lot)
            optimiseur.zero_grad()
            perte.backward()
            optimiseur.step()

            if etape % 50 == 0:
                precision, f1_macro = _evaluer()
                print(f"  [stgcn graine={graine}] etape {etape:4d}/{nb_etapes}  "
                      f"ce={perte.item():.3f}  precision_val={precision:.3f} f1_macro_val={f1_macro:.3f}", flush=True)
            etape += 1

    precision_finale, f1_finale = _evaluer()
    if suffixe:
        torch.save(modele.state_dict(), f"mmfi-src/checkpoints_{suffixe}_graine{graine}.pt")
    return precision_finale, f1_finale


if __name__ == "__main__":
    resultats = {}
    for graine in (0, 1, 2):
        print(f"\n=== stgcn_squelette graine={graine} ===")
        precision, f1 = entrainer_stgcn(graine=graine, suffixe="stgcn_squelette")
        resultats[graine] = (precision, f1)
        print(f"RESULTAT_PARTIEL stgcn_graine{graine} precision={precision:.3f} f1={f1:.3f}")

    print("\n\n=== RESUME MULTI-GRAINES (ST-GCN, squelette verite-terrain seul) ===")
    for g, (p, f) in resultats.items():
        print(f"graine={g}  precision={p:.3f}  f1={f:.3f}")
    ps = [resultats[g][0] for g in (0, 1, 2)]
    fs = [resultats[g][1] for g in (0, 1, 2)]
    print(f"stgcn: precision mean={statistics.mean(ps):.3f} stdev={statistics.stdev(ps):.3f}  "
          f"f1 mean={statistics.mean(fs):.3f} stdev={statistics.stdev(fs):.3f}")
    print("references : from-scratch LiDAR=0.509+-0.044 | squelette-auxiliaire=0.549+-0.009 | "
          "laplacien+SVM regle (meilleur LiDAR-only)=0.659")
    print("ENTRAINEMENT_STGCN_OK")
