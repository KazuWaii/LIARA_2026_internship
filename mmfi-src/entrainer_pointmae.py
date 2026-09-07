"""Fine-tuning de ModeleClassificationPointMAE (modele_pointmae.py) --
encodeur Point-MAE pre-entraine sur ShapeNet (poids officiels charges,
cf. modele_pointmae.charger_poids_pretrain), isolation de la personne +
groupement FPS+kNN par frame (dataset_pointmae.py). Meme protocole
d'evaluation que tout le reste du projet (precision/f1 macro sur 27
classes, decouper_train_val) pour comparabilite directe.

`geler_encodeur=True` (linear probing, comme evaluation.py) : seule la tete
(pool+Linear) s'entraine -- recommande par la recherche PointGST comme
alternative au fine-tuning complet a cette echelle de donnees (surapprentissage
du fine-tuning complet deja documente pour notre propre JEPA, cf. README).
`geler_encodeur=False` : fine-tuning complet (encodeur + tete), taux
d'apprentissage plus faible pour l'encodeur (deja pre-entraine)."""

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset_jepa import decouper_train_val
from dataset_pointmae import MMFiDatasetPointMAE
from modele_pointmae import ModeleClassificationPointMAE, charger_poids_pretrain

# Checkpoint officiel Point-MAE (poids ShapeNet-pretrained, 332 Mo) --
# telecharge depuis github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth
# (poids seuls, aucun code de leur depot execute, cf. modele_pointmae.py).
# Stocke sur D: (comme JRDB/ONCE, cf. README) -- pas dans le depot git (trop
# volumineux) ; re-telecharger a cette URL si absent.
CHECKPOINT_PRETRAIN = r"D:\pretrained_checkpoints\point_mae\pretrain.pth"


def entrainer_pointmae(nb_etapes=2000, taille_lot=32, taux_apprentissage_tete=1e-3,
                        taux_apprentissage_encodeur=1e-5, geler_encodeur=True,
                        graine=0, nb_workers=4, suffixe=None, verbeux_tous=50):
    torch.manual_seed(graine)
    appareil = "cuda" if torch.cuda.is_available() else "cpu"

    sequences_train, sequences_val = decouper_train_val()
    chargeur_train = DataLoader(MMFiDatasetPointMAE(sequences_train), batch_size=taille_lot,
                                 shuffle=True, num_workers=nb_workers, persistent_workers=nb_workers > 0)
    chargeur_val = DataLoader(MMFiDatasetPointMAE(sequences_val), batch_size=taille_lot,
                               shuffle=True, num_workers=nb_workers, persistent_workers=nb_workers > 0)

    modele = ModeleClassificationPointMAE(geler_encodeur=geler_encodeur).to(appareil)
    charger_poids_pretrain(modele.encodeur, CHECKPOINT_PRETRAIN, appareil)
    if geler_encodeur:
        modele.encodeur.eval()

    if geler_encodeur:
        optimiseur = torch.optim.AdamW(modele.tete.parameters(), lr=taux_apprentissage_tete)
    else:
        optimiseur = torch.optim.AdamW([
            dict(params=modele.encodeur.parameters(), lr=taux_apprentissage_encodeur),
            dict(params=modele.tete.parameters(), lr=taux_apprentissage_tete),
        ])

    def _pas(lot):
        groupes, centres, labels = lot
        groupes, centres, labels = groupes.to(appareil), centres.to(appareil), labels.to(appareil)
        logits = modele(groupes, centres)
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
        if geler_encodeur:
            modele.encodeur.eval()
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
    if geler_encodeur:
        modele.encodeur.eval()
    etape = 0
    while etape < nb_etapes:
        for lot in chargeur_train:
            if etape >= nb_etapes:
                break
            perte, _, _ = _pas(lot)
            optimiseur.zero_grad()
            perte.backward()
            optimiseur.step()

            if etape % verbeux_tous == 0:
                precision, f1_macro = _evaluer()
                print(f"  [pointmae graine={graine} geler={geler_encodeur}] etape {etape:4d}/{nb_etapes}  "
                      f"ce={perte.item():.3f}  precision_val={precision:.3f} f1_macro_val={f1_macro:.3f}", flush=True)
            etape += 1

    precision_finale, f1_finale = _evaluer()
    if suffixe:
        torch.save(modele.state_dict(), f"mmfi-src/checkpoints_{suffixe}_graine{graine}.pt")
    return precision_finale, f1_finale


if __name__ == "__main__":
    import time
    t0 = time.time()
    precision, f1 = entrainer_pointmae(nb_etapes=40, taille_lot=16, graine=0, nb_workers=0, verbeux_tous=10)
    print(f"\nRESULTAT_SMOKE precision={precision:.3f} f1={f1:.3f}  duree={time.time()-t0:.1f}s")
    print("ENTRAINEMENT_POINTMAE_SMOKE_OK")
