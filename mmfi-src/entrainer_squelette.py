"""Entrainement multi-tache (4 septembre matin) : classification (27 actions)
+ regression du squelette 3D verite-terrain, perte conjointe, from scratch --
teste si le signal auxiliaire dense (squelette, jamais utilise jusqu'ici,
cf. squelette_dataset.py) ameliore la classification par rapport au
from-scratch classification-seule deja mesure (0,509+-0,044 sur 3 graines,
cf. README/Tracking 3-septembre). Meme protocole que probe_lineaire
(geler_encodeur=False) pour comparabilite directe : 2000 pas, lot=64,
memes graines 0/1/2.

Ponderation perte : lambda_squelette choisi empiriquement (smoke test) pour
que les deux termes soient du meme ordre de grandeur au debut de
l'entrainement (CE ~ ln(27)=3.3 a l'initialisation, MSE squelette mesuree
empiriquement sur des cibles centrees ~0.01-0.05 selon l'amplitude des
mouvements -- cf. smoke test)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset_jepa import decouper_train_val
from squelette_dataset import MMFiDatasetSquelette
from modele_squelette import ModeleSquelette


def entrainer_squelette(
    nb_etapes=2000, taille_lot=64, taux_apprentissage=1e-3,
    lambda_squelette=1.0, graine=0, nb_workers=4,
    chemin_checkpoint_jepa=None, geler_encodeur_jepa=False,
    suffixe=None,
):
    torch.manual_seed(graine)
    appareil = "cuda" if torch.cuda.is_available() else "cpu"

    sequences_train, sequences_val = decouper_train_val()
    chargeur_train = DataLoader(MMFiDatasetSquelette(sequences_train), batch_size=taille_lot,
                                 shuffle=True, num_workers=nb_workers, persistent_workers=nb_workers > 0)
    chargeur_val = DataLoader(MMFiDatasetSquelette(sequences_val), batch_size=taille_lot,
                               shuffle=True, num_workers=nb_workers, persistent_workers=nb_workers > 0)

    modele = ModeleSquelette().to(appareil)
    if chemin_checkpoint_jepa is not None:
        modele.charger_poids_jepa(chemin_checkpoint_jepa, appareil)
        if geler_encodeur_jepa:
            for p in list(modele.tokenizer.parameters()) + list(modele.encodeur.parameters()):
                p.requires_grad = False

    optimiseur = torch.optim.AdamW([p for p in modele.parameters() if p.requires_grad], lr=taux_apprentissage)

    def _pas(lot):
        tenseur, squelette_cible, labels = lot
        tenseur, squelette_cible, labels = tenseur.to(appareil), squelette_cible.to(appareil), labels.to(appareil)
        logits, squelette_pred = modele(tenseur)
        perte_ce = nn.functional.cross_entropy(logits, labels)
        perte_mse = nn.functional.mse_loss(squelette_pred, squelette_cible)
        perte = perte_ce + lambda_squelette * perte_mse
        return perte, perte_ce, perte_mse, logits, labels

    @torch.no_grad()
    def _evaluer():
        modele.eval()
        correct, total, preds_tous, labels_tous, mses = 0, 0, [], [], []
        for i, lot in enumerate(chargeur_val):
            if i >= 20:
                break
            _, _, mse, logits, labels = _pas(lot)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            preds_tous.append(preds.cpu())
            labels_tous.append(labels.cpu())
            mses.append(mse.item())
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
        return precision, f1_macro, sum(mses) / len(mses) if mses else float("nan")

    modele.train()
    etape = 0
    while etape < nb_etapes:
        for lot in chargeur_train:
            if etape >= nb_etapes:
                break
            perte, perte_ce, perte_mse, _, _ = _pas(lot)
            optimiseur.zero_grad()
            perte.backward()
            optimiseur.step()

            if etape % 50 == 0:
                precision, f1_macro, mse_val = _evaluer()
                print(f"  [squelette graine={graine}] etape {etape:4d}/{nb_etapes}  "
                      f"ce={perte_ce.item():.3f} mse={perte_mse.item():.4f}  "
                      f"precision_val={precision:.3f} f1_macro_val={f1_macro:.3f} mse_val={mse_val:.4f}")
            etape += 1

    precision_finale, f1_finale, mse_finale = _evaluer()

    if suffixe:
        torch.save(modele.state_dict(), f"mmfi-src/checkpoints_{suffixe}_graine{graine}.pt")

    return precision_finale, f1_finale, mse_finale


if __name__ == "__main__":
    precision, f1, mse = entrainer_squelette(graine=0)
    print(f"\nRESULTAT_PARTIEL squelette_graine0 precision={precision:.3f} f1={f1:.3f} mse={mse:.4f}")
    print("ENTRAINEMENT_SQUELETTE_SMOKE_OK")
