"""Distillation squelette->LiDAR par "hallucination de modalite" (Hoffman
et al. CVPR'16 ; recette verifiee dans le code reel de Garcia, Morerio &
Murino, ECCV 2018, "Modality Distillation with Multiple Stream Networks",
github.com/ncgarcia/modality-distillation -- trouvee par le workflow de
recherche du 5 septembre, cf. Tracking). PAS notre precedente tentative
multi-tache naive (entrainer_squelette.py, +0,04 seulement, regression MSE
directe des coordonnees xyz par un encodeur PARTAGE) -- ici DEUX modeles
SEPARES : un enseignant squelette FIGE (ModeleSTGCN, deja entraine,
0,994+-0,007 de precision, cf. entrainer_stgcn.py) et un etudiant LiDAR-seul
(ModeleClassificationLidar) entraine avec TROIS termes de perte :

1. CE dure sur le vrai label (comme le from-scratch de base).
2. CE molle (temperature T) entre les distributions softmax etudiant/enseignant
   -- distille la geometrie de la frontiere de decision de l'enseignant, pas
   seulement sa prediction dure.
3. L2 entre la representation poolee de l'etudiant (projetee vers la
   dimension de l'enseignant par un petit adaptateur appris) et celle de
   l'enseignant -- distille aussi la geometrie interne des features.

L'enseignant ne recoit JAMAIS de gradient (`torch.no_grad()` + parametres
geles) -- a l'inference, seul l'etudiant (LiDAR seul) est utilise, l'
enseignant est jete (vrai "test-time modality dropout", contrairement a
MTGEA dont le code reel garde le squelette en entree meme au test, verifie
par le meme workflow de recherche).

Poids par defaut reprennent ceux verifies dans le code source de Garcia et
al. (lambda1=0.5, lambda2=0.5, lambda3=0.01, T=10) -- point de depart, pas
necessairement optimal pour notre echelle de donnees (sweep_distillation.py
teste des variantes avant le run complet 3 graines)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from dataset_jepa import decouper_train_val
from squelette_dataset import MMFiDatasetDistillation
from modele_stgcn import ModeleSTGCN
from modele_squelette import ModeleClassificationLidar

DIM_FEAT_ENSEIGNANT = 2 * 256  # ModeleSTGCN, dims=(64,128,256) par defaut
DIM_FEAT_ETUDIANT = 2 * 192    # ModeleClassificationLidar, dim=192 par defaut


def entrainer_distillation(
    chemin_checkpoint_enseignant, nb_etapes=2000, taille_lot=64, taux_apprentissage=1e-3,
    lambda1=0.5, lambda2=0.5, lambda3=0.01, temperature=10.0,
    graine=0, nb_workers=4, suffixe=None, verbeux_tous=50,
):
    torch.manual_seed(graine)
    appareil = "cuda" if torch.cuda.is_available() else "cpu"

    sequences_train, sequences_val = decouper_train_val()
    chargeur_train = DataLoader(MMFiDatasetDistillation(sequences_train), batch_size=taille_lot,
                                 shuffle=True, num_workers=nb_workers, persistent_workers=nb_workers > 0)
    chargeur_val = DataLoader(MMFiDatasetDistillation(sequences_val), batch_size=taille_lot,
                               shuffle=True, num_workers=nb_workers, persistent_workers=nb_workers > 0)

    enseignant = ModeleSTGCN().to(appareil)
    enseignant.load_state_dict(torch.load(chemin_checkpoint_enseignant, map_location=appareil))
    enseignant.eval()
    for p in enseignant.parameters():
        p.requires_grad = False

    etudiant = ModeleClassificationLidar().to(appareil)
    adaptateur = nn.Linear(DIM_FEAT_ETUDIANT, DIM_FEAT_ENSEIGNANT).to(appareil)

    optimiseur = torch.optim.AdamW(
        list(etudiant.parameters()) + list(adaptateur.parameters()), lr=taux_apprentissage)

    def _pas(lot):
        tenseur, squelette, labels = lot
        tenseur, squelette, labels = tenseur.to(appareil), squelette.to(appareil), labels.to(appareil)

        with torch.no_grad():
            logits_ens, feat_ens = enseignant(squelette, retourner_features=True)

        logits_etu, feat_etu = etudiant(tenseur, retourner_features=True)
        feat_etu_proj = adaptateur(feat_etu)

        perte_dure = F.cross_entropy(logits_etu, labels)
        log_p_etu = F.log_softmax(logits_etu / temperature, dim=-1)
        p_ens = F.softmax(logits_ens / temperature, dim=-1)
        perte_molle = -(p_ens * log_p_etu).sum(dim=-1).mean()
        perte_feat = F.mse_loss(feat_etu_proj, feat_ens)

        perte_distill = (1 - lambda1) * perte_dure + lambda1 * perte_molle
        perte = lambda2 * perte_distill + (1 - lambda2) * lambda3 * perte_feat
        return perte, perte_dure, perte_molle, perte_feat, logits_etu, labels

    @torch.no_grad()
    def _evaluer():
        etudiant.eval()
        correct, total, preds_tous, labels_tous = 0, 0, [], []
        for i, lot in enumerate(chargeur_val):
            if i >= 20:
                break
            _, _, _, _, logits, labels = _pas(lot)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            preds_tous.append(preds.cpu())
            labels_tous.append(labels.cpu())
        etudiant.train()
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

    etudiant.train()
    etape = 0
    while etape < nb_etapes:
        for lot in chargeur_train:
            if etape >= nb_etapes:
                break
            perte, perte_dure, perte_molle, perte_feat, _, _ = _pas(lot)
            optimiseur.zero_grad()
            perte.backward()
            optimiseur.step()

            if etape % verbeux_tous == 0:
                precision, f1_macro = _evaluer()
                print(f"  [distill graine={graine} T={temperature} l1={lambda1} l2={lambda2} l3={lambda3}] "
                      f"etape {etape:4d}/{nb_etapes}  dure={perte_dure.item():.3f} molle={perte_molle.item():.3f} "
                      f"feat={perte_feat.item():.4f}  precision_val={precision:.3f} f1_macro_val={f1_macro:.3f}",
                      flush=True)
            etape += 1

    precision_finale, f1_finale = _evaluer()
    if suffixe:
        torch.save(etudiant.state_dict(), f"mmfi-src/checkpoints_{suffixe}_graine{graine}.pt")
    return precision_finale, f1_finale


if __name__ == "__main__":
    precision, f1 = entrainer_distillation(
        chemin_checkpoint_enseignant="mmfi-src/checkpoints_stgcn_squelette_graine0.pt",
        nb_etapes=400, graine=0, verbeux_tous=25)
    print(f"\nRESULTAT_SMOKE distillation precision={precision:.3f} f1={f1:.3f}")
    print("ENTRAINEMENT_DISTILLATION_SMOKE_OK")
