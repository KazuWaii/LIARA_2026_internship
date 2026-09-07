"""Linear probing sur les 27 actions de MM-Fi -- premiere metrique
comparable aux articles (precision de classification en aval), plutot que
les seuls diagnostics de sante de l'entrainement pretexte (perte, similarite
d'effondrement) utilises jusqu'ici. Meme principe que Point-JEPA : encodeur
gele, pooling mean+max sur les patches, tete lineaire entrainee sur les
features (note 3 : "on freeze the learned context encoder... utilize max and
mean pooling on the output of the Transformer encoder").

Compare 4 conditions : grille/points, pre-entraine/aleatoire -- le controle
aleatoire (encodeur jamais entraine) verifie que le pre-entrainement apporte
reellement quelque chose, pas seulement que le probing lineaire sait separer
les classes tout seul.

Mode fine-tuning ajoute le 3 septembre (soir), suite a la lecture de DeSPITE
(ICCV 2025, arXiv:2506.13897) : leurs propres resultats montrent que le
linear probing SOUS-ESTIME l'apport du pre-entrainement -- "linear probing
shows weaker performance, indicating pretraining benefits accumulate through
fine-tuning" (gains fine-tunes +1.4 a +4.7 points sur HMPEAR/MSR-Action3D,
contre un ecart bien plus faible en probing lineaire). `probe_lineaire`
prend maintenant `geler_encodeur` : True (defaut, comportement inchange) =
encodeur completement gele, comparable aux runs precedents ; False =
tokenizer+encodeur_contexte degeles et entraines avec la tete (taux
d'apprentissage plus faible que la tete, pratique standard), pour verifier
si l'ecart pre-entraine/aleatoire de last night (0.12 vs 0.10, quasi egalite)
persiste une fois l'encodeur autorise a s'adapter."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import nn
from torch.utils.data import DataLoader

import mmfi_dataset as mmfi
from dataset_jepa import MMFiDatasetJEPA, MMFiDatasetJEPAPoints, decouper_train_val
from modele import ModeleJEPA, ModeleJEPAPoints

CODES_ACTIONS = sorted(mmfi.ACTIONS.keys())
INDEX_ACTION = {code: i for i, code in enumerate(CODES_ACTIONS)}
NB_CLASSES = len(CODES_ACTIONS)


class _DatasetProbeMixin:
    def _label(self, idx):
        _, _, action = self.sequences[idx]
        return INDEX_ACTION[action]


class MMFiDatasetProbe(_DatasetProbeMixin, MMFiDatasetJEPA):
    def __getitem__(self, idx):
        return super().__getitem__(idx), self._label(idx)


class MMFiDatasetProbePoints(_DatasetProbeMixin, MMFiDatasetJEPAPoints):
    def __getitem__(self, idx):
        groupes, centres = super().__getitem__(idx)
        return groupes, centres, self._label(idx)


def extraire_patches(modele, lot, methode, appareil):
    """Encodeur de contexte, SANS masquage (tout le signal visible, comme a
    l'usage reel) -> (B, N, dim), pas encore pooles.

    Pas de @torch.no_grad() (retire le 3 septembre, mode fine-tuning) : les
    parametres geles ont requires_grad=False (cf. construire_modele), donc
    autograd ne construit deja aucun graphe pour eux -- le decorateur etait
    redondant en mode gele et bloquait le gradient en mode fine-tuning."""
    if methode == "points":
        groupes, centres = lot
        patches = modele.tokenizer(groupes.to(appareil), centres.to(appareil))
    else:
        patches = modele.tokenizer(lot.to(appareil))
    return modele.encodeur_contexte(patches)


def pooler_mean_max(z, stats=None):
    """Pooling mean+max sur les patches -> (B, 2*dim). `stats` standardise si
    fourni -- necessaire, le mean-pooling sur 192-512 patches amplifie
    l'effondrement partiel deja mesure pendant l'entrainement (~0.25-0.36 de
    similarite par patch) au point ou la plupart des fenetres s'ecrasent
    presque au meme point (similarite ~0.999 mesuree le 2 septembre)."""
    features = torch.cat([z.mean(dim=1), z.max(dim=1).values], dim=-1)
    if stats is not None:
        moyenne, ecart_type = stats
        features = (features - moyenne) / (ecart_type + 1e-6)
    return features


class PoolingAttentif(nn.Module):
    """Alternative au mean+max pooling (2 septembre) : au lieu de moyenner
    aveuglement sur tous les patches -- ce qui amplifie l'effondrement -- une
    requete apprise pondere les patches par attention, comme le "attentive
    probing" utilise par V-JEPA plutot qu'un simple mean-pool. Seul module
    entraine avec la tete lineaire ; l'encodeur reste gele."""

    def __init__(self, dim):
        super().__init__()
        self.requete = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)

    def forward(self, z):
        q = self.requete.expand(z.shape[0], -1, -1)
        k, v = self.proj_k(z), self.proj_v(z)
        poids = (q @ k.transpose(-1, -2) / (z.shape[-1] ** 0.5)).softmax(dim=-1)
        return (poids @ v).squeeze(1)


@torch.no_grad()
def calculer_stats_normalisation(modele, chargeur, methode, appareil, nb_lots=20):
    """Moyenne/ecart-type par dimension (pooling mean+max) sur un echantillon
    d'entrainement, calcules une seule fois et reutilises pour standardiser
    train et val."""
    toutes = []
    for i, lot in enumerate(chargeur):
        if i >= nb_lots:
            break
        entree = (lot[0], lot[1]) if methode == "points" else lot[0]
        z = extraire_patches(modele, entree, methode, appareil)
        toutes.append(pooler_mean_max(z))
    toutes = torch.cat(toutes, dim=0)
    return toutes.mean(dim=0, keepdim=True), toutes.std(dim=0, keepdim=True)


def construire_modele(methode, chemin_checkpoint, appareil, geler_encodeur=True):
    """geler_encodeur=False (3 septembre) : degele tokenizer+encodeur_contexte
    -- les deux SEULS sous-modules que extraire_patches exerce -- pour le
    mode fine-tuning. encodeur_cible/predicteur/masquage restent geles dans
    tous les cas (jamais utilises en probing/fine-tuning, gele = inoffensif)."""
    ModeleClasse = ModeleJEPAPoints if methode == "points" else ModeleJEPA
    modele = ModeleClasse().to(appareil)
    if chemin_checkpoint is not None:
        modele.load_state_dict(torch.load(chemin_checkpoint, map_location=appareil))
    for p in modele.parameters():
        p.requires_grad = False
    if geler_encodeur:
        modele.eval()
    else:
        for p in modele.tokenizer.parameters():
            p.requires_grad = True
        for p in modele.encodeur_contexte.parameters():
            p.requires_grad = True
    return modele


def probe_lineaire(
    methode, chemin_checkpoint, nb_etapes=2000, taille_lot=64,
    taux_apprentissage=1e-3, graine=0, pooling="mean_max", nb_workers=4,
    geler_encodeur=True, taux_apprentissage_encodeur=1e-4,
):
    """Entraine une tete lineaire (+ pooling attentif si `pooling="attentif"`)
    sur des features gelees (geler_encodeur=True, comportement d'origine) ou
    fine-tune tokenizer+encodeur_contexte en meme temps que la tete
    (geler_encodeur=False, 3 septembre soir -- cf. docstring du module,
    DeSPITE montre que le linear probing seul sous-estime l'apport du
    pre-entrainement). taux_apprentissage_encodeur : LR separe pour
    l'encodeur degele, plus faible que celui de la tete (pratique standard
    fine-tuning -- ne pas detruire les features pre-entrainees trop vite).

    Retourne (precision_val_finale, f1_macro_val_finale, historique_precision)."""
    assert pooling in ("mean_max", "attentif")
    torch.manual_seed(graine)
    appareil = "cuda" if torch.cuda.is_available() else "cpu"

    sequences_train, sequences_val = decouper_train_val()
    DatasetClasse = MMFiDatasetProbePoints if methode == "points" else MMFiDatasetProbe
    # num_workers>0 (3 septembre) : le chargement (lecture des .bin + FPS/kNN
    # pour la methode points) est CPU, pas GPU -- le premier run complet
    # (batch=64) tournait a 0% d'utilisation GPU, ~4s/pas, soit ~9h pour les
    # 4 conditions. persistent_workers evite de relancer les workers a chaque
    # epoque (fenetres courtes -> epoques frequentes).
    chargeur_train = DataLoader(DatasetClasse(sequences_train), batch_size=taille_lot, shuffle=True,
                                 num_workers=nb_workers, persistent_workers=nb_workers > 0)
    chargeur_val = DataLoader(DatasetClasse(sequences_val), batch_size=taille_lot, shuffle=True,
                               num_workers=nb_workers, persistent_workers=nb_workers > 0)

    modele = construire_modele(methode, chemin_checkpoint, appareil, geler_encodeur)
    dim = modele.tokenizer.dim

    parametres_entraines = []
    if pooling == "attentif":
        module_pooling = PoolingAttentif(dim).to(appareil)
        parametres_entraines += list(module_pooling.parameters())
        dim_features = dim
        stats = None
    else:
        module_pooling = None
        dim_features = 2 * dim
        stats = calculer_stats_normalisation(modele, chargeur_train, methode, appareil)

    tete = nn.Linear(dim_features, NB_CLASSES).to(appareil)
    groupes_optimiseur = [{"params": parametres_entraines + list(tete.parameters()), "lr": taux_apprentissage}]
    if not geler_encodeur:
        # LR separe, plus faible -- l'encodeur part deja d'un bon point
        # (pre-entraine ou non), la tete part de zero et a besoin d'apprendre
        # plus vite sans quoi elle domine le signal de perte au debut.
        parametres_encodeur = list(modele.tokenizer.parameters()) + list(modele.encodeur_contexte.parameters())
        groupes_optimiseur.append({"params": parametres_encodeur, "lr": taux_apprentissage_encodeur})
    optimiseur = torch.optim.AdamW(groupes_optimiseur)

    def _pas(lot):
        if methode == "points":
            groupes, centres, labels = lot
            entree = (groupes, centres)
        else:
            tenseur, labels = lot
            entree = tenseur
        z = extraire_patches(modele, entree, methode, appareil)
        features = module_pooling(z) if pooling == "attentif" else pooler_mean_max(z, stats)
        return features, labels.to(appareil)

    def _evaluer():
        tete.eval()
        if not geler_encodeur:
            modele.eval()
        correct, total, preds_tous, labels_tous = 0, 0, [], []
        with torch.no_grad():
            for i, lot in enumerate(chargeur_val):
                if i >= 20:
                    break
                features, labels = _pas(lot)
                preds = tete(features).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
                preds_tous.append(preds.cpu())
                labels_tous.append(labels.cpu())
        tete.train()
        if not geler_encodeur:
            modele.train()
        precision = correct / total if total else float("nan")
        preds_tous, labels_tous = torch.cat(preds_tous), torch.cat(labels_tous)
        f1s = []
        for c in range(NB_CLASSES):
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

    tete.train()
    if not geler_encodeur:
        modele.train()
    mode_txt = "gele" if geler_encodeur else "finetune"
    etape = 0
    historique = []
    while etape < nb_etapes:
        for lot in chargeur_train:
            if etape >= nb_etapes:
                break
            features, labels = _pas(lot)
            logits = tete(features)
            perte = nn.functional.cross_entropy(logits, labels)

            optimiseur.zero_grad()
            perte.backward()
            optimiseur.step()

            if etape % 50 == 0:
                precision, f1_macro = _evaluer()
                historique.append((etape, precision, f1_macro))
                print(f"  [{methode}, {mode_txt}, {os.path.basename(chemin_checkpoint) if chemin_checkpoint else 'ALEATOIRE'}] "
                      f"etape {etape:3d}/{nb_etapes}  perte={perte.item():.3f}  "
                      f"precision_val={precision:.3f}  f1_macro_val={f1_macro:.3f}")
            etape += 1

    precision_finale, f1_finale = _evaluer()
    return precision_finale, f1_finale, historique


if __name__ == "__main__":
    resultats = {}
    for methode, chemin in [
        ("grille", "mmfi-src/checkpoints_grille/final.pt"),
        ("grille", None),
        ("points", "mmfi-src/checkpoints_points/final.pt"),
        ("points", None),
    ]:
        nom = f"{methode}_{'preentraine' if chemin else 'aleatoire'}"
        print(f"\n=== {nom} ===")
        precision, f1, hist = probe_lineaire(methode, chemin)
        resultats[nom] = (precision, f1)

    print("\n\n=== RESUME ===")
    print(f"{'condition':25s} {'precision':>10s} {'f1_macro':>10s}")
    for nom, (p, f) in resultats.items():
        print(f"{nom:25s} {p:10.3f} {f:10.3f}")
