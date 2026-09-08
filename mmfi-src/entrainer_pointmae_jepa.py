"""Pre-entrainement JEPA continu de l'encodeur Point-MAE (modele_pointmae_jepa.py),
PARTANT des poids ShapeNet-pretrained -- adaptation de domaine auto-supervisee
a MM-Fi (masquage de groupes + prediction en espace d'embedding), SANS les
labels de classification. Entraine sur les groupes bruts DEJA precalcules
(groupes_pointmae.npz, cf. extraire_pointmae_groupes.py -- meme cache que le
fine-tuning supervise), filtre au split TRAIN uniquement (pour ne pas
contaminer l'evaluation en aval, meme si l'auto-supervision n'utilise aucun
label). Chaque frame d'une fenetre est un echantillon independant du pretexte
(pas de masquage temporel, cf. modele_pointmae_jepa.py)."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from dataset_jepa import decouper_train_val
from entrainer_laplacien_v4 import _masque
from modele_pointmae import charger_poids_pretrain
from modele_pointmae_jepa import ModelePointMAEJepa
from perte import perte_uniformite_poolee

CHECKPOINT_SHAPENET = r"D:\pretrained_checkpoints\point_mae\pretrain.pth"


@torch.no_grad()
def _similarite_poolee(embeddings):
    """Diagnostic d'effondrement (comme diagnostic_effondrement_poolee du
    reste du projet) : similarite cosinus moyenne entre paires DIFFERENTES
    du lot, sur la representation mean+max poolee -- doit rester basse
    (proche de 0) pour un entrainement sain, PAS proche de 1 (effondrement)."""
    pool = torch.cat([embeddings.mean(dim=1), embeddings.max(dim=1).values], dim=-1)
    pool = nn_normalize(pool)
    sim = pool @ pool.T
    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    return sim[mask].mean().item()


def nn_normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def entrainer_pointmae_jepa(groupes_train, centres_train, nb_etapes=3000, taille_lot=64,
                             taux_apprentissage=1e-4, poids_uniformite=1.0, masquage_bloc=False,
                             graine=0, appareil="cuda", verbeux_tous=100, suffixe=None):
    torch.manual_seed(graine)
    modele = ModelePointMAEJepa().to(appareil)
    charger_poids_pretrain(modele.encodeur_contexte, CHECKPOINT_SHAPENET, appareil)
    modele.encodeur_cible.load_state_dict(modele.encodeur_contexte.state_dict())

    optimiseur = torch.optim.AdamW(
        [p for n, p in modele.named_parameters() if not n.startswith("encodeur_cible")],
        lr=taux_apprentissage)

    n_train = groupes_train.shape[0]
    modele.train()
    t0 = time.time()
    for etape in range(nb_etapes):
        idx = np.random.randint(0, n_train, size=taille_lot)
        g = torch.from_numpy(groupes_train[idx]).float().to(appareil)
        c = torch.from_numpy(centres_train[idx]).float().to(appareil)

        prediction, cible, contexte_encode = modele(g, c, masquage_bloc=masquage_bloc)
        perte_l1 = (prediction - cible).abs().mean()
        perte_unif = perte_uniformite_poolee(contexte_encode)
        perte = perte_l1 + poids_uniformite * perte_unif

        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
        modele.mettre_a_jour_cible()

        if etape % verbeux_tous == 0:
            sim = _similarite_poolee(contexte_encode.detach())
            print(f"  [pointmae_jepa graine={graine}] etape {etape:4d}/{nb_etapes}  "
                  f"l1={perte_l1.item():.4f} unif={perte_unif.item():.4f} sim_poolee={sim:.3f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    if suffixe:
        torch.save(modele.encodeur_contexte.state_dict(), f"mmfi-src/checkpoints_{suffixe}.pt")
    return modele


if __name__ == "__main__":
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load("mmfi-src/groupes_pointmae.npz")
    groupes, centres, y, envs, sujets = data["groupes"], data["centres"], data["y"], data["envs"], data["sujets"]

    seq_train, seq_val = decouper_train_val()
    m_train = _masque(seq_train, envs, sujets)
    groupes_train, centres_train = groupes[m_train], centres[m_train]
    print(f"fenetres train: {groupes_train.shape}")

    # aplatit (N_fenetres, T=8, G, K, 3) -> (N_fenetres*8, G, K, 3) : chaque
    # frame est un echantillon independant du pretexte JEPA
    T = groupes_train.shape[1]
    groupes_train = groupes_train.reshape(-1, *groupes_train.shape[2:])
    centres_train = centres_train.reshape(-1, *centres_train.shape[2:])
    print(f"frames train (aplati): {groupes_train.shape}")

    entrainer_pointmae_jepa(groupes_train, centres_train, appareil=appareil,
                             suffixe="pointmae_jepa")
    print("ENTRAINEMENT_POINTMAE_JEPA_OK")
