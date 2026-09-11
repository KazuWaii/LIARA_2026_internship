"""Autre "add-in" JEPA demande par l'utilisateur (broader comparison) :
au lieu de re-adapter l'ENCODEUR spatial Point-MAE (modele_pointmae_jepa.py,
echec net, cf. Tracking), on garde l'encodeur ShapeNet FIGE tel quel (features
deja extraites, features_pointmae.npz) et on ajoute un JEPA TEMPOREL par
dessus : un petit Transformer contexte/cible/predicteur (reutilise
Encodeur/Predicteur du reste du projet, agnostiques a la dimension) apprend
a PREDIRE l'embedding de frames masquees a partir des frames visibles d'une
fenetre (8 frames/fenetre) -- teste si un agregateur temporel APPRIS
(auto-supervise, sans labels) bat le simple pooling mean+max fige utilise
jusqu'ici pour combiner les 8 features par-frame en un vecteur par fenetre.

8 positions temporelles seulement -> embedding de position appris (table,
pas besoin de sin-cos). Meme recette anti-effondrement que le reste du
projet : perte_uniformite_poolee des le depart sur la representation
poolee du CONTEXTE."""

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch import nn

from dataset_jepa import decouper_train_val, decouper_train_val_environnement
from protocole import masque_lignes
from encodeur import Encodeur
from predicteur import Predicteur
from perte import perte_uniformite_poolee

DIM_ENTREE = 768
NB_FRAMES = 8


class ModeleTemporalJepa(nn.Module):
    def __init__(self, dim=256, dim_predicteur=128, profondeur_encodeur=3, profondeur_predicteur=2,
                 nb_tetes=4, ratio_masque=0.5):
        super().__init__()
        self.projection = nn.Linear(DIM_ENTREE, dim)
        self.pos_embed = nn.Embedding(NB_FRAMES, dim)
        self.encodeur_contexte = Encodeur(dim, profondeur_encodeur, nb_tetes)
        import copy
        self.encodeur_cible = copy.deepcopy(self.encodeur_contexte)
        for p in self.encodeur_cible.parameters():
            p.requires_grad = False
        self.predicteur = Predicteur(dim, dim_predicteur, profondeur_predicteur, nb_tetes)
        self.token_masque = nn.Parameter(torch.zeros(dim))
        nn.init.trunc_normal_(self.token_masque, std=0.02)
        self.ratio_masque = ratio_masque
        self.dim = dim

    @torch.no_grad()
    def mettre_a_jour_cible(self, decroissance=0.996):
        for p_ctx, p_cib in zip(self.encodeur_contexte.parameters(), self.encodeur_cible.parameters()):
            p_cib.data.mul_(decroissance).add_(p_ctx.data, alpha=1 - decroissance)

    def _tokens(self, x):
        """x: (B,8,768) -> (B,8,dim) avec position ajoutee."""
        pos = self.pos_embed(torch.arange(NB_FRAMES, device=x.device)).unsqueeze(0)
        return self.projection(x) + pos

    def forward(self, x):
        """x: (B,8,768) features par-frame deja extraites (Point-MAE fige)."""
        B = x.shape[0]
        tokens = self._tokens(x)  # (B,8,dim)
        n_masque = max(1, int(round(NB_FRAMES * self.ratio_masque)))
        perm = torch.rand(B, NB_FRAMES, device=x.device).argsort(dim=1)
        idx_masque, idx_ctx = perm[:, :n_masque], perm[:, n_masque:]

        tokens_ctx = torch.gather(tokens, 1, idx_ctx.unsqueeze(-1).expand(-1, -1, self.dim))
        contexte_encode = self.encodeur_contexte(tokens_ctx)

        with torch.no_grad():
            cible_complete = self.encodeur_cible(tokens)
        cible = torch.gather(cible_complete, 1, idx_masque.unsqueeze(-1).expand(-1, -1, self.dim))

        pos_cible = self.pos_embed(idx_masque)  # (B, n_masque, dim)
        tokens_predicteur = self.token_masque.view(1, 1, -1) + pos_cible
        prediction = self.predicteur(contexte_encode, tokens_predicteur)
        return prediction, cible, contexte_encode

    @torch.no_grad()
    def extraire(self, x):
        """Inference : PAS de masquage, encode la fenetre complete -> (B,8,dim)."""
        tokens = self._tokens(x)
        return self.encodeur_contexte(tokens)


def entrainer(X_train, nb_etapes=1500, taille_lot=128, taux_apprentissage=1e-3,
              poids_uniformite=1.0, graine=0, appareil="cuda", verbeux_tous=200):
    torch.manual_seed(graine)
    modele = ModeleTemporalJepa().to(appareil)
    optimiseur = torch.optim.AdamW(
        [p for n, p in modele.named_parameters() if not n.startswith("encodeur_cible")],
        lr=taux_apprentissage)
    n = X_train.shape[0]
    modele.train()
    for etape in range(nb_etapes):
        idx = np.random.randint(0, n, size=taille_lot)
        x = torch.from_numpy(X_train[idx]).float().to(appareil)
        prediction, cible, contexte_encode = modele(x)
        perte_l1 = (prediction - cible).abs().mean()
        perte_unif = perte_uniformite_poolee(contexte_encode)
        perte = perte_l1 + poids_uniformite * perte_unif
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
        modele.mettre_a_jour_cible()
        if etape % verbeux_tous == 0:
            print(f"  [temporal_jepa graine={graine}] etape {etape:4d}/{nb_etapes}  "
                  f"l1={perte_l1.item():.4f} unif={perte_unif.item():.4f}", flush=True)
    return modele


if __name__ == "__main__":
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load("mmfi-src/features_pointmae.npz")
    X, y, envs, sujets = data["X"], data["y"], data["envs"], data["sujets"]
    seq_train, seq_val = decouper_train_val()
    m_train, m_val = masque_lignes(seq_train, envs, sujets), masque_lignes(seq_val, envs, sujets)
    X_train_np, X_val_np = X[m_train], X[m_val]
    y_train, y_val = y[m_train], y[m_val]
    print(f"train: {X_train_np.shape}  val: {X_val_np.shape}")

    modele = entrainer(X_train_np, appareil=appareil)

    modele.eval()
    with torch.no_grad():
        emb_train = modele.extraire(torch.from_numpy(X_train_np).float().to(appareil)).cpu().numpy()
        emb_val = modele.extraire(torch.from_numpy(X_val_np).float().to(appareil)).cpu().numpy()

    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    def pool(e):
        return np.concatenate([e.mean(axis=1), e.max(axis=1)], axis=1)

    Xtr, Xva = pool(emb_train), pool(emb_val)
    scaler = StandardScaler().fit(Xtr)
    svm = SVC(kernel="rbf", C=3.0, gamma="scale").fit(scaler.transform(Xtr), y_train)
    preds = svm.predict(scaler.transform(Xva))
    acc = (preds == y_val).mean()
    f1 = f1_score(y_val, preds, average="macro")
    print(f"\nRESULTAT temporal_jepa + SVM : accuracy={acc:.3f} f1_macro={f1:.3f}")
    print("reference (mean+max pooling fige + SVM) : 0.949")
    print("ENTRAINEMENT_TEMPORAL_JEPA_OK")
