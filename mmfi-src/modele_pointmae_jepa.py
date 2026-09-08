"""JEPA applique a l'encodeur Point-MAE (piste demandee explicitement :
"adding jepa architecture to current performing method" -- le "current
performing method" est le transfer learning Point-MAE/ShapeNet, cf.
extraire_pointmae.py, meilleur resultat du projet, 0,949).

PAS un JEPA from scratch (deja teste, echec, cf. README section JEPA) :
ici l'encodeur de contexte PART des poids ShapeNet-pretrained (comme le
linear probing/fine-tuning), et le pre-entrainement JEPA continue de
l'ADAPTER a MM-Fi de facon AUTO-SUPERVISEE (masquage + prediction en espace
d'embedding, pas de labels) -- ni le probing gele pur (features fixes) ni
le fine-tuning supervise complet (deja teste, echoue, 0,894, sur-apprend
sur seulement 864 sequences) : une troisieme option, adaptation de domaine
sans les labels de classification.

Masquage au niveau des GROUPES, PAR FRAME (comme le MAE original de
Point-MAE, pas de masquage temporel ici) : ~50% des 64 groupes d'une frame
servent de CONTEXTE, le reste de CIBLE. Reutilise integralement
Predicteur (predicteur.py) et perte_uniformite_poolee (perte.py) --
lecon retenue du JEPA maison (README, section "Reconstruction V-JEPA") :
EMA+stop-gradient+masquage seuls ne suffisent PAS a empecher l'effondrement
sur ce type de donnees, meme avec un encodeur pre-entraine -- la
regularisation d'uniformite poolee est activee des le depart plutot que
redecouverte par tatonnement."""

import copy

import torch
from torch import nn

from modele_pointmae import EncodeurPointMAE, DIM
from predicteur import Predicteur


class ModelePointMAEJepa(nn.Module):
    def __init__(self, dim_predicteur=192, profondeur_predicteur=4, nb_tetes=6, ratio_masque=0.5):
        super().__init__()
        self.encodeur_contexte = EncodeurPointMAE()
        self.encodeur_cible = copy.deepcopy(self.encodeur_contexte)
        for p in self.encodeur_cible.parameters():
            p.requires_grad = False

        self.predicteur = Predicteur(DIM, dim_predicteur, profondeur_predicteur, nb_tetes)
        self.token_masque = nn.Parameter(torch.zeros(DIM))
        nn.init.trunc_normal_(self.token_masque, std=0.02)
        self.ratio_masque = ratio_masque

    @torch.no_grad()
    def mettre_a_jour_cible(self, decroissance=0.996):
        for p_ctx, p_cib in zip(self.encodeur_contexte.parameters(), self.encodeur_cible.parameters()):
            p_cib.data.mul_(decroissance).add_(p_ctx.data, alpha=1 - decroissance)

    def _masque_bloc(self, centres, n_masque):
        """Masquage par BLOC spatial (comme Point-MAE/V-JEPA original --
        PAS un sous-ensemble IID aleatoire de groupes, cf. note module) :
        un centre "graine" tire au hasard, puis les n_masque groupes les
        PLUS PROCHES de cette graine (par distance euclidienne des centres)
        forment le bloc masque -- une region spatialement CONTIGUE, pas
        eparpillee dans tout le nuage."""
        B, G, _ = centres.shape
        graine = torch.randint(0, G, (B,), device=centres.device)
        centre_graine = centres[torch.arange(B), graine].unsqueeze(1)  # (B,1,3)
        d = ((centres - centre_graine) ** 2).sum(-1)  # (B,G)
        idx_tries = d.argsort(dim=1)
        return idx_tries[:, :n_masque], idx_tries[:, n_masque:]

    def forward(self, groupes, centres, masquage_bloc=False):
        """groupes: (B,G,K,3), centres: (B,G,3) -- UNE frame a la fois (le
        pretexte opere frame par frame, comme Point-MAE original, pas de
        masquage temporel). `masquage_bloc` : masque une region spatiale
        CONTIGUE (variante "V-JEPA-style", cf. _masque_bloc) au lieu d'un
        sous-ensemble IID aleatoire de groupes (comportement par defaut,
        version testee en premier). Retourne (prediction, cible, contexte_encode)."""
        B, G, K, _ = groupes.shape
        n_masque = max(1, int(round(G * self.ratio_masque)))

        if masquage_bloc:
            idx_masque, idx_contexte = self._masque_bloc(centres, n_masque)
        else:
            perm = torch.rand(B, G, device=groupes.device).argsort(dim=1)
            idx_masque, idx_contexte = perm[:, :n_masque], perm[:, n_masque:]

        groupes_ctx = torch.gather(groupes, 1, idx_contexte.view(B, -1, 1, 1).expand(-1, -1, K, 3))
        centres_ctx = torch.gather(centres, 1, idx_contexte.unsqueeze(-1).expand(-1, -1, 3))
        centres_cible = torch.gather(centres, 1, idx_masque.unsqueeze(-1).expand(-1, -1, 3))

        contexte_encode = self.encodeur_contexte(groupes_ctx, centres_ctx)

        with torch.no_grad():
            cible_complete = self.encodeur_cible(groupes, centres)
        cible = torch.gather(cible_complete, 1, idx_masque.unsqueeze(-1).expand(-1, -1, DIM))

        pos_embed_cible = self.encodeur_contexte.pos_embed(centres_cible)
        tokens_predicteur = self.token_masque.view(1, 1, -1) + pos_embed_cible
        prediction = self.predicteur(contexte_encode, tokens_predicteur)

        return prediction, cible, contexte_encode
