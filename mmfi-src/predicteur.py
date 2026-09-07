"""Predicteur : transformer plus etroit que l'encodeur (meme principe que
V-JEPA et Point-JEPA -- asymetrie contexte/predicteur, l'un des mecanismes
anti-effondrement standard, cf. note 3/4). Prend les tokens de contexte deja
encodes + les tokens masque+position des positions cibles (construits dans
masquage.py), predit un embedding par position cible."""

import torch
from torch import nn
from encodeur import Encodeur


class Predicteur(nn.Module):
    def __init__(self, dim, dim_predicteur, profondeur, nb_tetes, ratio_mlp=4.0, init_std=0.02):
        super().__init__()
        self.projection_entree = nn.Linear(dim, dim_predicteur)
        self.encodeur = Encodeur(dim_predicteur, profondeur, nb_tetes, ratio_mlp)
        self.projection_sortie = nn.Linear(dim_predicteur, dim)

        # Encodeur interne s'initialise deja lui-meme (cf. encodeur.py) --
        # ici, seules les deux projections propres au predicteur (mirroir de
        # predictor_embed/predictor_proj, V-JEPA _init_weights).
        nn.init.trunc_normal_(self.projection_entree.weight, std=init_std)
        nn.init.constant_(self.projection_entree.bias, 0)
        nn.init.trunc_normal_(self.projection_sortie.weight, std=init_std)
        nn.init.constant_(self.projection_sortie.bias, 0)

    def forward(self, contexte_encode, tokens_predicteur):
        """contexte_encode: (B, |ctx|, dim) -- sortie de l'encodeur de contexte.
        tokens_predicteur: (|cible|, dim) partages sur le batch (methode
        points, inchangee), OU (B, |cible|, dim) deja par-echantillon
        (methode grille depuis le 3 septembre, cf. masquage.py)."""
        B = contexte_encode.shape[0]
        if tokens_predicteur.dim() == 2:
            tokens_predicteur = tokens_predicteur.unsqueeze(0).expand(B, -1, -1)

        x = self.projection_entree(torch.cat([contexte_encode, tokens_predicteur], dim=1))
        x = self.encodeur(x)

        nb_cible = tokens_predicteur.shape[1]
        sortie_cible = x[:, -nb_cible:]
        return self.projection_sortie(sortie_cible)
