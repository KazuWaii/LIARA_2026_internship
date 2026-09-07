"""Encodeur Transformer standard (blocs ViT), utilise a la fois pour
l'encodeur de contexte (entraine) et l'encodeur cible (copie EMA, cf.
modele.py) -- meme architecture pour les deux, comme dans I-JEPA/V-JEPA et
Point-JEPA (note 3, note 4).

Init alignee sur V-JEPA le 3 septembre (reference_vjepa/src/models/
vision_transformer.py, _init_weights/_rescale_blocks) : troncature normale
(std=0.02) sur les poids Linear, et surtout la rescale par profondeur des
projections de sortie (attn.out_proj, mlp[-1]) -- param.div_(sqrt(2*layer_id))
-- stabilite standard des transformers profonds residuels (DeiT/BEiT),
absente de notre implementation jusqu'ici."""

import math

from torch import nn


class BlocTransformer(nn.Module):
    def __init__(self, dim, nb_tetes, ratio_mlp=4.0):
        super().__init__()
        self.norme1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, nb_tetes, batch_first=True)
        self.norme2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * ratio_mlp)),
            nn.GELU(),
            nn.Linear(int(dim * ratio_mlp), dim),
        )

    def forward(self, x):
        h = self.norme1(x)
        attn, _ = self.attention(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.mlp(self.norme2(x))
        return x


class Encodeur(nn.Module):
    def __init__(self, dim, profondeur, nb_tetes, ratio_mlp=4.0, init_std=0.02):
        super().__init__()
        self.blocs = nn.ModuleList([BlocTransformer(dim, nb_tetes, ratio_mlp) for _ in range(profondeur)])
        self.norme_finale = nn.LayerNorm(dim)

        self.init_std = init_std
        self.apply(self._init_poids)
        self._rescale_blocs()

    def _init_poids(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocs(self):
        """Divise les projections de sortie (attention, MLP) de chaque bloc
        par sqrt(2*profondeur) -- stabilise l'accumulation residuelle sur des
        transformers profonds (mirroir de _rescale_blocks, V-JEPA)."""
        for i, bloc in enumerate(self.blocs, start=1):
            bloc.attention.out_proj.weight.data.div_(math.sqrt(2.0 * i))
            bloc.mlp[-1].weight.data.div_(math.sqrt(2.0 * i))

    def forward(self, x):
        for bloc in self.blocs:
            x = bloc(x)
        return self.norme_finale(x)
