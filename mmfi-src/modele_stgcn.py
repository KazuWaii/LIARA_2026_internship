"""ST-GCN (Yan, Xiong & Lin, AAAI 2018) simplifie -- reconnaissance d'action
a partir du squelette 3D VERITE-TERRAIN de MM-Fi utilise comme entree
PRINCIPALE (pas comme signal auxiliaire de regression, contrairement a
modele_squelette.py). Piste "plafond squelette" (recherche multi-agents,
5 septembre, cf. Tracking) : le squelette verite-terrain n'a jamais servi
que de cible de regression jusqu'ici ; un GNN squelette-standard entraine
DESSUS directement quantifie ce qui est atteignable si on avait un squelette
parfait (pas estime par pose-estimation depuis le nuage brut, comme le fait
le pipeline original MM-Fi AGCN/CTRGCN, qui plafonne a 54,44%/35,97% sur la
modalite LiDAR a cause des erreurs d'estimation de pose qui se propagent).

PAS un resultat LiDAR-only : le squelette verite-terrain est dans un repere
DIFFERENT du nuage LiDAR brut, sans calibration inter-capteurs disponible
(cf. README, section "decouverte annexe") -- valeur de reference/plafond,
utile pour juger si une distillation squelette->LiDAR vaudrait la peine
(si le plafond est tres au-dessus de 0,659, l'ecart motive d'investir dans
un mecanisme de "bridging" ; si le plafond est proche, le squelette
n'apporte pas grand-chose de plus que ce que le LiDAR seul peut deja capter).

SIMPLIFICATION DOCUMENTEE par rapport a l'article original (meme esprit que
laplacien_spectral.py, qui documente ses propres simplifications) : ST-GCN
original utilise un partitionnement spatial en 3 sous-ensembles (racine,
centripete, centrifuge, relatifs a un joint "centre de gravite" choisi) avec
un poids appris PAR sous-ensemble. Ici : UNE SEULE matrice d'adjacence
normalisee (GCN standard, Kipf & Welling) avec une pondération d'arête
apprise (element-wise sur A, mecanisme "edge importance weighting" de
l'article original, conserve car peu couteux) -- pas de partitionnement a
3 sous-ensembles. Choix justifie par la taille de notre graphe (17 joints,
bien plus petit que les squelettes 25-joints de NTU-RGBD pour lesquels
ST-GCN a ete concu) et le temps disponible."""

import torch
from torch import nn

NB_JOINTS = 17
# Aretes du squelette COCO-17 (format ground_truth.npy de MM-Fi) :
# 0 nez, 1/2 yeux, 3/4 oreilles, 5/6 epaules, 7/8 coudes, 9/10 poignets,
# 11/12 hanches, 13/14 genoux, 15/16 chevilles.
ARETES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _adjacence_normalisee():
    """Matrice d'adjacence COCO-17 (aretes du squelette + boucles propres),
    normalisation symetrique D^-1/2 A D^-1/2 (GCN standard)."""
    A = torch.eye(NB_JOINTS)
    for i, j in ARETES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    D_inv_sqrt = A.sum(dim=1).pow(-0.5)
    return D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)


class BlocSTGCN(nn.Module):
    """Un bloc = convolution spatiale sur le graphe squelette (agregation
    des voisins ponderee par l'adjacence + melange de canaux) puis
    convolution temporelle 1D (par joint, le long du temps) + residuel --
    structure standard ST-GCN."""

    def __init__(self, dim_in, dim_out, noyau_temporel=9, stride_temporel=1):
        super().__init__()
        self.register_buffer("A", _adjacence_normalisee())
        self.poids_aretes = nn.Parameter(torch.ones(NB_JOINTS, NB_JOINTS))  # edge importance weighting (ST-GCN)
        self.spatial = nn.Linear(dim_in, dim_out)
        self.bn_spatial = nn.BatchNorm2d(dim_out)
        pad = (noyau_temporel - 1) // 2
        self.temporel = nn.Conv2d(dim_out, dim_out, kernel_size=(noyau_temporel, 1),
                                   stride=(stride_temporel, 1), padding=(pad, 0))
        self.bn_temporel = nn.BatchNorm2d(dim_out)
        self.activation = nn.GELU()

        if dim_in != dim_out or stride_temporel != 1:
            self.residuel = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=(stride_temporel, 1)),
                nn.BatchNorm2d(dim_out),
            )
        else:
            self.residuel = nn.Identity()

    def forward(self, x):
        """x: (B, C_in, T, V) -> (B, C_out, T', V)."""
        res = self.residuel(x)
        A_ponderee = self.A * self.poids_aretes
        z = torch.einsum("vw,bcTw->bcTv", A_ponderee, x)  # agregation des voisins, (B, C_in, T, V)
        z = z.permute(0, 2, 3, 1)                          # (B, T, V, C_in)
        z = self.spatial(z).permute(0, 3, 1, 2)            # (B, C_out, T, V)
        z = self.activation(self.bn_spatial(z))
        z = self.bn_temporel(self.temporel(z))
        return self.activation(z + res)


class ModeleSTGCN(nn.Module):
    def __init__(self, nb_classes=27, dims=(64, 128, 256)):
        super().__init__()
        self.bn_entree = nn.BatchNorm1d(3 * NB_JOINTS)
        blocs = []
        dim_in = 3
        for i, dim_out in enumerate(dims):
            blocs.append(BlocSTGCN(dim_in, dim_out, stride_temporel=2 if i > 0 else 1))
            dim_in = dim_out
        self.blocs = nn.ModuleList(blocs)
        self.tete = nn.Linear(2 * dims[-1], nb_classes)

    def forward(self, squelette, retourner_features=False):
        """squelette: (B, T, 17, 3), deja centre par frame -> logits (B, nb_classes),
        ou (logits, pool) si retourner_features -- `pool` (B, 2*dims[-1]) est la
        representation poolee juste avant la tete lineaire, utilisee comme cible
        de distillation de features (entrainer_distillation.py, enseignant fige)."""
        B, T, V, C = squelette.shape
        x = squelette.permute(0, 3, 2, 1).contiguous().view(B, C * V, T)
        x = self.bn_entree(x).view(B, C, V, T).permute(0, 1, 3, 2).contiguous()  # (B, C, T, V)
        for bloc in self.blocs:
            x = bloc(x)
        pool = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=-1)
        logits = self.tete(pool)
        return (logits, pool) if retourner_features else logits
