"""Encodeur Point-MAE (Pang et al., ECCV 2022, arXiv:2203.06604) --
reimplementation PURE PYTORCH pour charger le checkpoint officiel
pre-entraine sur ShapeNet (github.com/Pang-Yatian/Point-MAE, fichier
`pretrain.pth` telecharge directement -- 332 Mo, poids seuls, AUCUN code de
leur depot execute) SANS dependre de leurs extensions CUDA compilees
(pointnet2_ops/KNN_CUDA/Chamfer distance) -- indisponibles sur cette
machine (pas de nvcc, pas de MSVC/cl.exe, verifie -- installer une chaine
CUDA Toolkit + Visual Studio Build Tools complete pour ce seul test aurait
ete un changement systeme lourd, non fait sans consultation explicite,
cf. session 5 septembre).

Piste "transfer learning source != cible" (recherche multi-agents du
5 septembre, PointGST/Point-JEPA-officiel/Point-MAE tous suitability=5) :
notre propre tentative JEPA a echoue en pre-entrainant ET fine-tunant sur
LE MEME petit corpus MM-Fi (864 sequences) -- la litterature suggere que le
transfert marche surtout quand la source de pre-entrainement est plus
grande/differente de la cible. ShapeNet (~50k objets CAD, categorie
totalement differente : objets rigides complets vs silhouette humaine
eparse) est un vrai changement de domaine, contrairement a notre echec.

ARCHITECTURE (reverse-engineered depuis les CLES et FORMES du state_dict du
checkpoint, PAS depuis leur code source -- verifiee coherente avec la
configuration Point-MAE publique connue : trans_dim=384, depth=12,
num_heads=6, mlp_ratio=4, group_size=32, num_group=64) :

  nuage brut -> FPS (64 centres) + kNN (32 voisins/centre), reutilise
  TEL QUEL de tokenisation.fps/grouper_knn (deja dans ce depot, methode
  points -- meme algorithme, valeurs K_CENTRES/K_VOISINS differentes)
    -> mini-PointNet (conv1d local+global, PAS le MiniPointNet du projet
       -- architecture differente, doit matcher EXACTEMENT les formes du
       checkpoint pour charger les poids) -> un token (384,) par groupe
    -> + positional embedding (MLP sur le centre du groupe, xyz)
    -> 12 blocs Transformer pre-norme standard (ViT/timm-style, qkv sans
       biais -- confirme par l'absence de cle attn.qkv.bias)
    -> LayerNorm finale

Le DECODEUR MAE (reconstruction, jamais necessaire en aval) et le
`mask_token` (masquage, pas de masquage a l'usage) sont IGNORES au
chargement -- seul l'encodeur (prefixe `module.MAE_encoder.` dans le
checkpoint) est charge."""

import math

import torch
from torch import nn

DIM = 384
PROFONDEUR = 12
NB_TETES = 6
MLP_RATIO = 4
NUM_GROUP = 64
GROUP_SIZE = 32


class EncodeurGroupe(nn.Module):
    """Mini-PointNet local+global (Point-BERT/Point-MAE) -- (B,G,K,3), deja
    centre par groupe (coordonnees relatives au centre) -> (B,G,384)."""

    def __init__(self):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Conv1d(128, 256, 1))
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Conv1d(512, DIM, 1))

    def forward(self, groupes):
        bs, g, k, _ = groupes.shape
        x = groupes.reshape(bs * g, k, 3).transpose(2, 1)  # (bs*g, 3, k)
        f = self.first_conv(x)  # (bs*g, 256, k)
        f_global = torch.max(f, dim=2, keepdim=True).values.expand(-1, -1, k)
        f = torch.cat([f_global, f], dim=1)  # (bs*g, 512, k)
        f = self.second_conv(f)  # (bs*g, 384, k)
        f = torch.max(f, dim=2).values  # (bs*g, 384)
        return f.reshape(bs, g, DIM)


class AttentionViT(nn.Module):
    def __init__(self, dim=DIM, nb_tetes=NB_TETES):
        super().__init__()
        self.nb_tetes = nb_tetes
        self.dim_tete = dim // nb_tetes
        self.echelle = self.dim_tete ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nb_tetes, self.dim_tete).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1) * self.echelle).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class BlocViT(nn.Module):
    def __init__(self, dim=DIM, nb_tetes=NB_TETES, mlp_ratio=MLP_RATIO):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionViT(dim, nb_tetes)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class EncodeurPointMAE(nn.Module):
    """Assemblage complet -- prend directement les groupes (B,G,K,3) et
    centres (B,G,3) DEJA calcules (FPS+kNN cote dataset, cf.
    dataset_pointmae.py, comme la methode points existante du projet)."""

    def __init__(self, dim=DIM, profondeur=PROFONDEUR, nb_tetes=NB_TETES):
        super().__init__()
        self.encodeur_groupe = EncodeurGroupe()
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, dim))
        self.blocks = nn.ModuleList([BlocViT(dim, nb_tetes) for _ in range(profondeur)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, groupes, centres):
        """groupes: (B,G,K,3) centres sur leur groupe ; centres: (B,G,3) --
        retourne (B,G,dim), tokens par groupe (pas encore poolés)."""
        tokens = self.encodeur_groupe(groupes) + self.pos_embed(centres)
        for bloc in self.blocks:
            tokens = bloc(tokens)
        return self.norm(tokens)


class ModeleClassificationPointMAE(nn.Module):
    """Etudiant LiDAR-seul pour la piste transfer-learning ShapeNet-pretrained
    (5 septembre) : EncodeurPointMAE applique PAR FRAME (les frames d'une
    fenetre partagent les memes poids, comme un CNN 2D applique image par
    image), pooling mean+max sur les groupes (par frame) PUIS sur les
    frames (temporel) -- meme principe mean+max que le reste du projet
    (evaluation.pooler_mean_max), applique a deux niveaux."""

    def __init__(self, nb_classes=27, geler_encodeur=False):
        super().__init__()
        self.encodeur = EncodeurPointMAE()
        if geler_encodeur:
            for p in self.encodeur.parameters():
                p.requires_grad = False
        self.tete = nn.Linear(4 * DIM, nb_classes)  # 2*dim (mean+max groupes) x 2 (mean+max frames)

    def forward(self, groupes, centres):
        """groupes: (B,T,G,K,3) ; centres: (B,T,G,3) -> logits (B,nb_classes)."""
        B, T, G, K, _ = groupes.shape
        tokens = self.encodeur(groupes.reshape(B * T, G, K, 3), centres.reshape(B * T, G, 3))  # (B*T,G,dim)
        pool_spatial = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)  # (B*T, 2*dim)
        pool_spatial = pool_spatial.reshape(B, T, 2 * DIM)
        pool_temporel = torch.cat([pool_spatial.mean(dim=1), pool_spatial.max(dim=1).values], dim=-1)  # (B,4*dim)
        return self.tete(pool_temporel)


def charger_poids_pretrain(encodeur, chemin_checkpoint, appareil="cpu"):
    """Charge UNIQUEMENT les poids de l'encodeur (prefixe module.MAE_encoder.)
    depuis le checkpoint officiel Point-MAE (pretrain.pth) -- decodeur MAE,
    mask_token, increase_dim ignores (jamais utilises en aval). Renomme les
    cles internes (encoder.* -> encodeur_groupe.*, blocks.blocks.N -> blocks.N)
    pour matcher les noms de modules ci-dessus."""
    ckpt = torch.load(chemin_checkpoint, map_location=appareil, weights_only=False)
    sd_brut = ckpt["base_model"]
    prefixe = "module.MAE_encoder."
    renomme = {}
    for k, v in sd_brut.items():
        if not k.startswith(prefixe):
            continue
        reste = k[len(prefixe):]
        reste = reste.replace("encoder.", "encodeur_groupe.", 1)
        reste = reste.replace("blocks.blocks.", "blocks.", 1)
        reste = reste.replace("mlp.fc1.", "mlp.0.", 1)
        reste = reste.replace("mlp.fc2.", "mlp.2.", 1)
        renomme[reste] = v
    manquantes, inattendues = encodeur.load_state_dict(renomme, strict=True)
    assert not manquantes and not inattendues, (manquantes, inattendues)
    return encodeur
