"""Masquage multi-blocs, deux methodes (fusionnees le 31 aout pour minimiser
le nombre de fichiers -- pas de code reellement partage entre les deux, deux
mecanismes differents regroupes par role).

== Methode grille ==

Reecrite le 3 septembre pour mirroirer fidelement le masquage de V-JEPA
(reference_vjepa/src/masks/multiblock3d.py) plutot que de s'en inspirer
librement -- deux differences avec l'ancienne version (avant le 3 septembre) :

1. Masque different PAR ECHANTILLON du lot (avant : un seul masque tire par
   lot, partage entre les 8 sequences) -- V-JEPA tire une position de bloc
   independante par image/clip, seule la TAILLE de bloc est partagee au sein
   d'un appel (`_sample_block_size` avec seed partagee, `_sample_block_mask`
   sans seed). Plus de diversite de signal par pas d'entrainement.
2. Deux groupes de masques simultanes (courte/longue portee, comme les deux
   entrees `mask:` de vitl16.yaml) plutot qu'un seul -- chaque groupe donne
   un probleme de prediction different (peu de gros blocs vs plusieurs
   petits), traites dans le meme pas en empilant sur la dimension batch
   (comme `apply_masks(..., concat=True)`), cf. modele.py.

Bloc echantillonne dans les dimensions SPATIALES (nb_h, nb_w) puis repete a
l'identique sur toute la profondeur temporelle (nb_t) -- masquage plein-temps
(cf. 4-adaptation_temporelle_ijepa_vjepa.txt), confirme correct contre le
code source reel de V-JEPA le 3 septembre (`temporal_pred_mask_scale=(1.0,1.0)`
dans multiblock3d.py).

Le token "vide" (tokenisation.py) et la perte scindee vide/occupe (perte.py)
ont ete retires en meme temps que cette reecriture -- V-JEPA n'a pas cette
notion (voir tokenisation.py pour le detail).

== Methode points ==

Blocs definis par region geometrique 3D (repere capteur brut, memes bornes
que le FOV grille) plutot que par indices de grille -- le masquage
plein-temps est automatique ici (une boite (x,y,z) n'a pas de dimension
temporelle). Echelles recalibrees le 31 aout pour aligner les proportions
cible/contexte sur celles de la methode grille (~50%/~29% median), pour que
la comparaison entre les deux methodes porte sur la tokenisation, pas sur un
ecart de reglage. Pas de token vide : FPS ne selectionne que des points reels.
"""

import torch
from torch import nn

from tokenisation import positional_embedding_points

# ----------------------------------------------------------------------
# Methode grille
# ----------------------------------------------------------------------

def _echantillonner_taille_bloc(nb_h, nb_w, echelle_scale, ratio_aspect):
    """(t_h, t_w) taille de bloc spatial -- mirroir de _sample_block_size
    (multiblock3d.py), partie spatiale seulement (la profondeur est toujours
    nb_t entier, masquage plein-temps)."""
    echelle = torch.empty(1).uniform_(*echelle_scale).item()
    aire = max(1.0, nb_h * nb_w * echelle)
    ratio = torch.empty(1).uniform_(*ratio_aspect).item()
    h = min(max(int(round((aire * ratio) ** 0.5)), 1), nb_h)
    w = min(max(int(round((aire / ratio) ** 0.5)), 1), nb_w)
    return h, w


def _echantillonner_masque_spatial(nb_h, nb_w, nb_blocs, echelle_scale, ratio_aspect):
    """Union de `nb_blocs` rectangles a positions INDEPENDANTES -> masque
    booleen (nb_h, nb_w), True = a l'interieur d'au moins un bloc.
    Mirroir de _sample_block_mask + la boucle npred de _MaskGenerator.__call__
    (multiblock3d.py) -- une position aleatoire differente par bloc, appele
    une fois par ECHANTILLON du lot (pas partage comme avant le 3 septembre)."""
    masque = torch.zeros(nb_h, nb_w, dtype=torch.bool)
    for _ in range(nb_blocs):
        h, w = _echantillonner_taille_bloc(nb_h, nb_w, echelle_scale, ratio_aspect)
        h0 = torch.randint(0, nb_h - h + 1, (1,)).item()
        w0 = torch.randint(0, nb_w - w + 1, (1,)).item()
        masque[h0:h0 + h, w0:w0 + w] = True
    return masque


def _echantillonner_groupe(taille_lot, nb_t, nb_h, nb_w, preset):
    """Un groupe de masques (ex. courte-portee), un tirage INDEPENDANT par
    echantillon du lot -- mirroir de _MaskGenerator.__call__(batch_size).
    Cible = union des blocs (nb_blocs), contexte = complement, retire tant
    que le contexte est vide. Les longueurs varient par echantillon (blocs a
    positions differentes) -> tronquees a la longueur minimale du lot avant
    de pouvoir empiler en tenseur rectangulaire (meme technique que
    min_keep_enc/min_keep_pred dans multiblock3d.py).

    Retourne (masque_enc, masque_pred), chacun (taille_lot, K) LongTensor
    d'indices dans la grille aplatie (nb_t*nb_h*nb_w)."""
    n = nb_t * nb_h * nb_w
    liste_pred, liste_enc = [], []
    min_pred = min_enc = n
    for _ in range(taille_lot):
        contexte_vide = True
        while contexte_vide:
            masque_spatial = _echantillonner_masque_spatial(
                nb_h, nb_w, preset["nb_blocs"], preset["echelle"], preset["ratio_aspect"])
            masque = masque_spatial.unsqueeze(0).expand(nb_t, -1, -1).reshape(-1)  # plein-temps
            idx_pred = masque.nonzero(as_tuple=True)[0]
            idx_enc = (~masque).nonzero(as_tuple=True)[0]
            contexte_vide = idx_enc.numel() == 0
        min_pred = min(min_pred, idx_pred.numel())
        min_enc = min(min_enc, idx_enc.numel())
        liste_pred.append(idx_pred)
        liste_enc.append(idx_enc)

    masque_pred = torch.stack([p[:min_pred] for p in liste_pred])
    masque_enc = torch.stack([e[:min_enc] for e in liste_enc])
    return masque_enc, masque_pred


def rassembler(x, indices):
    """x: (B,N,dim). indices: (B,K) LongTensor. Retourne (B,K,dim) --
    primitive de gather utilisee pour extraire le contexte/la cible d'un
    groupe de masque (cf. modele.py). Note (3 septembre) : contrairement a
    apply_masks (reference_vjepa/src/masks/utils.py), pas de variante qui
    empile plusieurs groupes sur la dimension batch -- nos deux groupes
    (courte/longue portee) produisent des tailles de contexte DIFFERENTES
    par construction (plus de petits blocs vs moins de gros blocs), donc pas
    concatenables sur dim=0 (torch.cat exige des tailles identiques sur les
    autres dimensions) ; chaque groupe est traite par un appel separe a
    l'encodeur/predicteur dans modele.py plutot qu'un seul appel empile.

    `indices` est toujours genere sur CPU (torch.randint/randperm dans
    _echantillonner_groupe, pas de raison de payer un aller-retour GPU pour
    un tirage aleatoire sur une grille de quelques centaines de cellules) --
    deplace ici vers le device de `x` (GPU en entrainement reel)."""
    indices = indices.to(x.device)
    return torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


# Presets par defaut -- deux groupes simultanes (courte/longue portee, comme
# les deux entrees `mask:` de vitl16.yaml), fractions de grille plutot que
# comptes absolus pour rester valables sur MM-Fi (4x6) et ONCE (4x18) sans
# changement (design dataset-agnostique, cf. plan du 3 septembre). Valeurs de
# depart, pas encore calibrees empiriquement sur notre grille (bien plus
# petite que celle de V-JEPA) -- a ajuster via un diagnostic court si le
# contexte/la cible s'averent trop desequilibres.
PRESET_GRILLE_COURTE_PORTEE = dict(nb_blocs=4, echelle=(0.1, 0.2), ratio_aspect=(0.75, 1.5))
PRESET_GRILLE_LONGUE_PORTEE = dict(nb_blocs=1, echelle=(0.5, 0.75), ratio_aspect=(0.75, 1.5))


class MasquageSpatioTemporel(nn.Module):
    def __init__(self, dim, nb_t, nb_h, nb_w, presets=None):
        super().__init__()
        self.dim = dim
        self.nb_t, self.nb_h, self.nb_w = nb_t, nb_h, nb_w
        self.presets = presets or (PRESET_GRILLE_COURTE_PORTEE, PRESET_GRILLE_LONGUE_PORTEE)

        # Un token masque APPRIS par groupe (comme num_mask_tokens=2 de
        # VisionTransformerPredictor, cf. reference_vjepa/src/models/predictor.py)
        # -- le predicteur lui-meme n'a pas besoin de savoir qu'il y a
        # plusieurs groupes, il recoit juste un lot deja concatene ou chaque
        # groupe porte son propre token.
        self.tokens_masque = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in self.presets])
        for t in self.tokens_masque:
            nn.init.trunc_normal_(t, std=0.02)

    def echantillonner(self, taille_lot):
        """Retourne (masques_enc, masques_pred) : chacun une liste de
        len(presets) tenseurs (taille_lot, K) LongTensor (K peut differer
        d'un groupe a l'autre)."""
        masques_enc, masques_pred = [], []
        for preset in self.presets:
            enc, pred = _echantillonner_groupe(taille_lot, self.nb_t, self.nb_h, self.nb_w, preset)
            masques_enc.append(enc)
            masques_pred.append(pred)
        return masques_enc, masques_pred

    def tokens_predicteur_groupe(self, i, masque_pred, pos_embed):
        """Token masque du groupe i (appris, cf. self.tokens_masque) +
        position gatheree aux indices cibles de CE groupe. masque_pred:
        (B,K_i) -- un seul groupe (les groupes ont des tailles de cible
        differentes par construction, traites separement, cf. modele.py).
        pos_embed: (N,dim), partage (tokenizer.pos_embed).

        Retourne (B, K_i, dim)."""
        B = masque_pred.shape[0]
        pe = pos_embed.view(1, -1, self.dim).expand(B, -1, -1)
        pe_cible = rassembler(pe, masque_pred)
        return self.tokens_masque[i].view(1, 1, -1) + pe_cible


# ----------------------------------------------------------------------
# Methode points
# ----------------------------------------------------------------------

# Bornes spatiales (repere capteur brut), memes valeurs que le FOV calibre
# empiriquement dans tokenisation.py -- pas d'ancre par-echantillon, une
# seule reference fixe pour tout le dataset.
BORNES_POINTS = {"x": (2.0, 4.0), "y": (-2.5, 2.5), "z": (-1.2, 0.9)}

# Echelles recalibrees (31 aout) pour aligner les proportions cible/contexte
# sur celles de la methode grille (~50%/~29% median) -- les valeurs reprises
# telles quelles des presets grille (0.4-0.6/0.5-0.85) donnaient 59%/19%, un
# ecart qui aurait confondu la comparaison entre methodes.
PRESET_POINTS_COURTE_PORTEE = dict(nb_blocs=3, echelle_min=0.08, echelle_max=0.15)
PRESET_POINTS_LONGUE_PORTEE = dict(nb_blocs=1, echelle_min=0.3, echelle_max=0.45)
PRESET_POINTS_CONTEXTE = dict(nb_blocs=1, echelle_min=0.5, echelle_max=0.85)


def echantillonner_bloc_3d(echelle, centres_reels):
    """Boite (x,y,z) couvrant ~`echelle` du volume de BORNES_POINTS, isotrope,
    centree sur un centre de patch REEL tire au hasard (pas une position
    uniforme dans BORNES_POINTS).

    BORNES_POINTS est presque vide : les points ne couvrent qu'une fine
    surface corporelle dans un volume englobant bien plus grand (verifie
    empiriquement : une boite uniforme rate le nuage de points dans ~15-25%
    des tirages, cf. discussion du 31 aout). Ancrer sur un point reel garantit
    qu'au moins ce point est couvert."""
    idx = torch.randint(0, centres_reels.shape[0], (1,)).item()
    ancre = centres_reels[idx]

    fraction_axe = echelle ** (1 / 3)
    bloc = {}
    for i, (axe, (mini, maxi)) in enumerate(BORNES_POINTS.items()):
        etendue = min((maxi - mini) * fraction_axe, maxi - mini)
        debut = torch.clamp(ancre[i] - etendue / 2, mini, maxi - etendue).item()
        bloc[axe] = (debut, debut + etendue)
    return bloc


def dans_bloc(centres, bloc):
    """centres: (...,3) -> bool (...)."""
    x, y, z = centres[..., 0], centres[..., 1], centres[..., 2]
    return (
        (x >= bloc["x"][0]) & (x <= bloc["x"][1])
        & (y >= bloc["y"][0]) & (y <= bloc["y"][1])
        & (z >= bloc["z"][0]) & (z <= bloc["z"][1])
    )


def masque_multiblocs_3d(centres, preset):
    centres_plats = centres.reshape(-1, 3)
    masque = torch.zeros(centres.shape[:-1], dtype=torch.bool, device=centres.device)
    for _ in range(preset["nb_blocs"]):
        echelle = torch.empty(1).uniform_(preset["echelle_min"], preset["echelle_max"]).item()
        masque |= dans_bloc(centres, echantillonner_bloc_3d(echelle, centres_plats))
    return masque


class MasquagePoints(nn.Module):
    def __init__(self, dim, preset_cible=None):
        super().__init__()
        self.dim = dim
        self.preset_cible = preset_cible or PRESET_POINTS_LONGUE_PORTEE

        self.token_masque = nn.Parameter(torch.zeros(dim))
        nn.init.trunc_normal_(self.token_masque, std=0.02)

    def echantillonner_masques(self, centres):
        """centres: (T,K,3) -- une fenetre de reference, masque partage sur
        le batch (meme simplification que la methode grille). Retourne
        (cible, contexte) bool (T*K,), disjoints, contexte garanti non vide."""
        cible = masque_multiblocs_3d(centres, self.preset_cible)

        contexte = ~cible
        for _ in range(10):
            essai = masque_multiblocs_3d(centres, PRESET_POINTS_CONTEXTE) & ~cible
            if essai.any():
                contexte = essai
                break

        return cible.reshape(-1), contexte.reshape(-1)

    def forward(self, patches, centres):
        """patches: (B, T*K, dim) sortie de TokenizerPoints. centres: (B,T,K,3).

        Retourne un dict : contexte, tokens_predicteur, idx_cible, idx_contexte
        -- meme structure que MasquageSpatioTemporel, sans occupation_cible
        (pas de notion de vide ici)."""
        B, T, K, _ = centres.shape
        centres_ref = centres[0]  # fenetre de reference pour l'echantillonnage du masque

        cible_bool, contexte_bool = self.echantillonner_masques(centres_ref)
        idx_cible = cible_bool.nonzero(as_tuple=True)[0]
        idx_contexte = contexte_bool.nonzero(as_tuple=True)[0]

        contexte_encode = patches[:, idx_contexte]

        t_idx = torch.arange(T, device=centres.device, dtype=torch.float32).view(T, 1).expand(T, K).reshape(-1)
        pe_cible = positional_embedding_points(centres_ref.reshape(-1, 3)[idx_cible], t_idx[idx_cible], self.dim)
        tokens_predicteur = self.token_masque.view(1, -1) + pe_cible

        return {
            "contexte": contexte_encode,
            "tokens_predicteur": tokens_predicteur,
            "idx_cible": idx_cible,
            "idx_contexte": idx_contexte,
        }
