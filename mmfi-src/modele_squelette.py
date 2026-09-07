"""Modele multi-tache supervise (4 septembre matin) : encodeur partage
(tokenizer + Transformer, memes briques que ModeleJEPA) suivi de DEUX tetes
entrainees conjointement -- classification (27 actions, pooling mean+max
comme evaluation.py) et regression du squelette 3D verite-terrain (signal
auxiliaire dense, cf. squelette_dataset.py). PAS un modele JEPA : pas de
masquage, pas d'encodeur cible EMA, pas de predicteur -- entrainement
supervise direct de bout en bout (from scratch ou initialise depuis un
checkpoint JEPA pre-entraine via `charger_poids_jepa`)."""

import torch
from torch import nn

from tokenisation import TokenizerSpatioTemporel
from encodeur import Encodeur

NB_JOINTS = 17


class TeteSquelette(nn.Module):
    """Pooling spatial (mean+max sur les nb_h*nb_w tokens de chaque pas de
    temps) puis MLP par pas de temps -> (B, nb_t, 17, 3). Independant du
    pooling temporel-complet utilise par la tete de classification : la pose
    est intrinsequement une quantite PAR PAS DE TEMPS, contrairement a la
    classe d'action qui est une propriete de toute la fenetre."""

    def __init__(self, dim, nb_h, nb_w):
        super().__init__()
        self.nb_h, self.nb_w = nb_h, nb_w
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(),
            nn.Linear(dim, NB_JOINTS * 3),
        )

    def forward(self, z):
        """z: (B, nb_t*nb_h*nb_w, dim) -> (B, nb_t, 17, 3)."""
        B, N, dim = z.shape
        nb_t = N // (self.nb_h * self.nb_w)
        z = z.view(B, nb_t, self.nb_h * self.nb_w, dim)
        pool = torch.cat([z.mean(dim=2), z.max(dim=2).values], dim=-1)  # (B, nb_t, 2*dim)
        return self.mlp(pool).view(B, nb_t, NB_JOINTS, 3)


class ModeleSquelette(nn.Module):
    def __init__(self, dim=192, profondeur_encodeur=4, nb_tetes=6, nb_classes=27,
                 hauteur=64, largeur=96, nb_frames=16, noyau_t=2, noyau_spatial=16):
        super().__init__()
        self.tokenizer = TokenizerSpatioTemporel(
            dim=dim, hauteur=hauteur, largeur=largeur, nb_frames=nb_frames,
            noyau_t=noyau_t, noyau_spatial=noyau_spatial,
        )
        self.encodeur = Encodeur(dim, profondeur_encodeur, nb_tetes)
        self.tete_classification = nn.Linear(2 * dim, nb_classes)
        self.tete_squelette = TeteSquelette(dim, self.tokenizer.nb_h, self.tokenizer.nb_w)

    def charger_poids_jepa(self, chemin_checkpoint, appareil):
        """Initialise tokenizer+encodeur depuis un checkpoint ModeleJEPA
        (memes noms de sous-modules : tokenizer, encodeur_contexte -> encodeur
        ici). Les tetes restent initialisees aleatoirement (pas de
        correspondance dans le checkpoint JEPA)."""
        etat = torch.load(chemin_checkpoint, map_location=appareil)
        etat_tokenizer = {k[len("tokenizer."):]: v for k, v in etat.items() if k.startswith("tokenizer.")}
        etat_encodeur = {k[len("encodeur_contexte."):]: v for k, v in etat.items() if k.startswith("encodeur_contexte.")}
        self.tokenizer.load_state_dict(etat_tokenizer)
        self.encodeur.load_state_dict(etat_encodeur)

    def forward(self, tenseur):
        """tenseur: (B,T,4,H,W) -> (logits_classe (B,nb_classes), squelette_pred (B,nb_t,17,3))."""
        patches = self.tokenizer(tenseur)
        z = self.encodeur(patches)
        pool_complet = torch.cat([z.mean(dim=1), z.max(dim=1).values], dim=-1)
        logits = self.tete_classification(pool_complet)
        squelette_pred = self.tete_squelette(z)
        return logits, squelette_pred


class ModeleClassificationLidar(nn.Module):
    """Classification LiDAR-seule (tokenizer+Transformer, MEME architecture/
    hyperparametres que le tronc partage de ModeleSquelette/ModeleJEPA -- pas
    de tete de regression squelette). Sert d'ETUDIANT pour la distillation
    squelette->LiDAR (entrainer_distillation.py, piste "hallucination de
    modalite", cf. modele_stgcn.py) : c'est l'architecture "from-scratch"
    directement comparable a la reference deja mesuree (0,509+-0,044, meme
    tronc via evaluation.construire_modele) mais assemblee ici en un modele
    autonome (pas besoin d'instancier tout ModeleJEPA pour n'utiliser que
    tokenizer+encodeur_contexte)."""

    def __init__(self, dim=192, profondeur_encodeur=4, nb_tetes=6, nb_classes=27,
                 hauteur=64, largeur=96, nb_frames=16, noyau_t=2, noyau_spatial=16):
        super().__init__()
        self.tokenizer = TokenizerSpatioTemporel(
            dim=dim, hauteur=hauteur, largeur=largeur, nb_frames=nb_frames,
            noyau_t=noyau_t, noyau_spatial=noyau_spatial,
        )
        self.encodeur = Encodeur(dim, profondeur_encodeur, nb_tetes)
        self.tete_classification = nn.Linear(2 * dim, nb_classes)

    def forward(self, tenseur, retourner_features=False):
        """tenseur: (B,T,4,H,W) -> logits (B,nb_classes), ou (logits, pool)
        (B,2*dim) si retourner_features -- `pool` sert de cible de
        distillation de features cote etudiant (projetee vers la dimension
        de l'enseignant par un adaptateur separe, cf. entrainer_distillation.py)."""
        patches = self.tokenizer(tenseur)
        z = self.encodeur(patches)
        pool_complet = torch.cat([z.mean(dim=1), z.max(dim=1).values], dim=-1)
        logits = self.tete_classification(pool_complet)
        return (logits, pool_complet) if retourner_features else logits
