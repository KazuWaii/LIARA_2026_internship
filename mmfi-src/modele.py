"""Assemble tokenizer + masquage + encodeur de contexte + encodeur cible (EMA,
stop-grad) + predicteur -- boucle JEPA complete, deux methodes (fusionnees le
31 aout pour minimiser le nombre de fichiers). `Encodeur`/`Predicteur` sont
strictement identiques et partages entre les deux -- transformers generiques,
indifferents a la facon dont les tokens ont ete produits."""

import copy
import torch
from torch import nn

from tokenisation import TokenizerSpatioTemporel, TokenizerPoints
from masquage import MasquageSpatioTemporel, MasquagePoints, rassembler
from encodeur import Encodeur
from predicteur import Predicteur


class ModeleJEPA(nn.Module):
    """Methode grille (projection 2D + conv 3D). Reecrite le 3 septembre
    pour mirroirer V-JEPA (init_video_model + forward_context/forward_target
    dans reference_vjepa/app/vjepa/train.py) : deux groupes de masques traites
    en parallele (empiles sur la dimension batch), forward retourne des
    LISTES predictions/cibles (une entree par groupe), pas des tenseurs
    concatenes -- cf. perte.py (perte_jepa) qui attend cette forme pour
    moyenner les groupes a egalite plutot que ponderes par leur taille."""

    def __init__(
        self, dim=192, dim_predicteur=96,
        profondeur_encodeur=4, profondeur_predicteur=2, nb_tetes=6,
        hauteur=64, largeur=96, nb_frames=16, noyau_t=2, noyau_spatial=16, presets=None,
    ):
        super().__init__()
        self.tokenizer = TokenizerSpatioTemporel(
            dim=dim, hauteur=hauteur, largeur=largeur, nb_frames=nb_frames,
            noyau_t=noyau_t, noyau_spatial=noyau_spatial,
        )

        self.masquage = MasquageSpatioTemporel(
            dim=dim, nb_t=self.tokenizer.nb_t, nb_h=self.tokenizer.nb_h, nb_w=self.tokenizer.nb_w,
            presets=presets,
        )

        self.encodeur_contexte = Encodeur(dim, profondeur_encodeur, nb_tetes)

        # Encodeur cible : copie EMA, jamais mise a jour par retropropagation
        # (stop-gradient + moyenne mobile -- recette anti-effondrement standard
        # I-JEPA/V-JEPA/Point-JEPA, cf. note 3/4).
        self.encodeur_cible = copy.deepcopy(self.encodeur_contexte)
        for p in self.encodeur_cible.parameters():
            p.requires_grad = False

        self.predicteur = Predicteur(dim, dim_predicteur, profondeur_predicteur, nb_tetes)

    @torch.no_grad()
    def mettre_a_jour_cible(self, decroissance=0.996):
        for p_ctx, p_cib in zip(self.encodeur_contexte.parameters(), self.encodeur_cible.parameters()):
            p_cib.data.mul_(decroissance).add_(p_ctx.data, alpha=1 - decroissance)

    def forward(self, tenseur):
        """tenseur: (B, T, 4, H, W).
        Retourne (predictions, cibles, contexte_encode) : les trois sont des
        LISTES de longueur len(self.masquage.presets) (2 par defaut), une
        entree par groupe de masque (courte/longue portee). Traites par un
        appel encodeur/predicteur SEPARE par groupe plutot qu'empiles sur la
        dimension batch (comme apply_masks(..., concat=True) dans V-JEPA) :
        nos deux groupes produisent des tailles de contexte/cible
        DIFFERENTES par construction (plus de petits blocs vs moins de gros
        blocs), pas concatenables sur dim=0 (cf. rassembler, masquage.py).
        perte.py moyenne les groupes a egalite a partir de ces listes."""
        patches = self.tokenizer(tenseur)  # (B, N, dim)

        masques_enc, masques_pred = self.masquage.echantillonner(patches.shape[0])

        with torch.no_grad():
            cible_complete = self.encodeur_cible(patches)  # (B, N, dim), sequence complete non masquee

        predictions, cibles, contexte_encode = [], [], []
        for i, (m_enc, m_pred) in enumerate(zip(masques_enc, masques_pred)):
            contexte_i = self.encodeur_contexte(rassembler(patches, m_enc))
            tokens_predicteur_i = self.masquage.tokens_predicteur_groupe(i, m_pred, self.tokenizer.pos_embed)
            predictions.append(self.predicteur(contexte_i, tokens_predicteur_i))
            cibles.append(rassembler(cible_complete, m_pred))
            contexte_encode.append(contexte_i)

        return predictions, cibles, contexte_encode


class ModeleJEPAPoints(nn.Module):
    """Methode points (FPS + kNN + mini-PointNet), pour comparaison."""

    def __init__(
        self, dim=192, dim_predicteur=96,
        profondeur_encodeur=4, profondeur_predicteur=2, nb_tetes=6,
        preset_cible=None,
    ):
        super().__init__()
        self.tokenizer = TokenizerPoints(dim=dim)
        self.masquage = MasquagePoints(dim=dim, preset_cible=preset_cible)

        self.encodeur_contexte = Encodeur(dim, profondeur_encodeur, nb_tetes)

        self.encodeur_cible = copy.deepcopy(self.encodeur_contexte)
        for p in self.encodeur_cible.parameters():
            p.requires_grad = False

        self.predicteur = Predicteur(dim, dim_predicteur, profondeur_predicteur, nb_tetes)

    @torch.no_grad()
    def mettre_a_jour_cible(self, decroissance=0.996):
        for p_ctx, p_cib in zip(self.encodeur_contexte.parameters(), self.encodeur_cible.parameters()):
            p_cib.data.mul_(decroissance).add_(p_ctx.data, alpha=1 - decroissance)

    def forward(self, groupes, centres):
        """groupes: (B,T,K,k,3), centres: (B,T,K,3).
        Retourne (predictions, cibles, contexte_encode)."""
        patches = self.tokenizer(groupes, centres)
        m = self.masquage(patches, centres)

        contexte_encode = self.encodeur_contexte(m["contexte"])
        predictions = self.predicteur(contexte_encode, m["tokens_predicteur"])

        with torch.no_grad():
            cible_encodee_complete = self.encodeur_cible(patches)
        cibles = cible_encodee_complete[:, m["idx_cible"]]

        return predictions, cibles, contexte_encode
