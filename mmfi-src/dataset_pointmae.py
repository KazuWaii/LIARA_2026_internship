"""Dataset pour ModeleClassificationPointMAE (modele_pointmae.py) --
fenetre LiDAR -> isolation de la personne PAR FRAME (reutilise
laplacien_spectral._voxels_statiques/_isoler_personne, la meme isolation
que notre meilleur pipeline LiDAR-only, 0,659) -> groupement FPS+kNN PAR
FRAME (reutilise tokenisation.fps/grouper_knn, meme algorithme que la
methode points existante, mais aux valeurs de Point-MAE : 64 centres/32
voisins au lieu de 32/24) -> (T, 64, 32, 3) groupes + (T, 64, 3) centres.

Voxels statiques (fond) mis en cache PAR SEQUENCE (comme
squelette_dataset.MMFiDatasetSquelette._cache_squelette) -- sans ce cache,
les recalculer a CHAQUE pas d'entrainement (fenetre re-echantillonnee a
chaque appel) serait beaucoup trop couteux (scan des ~300 frames de la
sequence entiere a chaque fois)."""

import numpy as np
import torch

import mmfi_dataset as mmfi
import tokenisation as tok
from dataset_jepa import _MMFiFenetreDataset
from laplacien_spectral import _voxels_statiques, _isoler_personne
from modele_pointmae import NUM_GROUP, GROUP_SIZE

CODES_ACTIONS = sorted(mmfi.ACTIONS.keys())
INDEX_ACTION = {code: i for i, code in enumerate(CODES_ACTIONS)}


class MMFiDatasetPointMAE(_MMFiFenetreDataset):
    """Chaque item -> (groupes (T,64,32,3) float32, centres (T,64,3) float32, label int)."""

    def __init__(self, sequences, racine=mmfi.RACINE_PAR_DEFAUT, longueur_fenetre=16,
                 num_group=NUM_GROUP, group_size=GROUP_SIZE):
        super().__init__(sequences, racine, longueur_fenetre)
        self.num_group = num_group
        self.group_size = group_size
        self._cache_voxels = {}

    def _voxels_sequence(self, idx):
        env, sujet, action = self.sequences[idx]
        cle = (env, sujet, action)
        if cle not in self._cache_voxels:
            frames = mmfi.charger_sequence_lidar(env, sujet, action, self.racine)
            self._cache_voxels[cle] = _voxels_statiques(frames)
        return self._cache_voxels[cle]

    def __getitem__(self, idx):
        env, sujet, action = self.sequences[idx]
        chemins = mmfi.lister_frames(env, sujet, action, self.racine)
        debut_max = len(chemins) - self.longueur_fenetre
        debut = torch.randint(0, debut_max + 1, (1,)).item() if debut_max > 0 else 0

        voxels = self._voxels_sequence(idx)
        groupes_fenetre, centres_fenetre = [], []
        for c in chemins[debut:debut + self.longueur_fenetre]:
            brut = mmfi.charger_frame_lidar(c)
            personne = _isoler_personne(brut, voxels)
            if personne.shape[0] == 0:
                personne = np.zeros((1, 3), dtype=np.float32)  # cas degenere (rare) -- garde les formes valides
            groupes, centres = tok.tokeniser_frame_points(personne, self.num_group, self.group_size)
            groupes_fenetre.append(groupes)
            centres_fenetre.append(centres)

        label = INDEX_ACTION[action]
        return torch.stack(groupes_fenetre), torch.stack(centres_fenetre), label
