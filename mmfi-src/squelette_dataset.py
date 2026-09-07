"""Dataset multi-tache (4 septembre matin) : fenetre LiDAR grille + squelette
3D verite-terrain aligne (MM-Fi fournit ground_truth.npy, (T,17,3) COCO-style,
un fichier par sequence, JAMAIS utilise jusqu'ici -- seul le label d'action
27-classes servait de signal supervise). Motivation : le squelette est un
signal BEAUCOUP plus dense (51 valeurs continues par frame brute, ~300
frames/sequence) que le label de classe seul (1 valeur/sequence), sur un
corpus de seulement 864 sequences d'entrainement -- cf. Tracking/mois/
4-septembre/ pour la decision et la recherche qui la motive.

Alignement verifie sur donnees reelles : len(ground_truth) == nb frames lidar
pour une sequence donnee (297==297 sur E01/S01/A01), frame i <-> ground_truth[i].

Cible de regression : squelette CENTRE sur son propre centroide par frame
(moyenne des 17 joints retranchee) -- pas la position absolue dans la piece.
Meme logique que la mise en garde deja documentee pour la methode points
(positional embedding absolu = raccourci de localisation plutot que de pose,
cf. README section reconstruction V-JEPA) : on veut forcer le modele a
apprendre la POSE (articulation relative des membres), pas seulement "ou`
dans la piece" -- signal degenere pour la reconnaissance d'action."""

import numpy as np
import torch

import mmfi_dataset as mmfi
import tokenisation as tok
from dataset_jepa import _MMFiFenetreDataset

NB_JOINTS = 17
CODES_ACTIONS = sorted(mmfi.ACTIONS.keys())
INDEX_ACTION = {code: i for i, code in enumerate(CODES_ACTIONS)}


class MMFiDatasetSquelette(_MMFiFenetreDataset):
    """Chaque item -> (tenseur_lidar (T,4,H,W), squelette_cible (nb_t,17,3), label int).

    squelette_cible sous-echantillonne a nb_t pas (moyenne de noyau_t=2 frames
    consecutives, meme reduction temporelle que le tubelet embedding du
    tokenizer -- cf. TokenizerSpatioTemporel) pour etre directement comparable
    a la resolution temporelle des tokens en sortie d'encodeur."""

    def __init__(self, sequences, racine=mmfi.RACINE_PAR_DEFAUT, longueur_fenetre=16,
                 noyau_t=2, profil=tok.PROFIL_MMFI):
        super().__init__(sequences, racine, longueur_fenetre)
        self.noyau_t = noyau_t
        self.profil = profil
        self._cache_squelette = {}

    def _squelette_sequence(self, idx):
        env, sujet, action = self.sequences[idx]
        cle = (env, sujet, action)
        if cle not in self._cache_squelette:
            self._cache_squelette[cle] = mmfi.charger_verite_terrain(env, sujet, action, self.racine)
        return self._cache_squelette[cle]

    def __getitem__(self, idx):
        env, sujet, action = self.sequences[idx]
        chemins = mmfi.lister_frames(env, sujet, action, self.racine)
        debut_max = len(chemins) - self.longueur_fenetre
        debut = torch.randint(0, debut_max + 1, (1,)).item() if debut_max > 0 else 0

        frames_points = [mmfi.charger_frame_lidar(c) for c in chemins[debut:debut + self.longueur_fenetre]]
        tenseur = torch.from_numpy(tok.empiler_fenetre_temporelle(frames_points, profil=self.profil))

        squelette = self._squelette_sequence(idx)[debut:debut + self.longueur_fenetre]  # (T,17,3)
        # Sous-echantillonnage temporel par moyenne de paires consecutives (T -> T//noyau_t),
        # meme reduction que le tubelet embedding (noyau_t=2 par defaut).
        nb_t = self.longueur_fenetre // self.noyau_t
        squelette = squelette[:nb_t * self.noyau_t].reshape(nb_t, self.noyau_t, NB_JOINTS, 3).mean(axis=1)
        squelette = squelette - squelette.mean(axis=1, keepdims=True)  # centre sur le centroide (pose relative)

        label = INDEX_ACTION[action]

        return tenseur, torch.from_numpy(squelette.astype(np.float32)), label


class MMFiDatasetSqueletteSeul(_MMFiFenetreDataset):
    """Squelette verite-terrain SEUL (pas de nuage LiDAR) comme entree
    PRINCIPALE d'un classifieur -- pas un signal auxiliaire de regression
    (cf. MMFiDatasetSquelette ci-dessus) mais l'entree elle-meme, pour
    ModeleSTGCN (modele_stgcn.py). Piste "plafond squelette" (recherche
    5 septembre) : quantifie ce qu'un GNN squelette-standard peut extraire
    du VRAI squelette (pas estime par pose-estimation comme le pipeline
    original MM-Fi), a comparer a notre meilleur resultat LiDAR-only
    (0,659, spectre Laplacien) et au multi-tache squelette-auxiliaire
    (0,549). PAS un resultat LiDAR-only (modalite privilegiee, cf. README).

    Chaque item -> (squelette (T,17,3) float32, label int) -- PAS de
    sous-echantillonnage temporel (contrairement a MMFiDatasetSquelette,
    qui reduisait a la resolution des tokens du tokenizer LiDAR) : le
    squelette est bon marche a charger, on garde toute la resolution
    temporelle de la fenetre (longueur_fenetre frames), ST-GCN reduit
    lui-meme le temps par ses convolutions temporelles a stride 2."""

    def __getitem__(self, idx):
        env, sujet, action = self.sequences[idx]
        chemins = mmfi.lister_frames(env, sujet, action, self.racine)
        debut_max = len(chemins) - self.longueur_fenetre
        debut = torch.randint(0, debut_max + 1, (1,)).item() if debut_max > 0 else 0

        squelette = self._squelette_sequence(idx)[debut:debut + self.longueur_fenetre]  # (T,17,3)
        squelette = squelette - squelette.mean(axis=1, keepdims=True)  # centre sur le centroide (pose relative)

        label = INDEX_ACTION[action]
        return torch.from_numpy(squelette.astype(np.float32)), label

    def _squelette_sequence(self, idx):
        env, sujet, action = self.sequences[idx]
        return mmfi.charger_verite_terrain(env, sujet, action, self.racine)


class MMFiDatasetDistillation(MMFiDatasetSqueletteSeul):
    """Pour entrainer_distillation.py (piste "hallucination de modalite",
    recherche multi-agents 5 septembre, mecanisme de Garcia/Morerio/Murino
    ECCV 2018) : chaque item -> (tenseur_lidar (T,4,H,W), squelette_PLEINE
    RESOLUTION (T,17,3), label). Contrairement a MMFiDatasetSquelette
    (regression auxiliaire, squelette sous-echantillonne a la resolution du
    tokenizer), le squelette ici n'est JAMAIS sous-echantillonne -- il passe
    dans le ModeleSTGCN enseignant (fige), qui attend la MEME resolution
    temporelle (T=longueur_fenetre brute) que celle utilisee pour
    l'entrainer (entrainer_stgcn.py/MMFiDatasetSqueletteSeul). Les deux
    branches (LiDAR, squelette) partagent la MEME fenetre temporelle
    (meme `debut` tire une seule fois) -- alignement necessaire pour que la
    distillation ait un sens (l'enseignant doit voir la MEME action/instant
    que l'etudiant)."""

    def __init__(self, sequences, racine=mmfi.RACINE_PAR_DEFAUT, longueur_fenetre=16,
                 profil=tok.PROFIL_MMFI):
        super().__init__(sequences, racine, longueur_fenetre)
        self.profil = profil

    def __getitem__(self, idx):
        env, sujet, action = self.sequences[idx]
        chemins = mmfi.lister_frames(env, sujet, action, self.racine)
        debut_max = len(chemins) - self.longueur_fenetre
        debut = torch.randint(0, debut_max + 1, (1,)).item() if debut_max > 0 else 0

        frames_points = [mmfi.charger_frame_lidar(c) for c in chemins[debut:debut + self.longueur_fenetre]]
        tenseur = torch.from_numpy(tok.empiler_fenetre_temporelle(frames_points, profil=self.profil))

        squelette = self._squelette_sequence(idx)[debut:debut + self.longueur_fenetre]  # (T,17,3), pleine resolution
        squelette = squelette - squelette.mean(axis=1, keepdims=True)  # centre sur le centroide (pose relative)

        label = INDEX_ACTION[action]
        return tenseur, torch.from_numpy(squelette.astype(np.float32)), label
