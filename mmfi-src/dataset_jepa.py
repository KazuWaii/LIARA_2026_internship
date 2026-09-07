"""Datasets PyTorch pour le pre-entrainement JEPA -- MM-Fi (deux variantes
grille/points), ONCE (grille, 1er septembre) et JRDB (grille, 3 septembre --
corpus domaine-proche indoor/outdoor humain, cf. Tracking/mois/3-septembre/),
partageant la meme logique de selection de fenetre temporelle.

Chaque item = une fenetre de `longueur_fenetre` frames consecutives, choisie
a une position aleatoire dans la sequence (re-echantillonnee a chaque appel,
comme l'echantillonnage de clips video de V-JEPA)."""

import torch
from torch.utils.data import Dataset

import mmfi_dataset as mmfi
import once_dataset as once
import jrdb_dataset as jrdb
import tokenisation as tok


class _MMFiFenetreDataset(Dataset):
    """Selection de fenetre partagee -- sous-classes : __getitem__ applique
    le pretraitement (grille ou points)."""

    def __init__(self, sequences, racine=mmfi.RACINE_PAR_DEFAUT, longueur_fenetre=16):
        self.racine = racine
        self.longueur_fenetre = longueur_fenetre
        self.sequences = [
            (env, sujet, action) for env, sujet, action in sequences
            if len(mmfi.lister_frames(env, sujet, action, racine)) >= longueur_fenetre
        ]

    def __len__(self):
        return len(self.sequences)

    def _fenetre_brute(self, idx):
        env, sujet, action = self.sequences[idx]
        chemins = mmfi.lister_frames(env, sujet, action, self.racine)

        debut_max = len(chemins) - self.longueur_fenetre
        debut = torch.randint(0, debut_max + 1, (1,)).item() if debut_max > 0 else 0

        return [mmfi.charger_frame_lidar(c) for c in chemins[debut:debut + self.longueur_fenetre]]


class MMFiDatasetJEPA(_MMFiFenetreDataset):
    """Methode grille (tokenisation.py) : chaque item -> (T, 4, H, W)."""

    def __getitem__(self, idx):
        tenseur = tok.empiler_fenetre_temporelle(self._fenetre_brute(idx))
        return torch.from_numpy(tenseur)


class MMFiDatasetJEPAPoints(_MMFiFenetreDataset):
    """Methode points (tokenisation.py) : FPS+kNN par frame (pretraitement
    non-appris, fait ici sur CPU pour que la sortie soit de forme fixe et donc
    batchable). Chaque item -> (groupes (T,K,k,3), centres (T,K,3))."""

    def __getitem__(self, idx):
        fenetre = self._fenetre_brute(idx)
        groupes, centres = zip(*[tok.tokeniser_frame_points(points) for points in fenetre])
        return torch.stack(groupes), torch.stack(centres)


class _OnceFenetreDataset(Dataset):
    """Meme principe que _MMFiFenetreDataset, sur les scenes ONCE (un scene
    id au lieu d'un triplet env/sujet/action). Compense l'ego-mouvement
    (2 septembre) : le vehicule se deplace entre les frames (contrairement au
    capteur fixe de MM-Fi), donc chaque fenetre est ramenee dans le repere de
    sa premiere frame avant projection -- cf. once_dataset.aligner_fenetre."""

    def __init__(
        self, scenes, racine=once.RACINE_PAR_DEFAUT,
        racine_annotations=once.RACINE_ANNOTATIONS_PAR_DEFAUT, longueur_fenetre=16,
    ):
        self.racine = racine
        self.racine_annotations = racine_annotations
        self.longueur_fenetre = longueur_fenetre
        self.scenes = [s for s in scenes if len(once.lister_frames(s, racine)) >= longueur_fenetre]
        self._cache_poses = {}  # scene -> poses, evite de reparser le JSON a chaque fenetre

    def __len__(self):
        return len(self.scenes)

    def _poses_scene(self, scene):
        if scene not in self._cache_poses:
            self._cache_poses[scene] = once.charger_poses(scene, self.racine_annotations)
        return self._cache_poses[scene]

    def _fenetre_brute(self, idx):
        scene = self.scenes[idx]
        chemins = once.lister_frames(scene, self.racine)
        debut_max = len(chemins) - self.longueur_fenetre
        debut = torch.randint(0, debut_max + 1, (1,)).item() if debut_max > 0 else 0
        fenetre_chemins = chemins[debut:debut + self.longueur_fenetre]

        points = [once.charger_frame_lidar(c) for c in fenetre_chemins]
        frame_ids = [int(c.stem) for c in fenetre_chemins]
        return once.aligner_fenetre(points, frame_ids, self._poses_scene(scene))


class OnceDatasetJEPA(_OnceFenetreDataset):
    """Methode grille, profil ONCE (tokenisation.PROFIL_ONCE) : chaque item
    -> (T, 4, H, W)."""

    def __getitem__(self, idx):
        tenseur = tok.empiler_fenetre_temporelle(self._fenetre_brute(idx), profil=tok.PROFIL_ONCE)
        return torch.from_numpy(tenseur)


def decouper_train_val_once(racine=once.RACINE_PAR_DEFAUT, fraction_val=0.1):
    """Decoupage par scene (pas de notion de sujet sur ONCE -- scenes
    disjointes entre train et val, meme principe que le split MM-Fi)."""
    scenes = once.lister_scenes(racine)
    nb_val = max(1, int(len(scenes) * fraction_val))
    return scenes[:-nb_val], scenes[-nb_val:]


class _JrdbFenetreDataset(Dataset):
    """Selection de fenetre JRDB -- meme principe que _MMFiFenetreDataset
    (capteur mobile mais pas de compensation d'ego-mouvement pour l'instant,
    cf. jrdb_dataset.py -- limite documentee, pas de donnees de pose
    telechargees)."""

    def __init__(self, scenes, racine=jrdb.RACINE_PAR_DEFAUT, longueur_fenetre=16):
        self.racine = racine
        self.longueur_fenetre = longueur_fenetre
        self.scenes = [s for s in scenes if len(jrdb.lister_frames(s, racine)) >= longueur_fenetre]

    def __len__(self):
        return len(self.scenes)

    def _fenetre_brute(self, idx):
        scene = self.scenes[idx]
        chemins = jrdb.lister_frames(scene, self.racine)
        debut_max = len(chemins) - self.longueur_fenetre
        debut = torch.randint(0, debut_max + 1, (1,)).item() if debut_max > 0 else 0
        return [jrdb.charger_frame_lidar(c) for c in chemins[debut:debut + self.longueur_fenetre]]


class JrdbDatasetJEPA(_JrdbFenetreDataset):
    """Methode grille, profil JRDB (tokenisation.PROFIL_JRDB) par defaut --
    `profil=tok.PROFIL_MMFI` pour un pre-entrainement combine MM-Fi+JRDB
    (meme forme de grille, batches melangeables), cf. entrainement combine."""

    def __init__(self, scenes, racine=jrdb.RACINE_PAR_DEFAUT, longueur_fenetre=16, profil=None):
        super().__init__(scenes, racine, longueur_fenetre)
        self.profil = profil or tok.PROFIL_JRDB

    def __getitem__(self, idx):
        tenseur = tok.empiler_fenetre_temporelle(self._fenetre_brute(idx), profil=self.profil)
        return torch.from_numpy(tenseur)


def decouper_train_val_jrdb(racine=jrdb.RACINE_PAR_DEFAUT, fraction_val=0.1):
    """Decoupage par sequence (pas de notion de sujet -- scenes de campus
    multi-personnes -- disjointes entre train et val)."""
    scenes = jrdb.lister_scenes(racine)
    nb_val = max(1, int(len(scenes) * fraction_val))
    return scenes[:-nb_val], scenes[-nb_val:]


def decouper_train_val(racine=mmfi.RACINE_PAR_DEFAUT, nb_sujets_val=8):
    """Decoupage par sujet (meme principe que le split cross-subject suggere
    par le toolbox MM-Fi officiel : sujets disjoints entre train et val)."""
    sequences = mmfi.lister_sequences(racine)
    sujets = sorted({s for _, s, _ in sequences})
    sujets_val = set(sujets[-nb_sujets_val:])

    train = [s for s in sequences if s[1] not in sujets_val]
    val = [s for s in sequences if s[1] in sujets_val]
    return train, val


def decouper_train_val_environnement(racine=mmfi.RACINE_PAR_DEFAUT, env_val="E04"):
    """Decoupage par ENVIRONNEMENT -- reproduit exactement le protocole du
    papier Laplacien (arXiv:2509.23255) : "30 sujets E1-E3 train / 10 sujets
    E4 val". Verifie (4 septembre) que MM-Fi partitionne deja les 40 sujets
    par environnement (10 sujets/environnement, aucun chevauchement), donc
    ce split par environnement EST un split par sujet, en plus d'etre un
    test de generalisation inter-environnement -- different de
    decouper_train_val() qui retient les 8 derniers sujets (S33-S40, tous
    dans E04) plutot que les 10 sujets complets d'un environnement."""
    sequences = mmfi.lister_sequences(racine)
    train = [s for s in sequences if s[0] != env_val]
    val = [s for s in sequences if s[0] == env_val]
    return train, val
