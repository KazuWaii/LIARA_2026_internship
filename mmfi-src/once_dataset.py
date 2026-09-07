"""Chargement des scenes LiDAR d'ONCE (raw_small/raw_medium, non etiquetees --
corpus concu pour le pre-entrainement SSL, cf. note 1).

Format verifie sur donnees reelles (extrait le 1er septembre) :
data/{scene}/lidar_roof/{timestamp}.bin, float32, reshape(-1,4) -> x,y,z,r
(intensite). ~67k points/frame, azimut sur 360 deg complets, 40 canaux
verticaux distincts, portee jusqu'a 200m (99e percentile a 111m) -- capteur
et echelle completement differents de MM-Fi, cf. PROFIL_ONCE dans
tokenisation.py.

Compensation de l'ego-mouvement (2 septembre) : contrairement a MM-Fi (capteur
fixe), le vehicule ONCE se deplace entre les frames -- verifie sur donnees
reelles, ~21.8m parcourus en 2.5s (5 frames a 500ms) sur la scene 000000,
~31 km/h. Sans compensation, "la meme cellule de grille" a deux instants
d'une fenetre ne represente pas le meme endroit reel, ce qui casse
l'hypothese meme du masquage plein-temps (V-JEPA) -- cf. AD-LiST-JEPA
(arXiv:2602.12540), qui resout ce probleme en ramenant chaque frame dans un
repere commun avant le masquage. Chaque frame ONCE porte un champ pose =
(qx,qy,qz,qw,tx,ty,tz), relatif a la premiere frame de la SCENE (pas de la
fenetre) -- `aligner_fenetre` recompose la transformation relative a la
premiere frame de la FENETRE, quelle que soit sa position dans la scene."""

import json
from pathlib import Path
import numpy as np

RACINE_PAR_DEFAUT = Path("D:/once_lidar/small/data")
RACINE_ANNOTATIONS_PAR_DEFAUT = Path("D:/once_lidar/small_annotations/data")


def lister_scenes(racine=RACINE_PAR_DEFAUT):
    return sorted(p.name for p in Path(racine).iterdir() if (p / "lidar_roof").is_dir())


def lister_frames(scene, racine=RACINE_PAR_DEFAUT):
    return sorted((Path(racine) / scene / "lidar_roof").glob("*.bin"))


def charger_frame_lidar(chemin_bin):
    """Lit un frame*.bin ONCE -> tableau (N, 4) x,y,z,r (float32)."""
    return np.fromfile(chemin_bin, dtype=np.float32).reshape(-1, 4)[:, :3]  # r non utilise (cf. tokenisation)


def _quat_vers_matrice_rotation(q):
    """q = (x,y,z,w) -> matrice de rotation (3,3). Formule standard, pas
    besoin de scipy pour une seule conversion."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)],
    ])


def charger_poses(scene, racine_annotations=RACINE_ANNOTATIONS_PAR_DEFAUT):
    """Retourne un dict frame_id (int) -> (R (3,3), t (3,)), pose relative a
    la premiere frame de la SCENE (telle que fournie par ONCE)."""
    with open(Path(racine_annotations) / scene / f"{scene}.json") as f:
        data = json.load(f)
    poses = {}
    for fr in data["frames"]:
        qx, qy, qz, qw, tx, ty, tz = fr["pose"]
        poses[int(fr["frame_id"])] = (_quat_vers_matrice_rotation((qx, qy, qz, qw)), np.array([tx, ty, tz]))
    return poses


def aligner_fenetre(fenetre_points, frame_ids, poses):
    """Ramene tous les points de la fenetre dans le repere de sa PREMIERE
    frame (pas celui de la scene) -- recompose la transformation relative a
    partir des poses relatives-a-la-scene fournies par ONCE. fenetre_points :
    liste de (N_i,3), frame_ids : liste d'int (meme ordre)."""
    R_w, t_w = poses[frame_ids[0]]
    alignes = []
    for points, fid in zip(fenetre_points, frame_ids):
        R_i, t_i = poses[fid]
        R_rel = R_w.T @ R_i
        t_rel = R_w.T @ (t_i - t_w)
        alignes.append((R_rel @ points.T).T + t_rel)
    return alignes
