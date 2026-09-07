"""Chargement des sequences LiDAR de JRDB (train split, non etiquete pour
notre usage -- corpus de pre-entrainement domaine-proche, cf. Tracking/mois/
3-septembre/, suite a la lecture de DeSPITE qui montre l'interet d'un corpus
plus grand pour le pre-entrainement point-cloud humain).

Format verifie sur donnees reelles (extrait le 3 septembre) :
train/pointclouds/lower_velodyne/{sequence}/{frame:06d}.pcd, PCD binaire
compresse (champs x,y,z,intensite,ring), ~17k points/frame. Capteur : un des
deux Velodyne VLP-16 du robot JackRabbot (sur les deux disponibles --
lower_velodyne seulement, upper_velodyne ignore pour l'instant pour eviter
une fusion multi-capteurs non calibree). Calibre empiriquement : 16 canaux
verticaux distincts a exactement -15 a +15 degres (pas de 2 degres, confirme
VLP-16 standard), azimut 360 degres complets, portee p1=0.9m/p50=3.7m/
p99=34.8m.

LIMITE CONNUE (3 septembre, pas encore resolue) : le robot JackRabbot SE
DEPLACE sur le campus pendant la capture -- contrairement a MM-Fi (capteur
fixe), et comme ONCE (vehicule, cf. once_dataset.py). Contrairement a ONCE,
aucune donnee de pose/odometrie n'a ete telechargee ici (seuls les zips
"Train Data, Subsets" de nuages de points) -- PAS de compensation
d'ego-mouvement appliquee pour l'instant. Le masquage plein-temps (V-JEPA)
suppose une meme cellule de grille = meme endroit reel d'une frame a l'autre
au sein d'une fenetre, ce qui est invalide pendant les deplacements du robot.
A corriger si les resultats de pre-entrainement sur JRDB semblent degrades
par ce biais -- necessiterait de telecharger separement les donnees de pose
du robot (compte JRDB deja cree, telechargement supplementaire seulement)."""

from pathlib import Path

from pypcd4 import PointCloud

RACINE_PAR_DEFAUT = Path("D:/jrdb/train/pointclouds/lower_velodyne")


def lister_scenes(racine=RACINE_PAR_DEFAUT):
    return sorted(p.name for p in Path(racine).iterdir() if p.is_dir())


def lister_frames(scene, racine=RACINE_PAR_DEFAUT):
    return sorted((Path(racine) / scene).glob("*.pcd"))


def charger_frame_lidar(chemin_pcd):
    """Lit un frame*.pcd JRDB (binaire, potentiellement compresse) ->
    tableau (N, 3) x,y,z (float32). intensite/ring non utilises (coherent
    avec ONCE, cf. once_dataset.py)."""
    return PointCloud.from_path(str(chemin_pcd)).numpy()[:, :3]
