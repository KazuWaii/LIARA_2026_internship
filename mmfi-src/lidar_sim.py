"""Simulateur LiDAR generique (lancer de rayons contre un maillage) -- brique
pour generer des nuages de points synthetiques a partir de sequences de
mouvement (ex. AMASS + corps SMPL), en attendant l'acces reel a ces donnees
(cf. Tracking/mois/3-septembre/1-despite_smpl_lidar_recherche.txt : creer un
compte AMASS/SMPL est une action que l'agent ne peut pas faire lui-meme).

Base sur trimesh + son moteur de lancer de rayons (recherche du 3 septembre :
aucun outil "cle en main" pour scanner un maillage humain avec un LiDAR
rotatif -- la brique geometrique existe (trimesh/Open3D), le generateur de
rayons LiDAR reste a ecrire soi-meme). Independant de tout format de maillage
particulier : prend n'importe quel trimesh.Trimesh (SMPL ou non), pour pouvoir
etre teste sans donnees SMPL/AMASS reelles."""

import numpy as np
import trimesh


def generer_rayons_lidar(nb_canaux, nb_azimuts, fov_pitch_deg, fov_yaw_deg=(-180.0, 180.0)):
    """Grille de directions de rayons (azimut x elevation), comme un LiDAR
    rotatif multi-faisceaux. Retourne (nb_canaux*nb_azimuts, 3) vecteurs
    unitaires, reperes capteur (x=avant, y=gauche, z=haut -- meme convention
    que tokenisation.py/mmfi_dataset.py)."""
    pitch = np.linspace(np.radians(fov_pitch_deg[0]), np.radians(fov_pitch_deg[1]), nb_canaux)
    yaw = np.linspace(np.radians(fov_yaw_deg[0]), np.radians(fov_yaw_deg[1]), nb_azimuts, endpoint=False)
    pitch_grille, yaw_grille = np.meshgrid(pitch, yaw, indexing="ij")

    x = np.cos(pitch_grille) * np.cos(yaw_grille)
    y = np.cos(pitch_grille) * np.sin(yaw_grille)
    z = np.sin(pitch_grille)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def scanner_maillage(mesh, origine, rayons, portee_max=10.0):
    """Lance `rayons` (N,3, unitaires) depuis `origine` (3,) contre `mesh`
    (trimesh.Trimesh) -- ne garde que le premier impact par rayon (occlusion
    naturelle), jusqu'a `portee_max`. Retourne (K,3) points touches, K<=N."""
    origines = np.tile(origine, (len(rayons), 1))
    points, index_rayons, _ = mesh.ray.intersects_location(
        ray_origins=origines, ray_directions=rayons, multiple_hits=False,
    )
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    distances = np.linalg.norm(points - origine, axis=1)
    dans_portee = distances <= portee_max
    return points[dans_portee].astype(np.float32)


if __name__ == "__main__":
    # Smoke test : pas de donnees SMPL/AMASS disponibles cette nuit (compte a
    # creer par l'utilisateur) -- capsule comme proxy grossier d'un corps
    # humain debout, juste pour valider la mecanique de lancer de rayons.
    corps = trimesh.creation.capsule(height=1.4, radius=0.25, count=(16, 16))
    corps.apply_translation([3.0, 0.0, 0.7])  # debout, ~3m devant le capteur, pieds au sol

    rayons = generer_rayons_lidar(nb_canaux=32, nb_azimuts=360, fov_pitch_deg=(-25.0, 15.0))
    print(f"rayons generes: {rayons.shape}")

    origine = np.array([0.0, 0.0, 1.0])
    nuage = scanner_maillage(corps, origine, rayons, portee_max=10.0)
    print(f"points touches: {nuage.shape[0]} / {rayons.shape[0]} rayons "
          f"({100 * nuage.shape[0] / rayons.shape[0]:.1f}% de la sphere)")
    if nuage.shape[0] > 0:
        print(f"etendue x: [{nuage[:,0].min():.2f}, {nuage[:,0].max():.2f}]  "
              f"y: [{nuage[:,1].min():.2f}, {nuage[:,1].max():.2f}]  "
              f"z: [{nuage[:,2].min():.2f}, {nuage[:,2].max():.2f}]")
    print("SMOKE_TEST_LIDAR_SIM_OK")
