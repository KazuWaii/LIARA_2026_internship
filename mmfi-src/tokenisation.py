"""Tokenisation, deux methodes (fusionnees dans ce fichier le 31 aout pour
minimiser le nombre de fichiers -- aucun code n'est reellement partage entre
les deux, ce sont deux algorithmes differents regroupes par role).

== Methode grille (projection 2D + conv 3D) ==

Projection multi-canaux (x,y,z,validite) en repere capteur brut (pas de
recentrage, cf. discussion du 31 aout), puis tokenizer V-JEPA repris
directement (conv 3D a stride fixe + positional embedding sin-cos 3D) : notre
grille empilee dans le temps a exactement la forme d'un clip video, donc la
meme tokenisation s'applique sans adaptation -- c'est pourquoi la
representation en grille a ete retenue plutot que la tokenisation par points
seule (note 5). Reprend la logique de resolution de collision (point le plus
proche gagne) de
lidar-src/lidar-exploration/data_projection/projection_unitree_l2_simple.py,
detachee de ROS2. FOV/grille calibres empiriquement sur MM-Fi (note 5, note 6).

== Methode points (FPS + kNN + mini-PointNet) ==

Alternative testee le 31 aout pour comparaison. Repris de Point-JEPA (note 3) :
FPS selectionne K points centres par frame, kNN regroupe les k plus proches
voisins, chaque groupe est normalise (moins le centre) et encode par un
mini-PointNet. Pas de sequenceur glouton : le positional embedding utilise les
coordonnees REELLES (x,y,z,t) de chaque patch (option B du 31 aout) --
l'attention spatio-temporelle apprend la correspondance entre patches d'une
frame a l'autre plutot que la garantir par construction (contrairement a la
grille, qui la garantit gratuitement). K=32/k=24, reduit par rapport a
Point-JEPA (K=64/k=32 sur ShapeNet) pour rester tractable -- avec T=16 frames,
512 patches au total contre 192 pour la grille (difference assumee, pas un
probleme a corriger, cf. README)."""

import numpy as np
import torch
from torch import nn

# ----------------------------------------------------------------------
# Methode grille -- pretraitement (projection)
# ----------------------------------------------------------------------

NB_CANAUX = 4  # x, y, z, validite

# Profils de projection par dataset -- calibres empiriquement sur un
# echantillon de chaque dataset (cf. README). MM-Fi : sujet unique, FOV etroit
# recadre sur la personne. ONCE : scene de conduite complete, 360 deg
# d'azimut, calibre sur le nombre reel de canaux du capteur 40-beam (le
# pitch a 40 pics distincts dans l'histogramme empirique -- HAUTEUR=40 colle
# a la resolution verticale native plutot que d'echantillonner en trop ou pas
# assez).
PROFIL_MMFI = dict(
    hauteur=64, largeur=96,
    fov_yaw=(-60.0, 60.0), fov_pitch=(-35.0, 20.0),
    distance=(1.0, 6.0),
)
PROFIL_ONCE = dict(
    hauteur=40, largeur=180,
    fov_yaw=(-180.0, 180.0), fov_pitch=(-25.0, 15.0),
    distance=(0.5, 120.0),
)
# Calibre le 3 septembre sur donnees reelles (jrdb_dataset.py, lower_velodyne
# seul) : 16 canaux verticaux distincts a exactement -15/+15 degres (pas de 2
# degres, VLP-16 standard), azimut 360 degres, portee p1=0.9m/p50=3.7m/
# p99=34.8m (distance clippee a 40m, au-dela = bruit longue-portee rare).
PROFIL_JRDB = dict(
    hauteur=16, largeur=180,
    fov_yaw=(-180.0, 180.0), fov_pitch=(-15.0, 15.0),
    distance=(0.5, 40.0),
)

# Alias retro-compatibles (valeurs MM-Fi, utilisees si aucun profil n'est passe).
HAUTEUR, LARGEUR = PROFIL_MMFI["hauteur"], PROFIL_MMFI["largeur"]
FOV_YAW_MIN_DEG, FOV_YAW_MAX_DEG = PROFIL_MMFI["fov_yaw"]
FOV_PITCH_MIN_DEG, FOV_PITCH_MAX_DEG = PROFIL_MMFI["fov_pitch"]
DISTANCE_MIN, DISTANCE_MAX = PROFIL_MMFI["distance"]


def projeter_multicanaux(points, profil=PROFIL_MMFI):
    """Projette un nuage (N,3), repere capteur brut -> grille
    (NB_CANAUX, profil['hauteur'], profil['largeur']).

    Canaux : x, y, z (du point le plus proche dans chaque cellule), validite
    (1 si une cellule contient un retour, 0 sinon -- cf. 6-empty_token_masquage).
    """
    hauteur, largeur = profil["hauteur"], profil["largeur"]
    fov_yaw_min, fov_yaw_max = profil["fov_yaw"]
    fov_pitch_min, fov_pitch_max = profil["fov_pitch"]
    distance_min, distance_max = profil["distance"]

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    distance = np.sqrt(x * x + y * y + z * z)

    valide = np.isfinite(distance) & (distance >= distance_min) & (distance <= distance_max)
    x, y, z, distance = x[valide], y[valide], z[valide], distance[valide]

    grille = np.zeros((NB_CANAUX, hauteur, largeur), dtype=np.float32)

    if distance.size == 0:
        return grille

    yaw = np.degrees(-np.arctan2(y, x))
    pitch = np.degrees(np.arcsin(np.clip(z / distance, -1.0, 1.0)))

    dans_fov = (
        (yaw >= fov_yaw_min) & (yaw <= fov_yaw_max)
        & (pitch >= fov_pitch_min) & (pitch <= fov_pitch_max)
    )
    x, y, z = x[dans_fov], y[dans_fov], z[dans_fov]
    yaw, pitch, distance = yaw[dans_fov], pitch[dans_fov], distance[dans_fov]

    if distance.size == 0:
        return grille

    colonne = np.floor(
        (yaw - fov_yaw_min) / (fov_yaw_max - fov_yaw_min) * largeur
    ).astype(np.int32)
    ligne = np.floor(
        (1.0 - (pitch - fov_pitch_min) / (fov_pitch_max - fov_pitch_min)) * hauteur
    ).astype(np.int32)
    colonne = np.clip(colonne, 0, largeur - 1)
    ligne = np.clip(ligne, 0, hauteur - 1)

    # Si plusieurs points tombent dans la meme cellule, seul le plus proche est
    # garde : on trie du plus loin au plus pres, le plus pres ecrase en dernier.
    ordre = np.argsort(distance)[::-1]
    l, c = ligne[ordre], colonne[ordre]
    grille[0, l, c] = x[ordre]
    grille[1, l, c] = y[ordre]
    grille[2, l, c] = z[ordre]
    grille[3, l, c] = 1.0

    return grille


def empiler_fenetre_temporelle(frames_points, profil=PROFIL_MMFI):
    """Projette chaque frame (repere capteur brut) -> tenseur
    (T, NB_CANAUX, profil['hauteur'], profil['largeur'])."""
    grilles = [projeter_multicanaux(points, profil) for points in frames_points]
    return np.stack(grilles, axis=0)


# ----------------------------------------------------------------------
# Methode grille -- tokenizer appris (conv 3D)
# ----------------------------------------------------------------------

def positional_embedding_sincos_3d(dim, nb_t, nb_h, nb_w):
    """Positional embedding fixe (T*H*W, dim), dim divise en 3 pour t/h/w."""
    assert dim % 3 == 0 and (dim // 3) % 2 == 0, "dim doit etre divisible par 3, chaque tiers pair"
    dim_par_axe = dim // 3

    def sincos_1d(taille, dim_axe):
        omega = 1.0 / (10000 ** (np.arange(dim_axe // 2) / (dim_axe // 2)))
        positions = np.arange(taille)
        angles = positions[:, None] * omega[None, :]
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)  # (taille, dim_axe)

    pe_t = sincos_1d(nb_t, dim_par_axe)
    pe_h = sincos_1d(nb_h, dim_par_axe)
    pe_w = sincos_1d(nb_w, dim_par_axe)

    grille = np.zeros((nb_t, nb_h, nb_w, dim), dtype=np.float32)
    grille[..., 0*dim_par_axe:1*dim_par_axe] = pe_t[:, None, None, :]
    grille[..., 1*dim_par_axe:2*dim_par_axe] = pe_h[None, :, None, :]
    grille[..., 2*dim_par_axe:3*dim_par_axe] = pe_w[None, None, :, :]

    return torch.from_numpy(grille.reshape(nb_t * nb_h * nb_w, dim))


class TokenizerSpatioTemporel(nn.Module):
    """Tubelet embedding -- un seul Conv3d a noyau=stride, mirroir de
    PatchEmbed3D (V-JEPA, reference_vjepa/src/models/utils/patch_embed.py) :
    extraction de blocs spatio-temporels non chevauchants + projection
    lineaire en une seule operation.

    Different de V-JEPA sur un point (3 septembre) : la validite (retour
    LiDAR present ou non) est un canal d'ENTREE de la conv (4 canaux
    x,y,z,validite) plutot qu'un signal traite a part -- avant, un token
    "vide" appris remplacait la sortie de la conv pour les cellules sans
    retour (mecanisme AD-L-JEPA/Voxel-MAE). Ce mecanisme a ete retire en
    meme temps que la perte scindee vide/occupe (cf. perte.py) : V-JEPA n'a
    pas cette notion (pixels toujours presents) et n'en a pas besoin -- le
    modele apprend a representer "pas de retour" implicitement a partir du
    canal de validite, comme il apprend a representer n'importe quel
    contenu de pixel."""

    def __init__(self, dim=192, noyau_t=2, noyau_spatial=16, hauteur=64, largeur=96, nb_frames=16):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv3d(
            in_channels=4,  # x, y, z, validite
            out_channels=dim,
            kernel_size=(noyau_t, noyau_spatial, noyau_spatial),
            stride=(noyau_t, noyau_spatial, noyau_spatial),
        )

        self.nb_t = nb_frames // noyau_t
        self.nb_h = hauteur // noyau_spatial
        self.nb_w = largeur // noyau_spatial
        self.noyau_t = noyau_t
        self.noyau_spatial = noyau_spatial

        pos_embed = positional_embedding_sincos_3d(dim, self.nb_t, self.nb_h, self.nb_w)
        self.register_buffer("pos_embed", pos_embed)  # (nb_t*nb_h*nb_w, dim), non entraine

    def forward(self, tenseur):
        """tenseur: (B, T, 4, H, W) -- canaux [x, y, z, validite].
        Retourne embeddings: (B, nb_t*nb_h*nb_w, dim)."""
        entree = tenseur.permute(0, 2, 1, 3, 4)  # (B,4,T,H,W) pour Conv3d

        patches = self.conv(entree)  # (B, dim, nb_t, nb_h, nb_w)
        B = patches.shape[0]
        patches = patches.permute(0, 2, 3, 4, 1).reshape(B, -1, self.dim)  # (B, N, dim)
        patches = patches + self.pos_embed.unsqueeze(0)
        return patches


# ----------------------------------------------------------------------
# Methode points -- pretraitement (FPS + kNN)
# ----------------------------------------------------------------------

K_CENTRES = 32
K_VOISINS = 24


def fps(points, k_centres):
    """Farthest point sampling -- indices (k_centres,) dans points (N,3).
    Depart deterministe : le point le plus proche de l'origine (repere
    capteur), pour reproductibilite."""
    N = points.shape[0]
    k_demande = k_centres  # BUG corrige (5 septembre) : la comparaison de padding
    # ci-dessous utilisait la constante globale K_CENTRES au lieu de CE PARAMETRE --
    # invisible tant que fps() n'etait appele qu'avec k_centres=K_CENTRES (methode
    # points du projet), mais renvoyait SILENCIEUSEMENT moins d'indices que demande
    # des qu'appele avec une autre valeur (ex. modele_pointmae.py, k_centres=64).
    k_centres = min(k_centres, N)
    indices = torch.zeros(k_centres, dtype=torch.long, device=points.device)
    distances = torch.full((N,), float("inf"), device=points.device)

    idx = torch.argmin((points ** 2).sum(-1))
    for i in range(k_centres):
        indices[i] = idx
        d = ((points - points[idx]) ** 2).sum(-1)
        distances = torch.minimum(distances, d)
        idx = torch.argmax(distances)

    if k_centres < k_demande:  # frame avec peu de points -- complete par repetition
        # indices[:manquant] serait FAUX si manquant > k_centres (tronque
        # silencieusement au lieu de repeter en boucle, bug trouve le 5
        # septembre en testant avec k_demande=64 sur des nuages a <16 points) --
        # indexation modulo pour boucler correctement quel que soit manquant.
        manquant = k_demande - k_centres
        indices = torch.cat([indices, indices[torch.arange(manquant) % k_centres]])
    return indices


def grouper_knn(points, indices_centres, k_voisins):
    """Pour chaque centre, les k plus proches voisins, coordonnees normalisees
    (moins le centre). Retourne (groupes (K,k,3), centres (K,3)). Complete
    par repetition du plus proche voisin si moins de k_voisins points sont
    disponibles au total (5 septembre -- necessaire pour les nuages
    ISOLES/parcimonieux, ex. modele_pointmae.py sur la personne apres
    soustraction de fond+DBSCAN, ~50-150 points, contrairement a l'usage
    existant de cette fonction sur les frames BRUTES, toujours >=1009
    points -- jamais sous-dimensionnees jusqu'ici)."""
    centres = points[indices_centres]
    k_demande = k_voisins
    k_voisins = min(k_voisins, points.shape[0])
    d = torch.cdist(centres, points)
    _, idx_voisins = torch.topk(d, k_voisins, largest=False, dim=-1)
    if k_voisins < k_demande:
        manquant = k_demande - k_voisins
        idx_voisins = torch.cat([idx_voisins, idx_voisins[:, :1].expand(-1, manquant)], dim=1)
    groupes = points[idx_voisins] - centres.unsqueeze(1)
    return groupes, centres


def tokeniser_frame_points(points, k_centres=K_CENTRES, k_voisins=K_VOISINS):
    """Pretraitement non-appris (CPU, dans le Dataset) : nuage (N,3) brut ->
    groupes (K,k,3) normalises + centres (K,3), repere capteur brut."""
    points = torch.as_tensor(points, dtype=torch.float32)
    indices = fps(points, k_centres)
    return grouper_knn(points, indices, k_voisins)


# ----------------------------------------------------------------------
# Methode points -- tokenizer appris (mini-PointNet)
# ----------------------------------------------------------------------

class MiniPointNet(nn.Module):
    """PointNet minimal (Point-JEPA, note 3) : MLP partage + max-pool +
    concatenation + MLP partage + max-pool -> un embedding par groupe."""

    def __init__(self, dim):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(3, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 2))
        self.mlp2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, groupes):
        """groupes: (..., k, 3) -> (..., dim)."""
        f = self.mlp1(groupes)
        f_max = f.max(dim=-2, keepdim=True).values.expand_as(f)
        f = torch.cat([f, f_max], dim=-1)
        f = self.mlp2(f)
        return f.max(dim=-2).values


def embedding_position_continue(valeurs, dim):
    """Sin-cos sur une coordonnee continue (pas un indice de grille) --
    valeurs: (...,) -> (..., dim)."""
    omega = 1.0 / (10000 ** (torch.arange(dim // 2, device=valeurs.device, dtype=torch.float32) / (dim // 2)))
    angles = valeurs.unsqueeze(-1) * omega
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def positional_embedding_points(centres, t_idx, dim):
    """Position (x,y,z,t) reelle -> embedding sin-cos, dim divise en 4 tiers.
    centres: (...,3), t_idx: (...,) -- meme forme sans la derniere dimension.
    Fonction partagee entre TokenizerPoints (tokens de contexte/cible) et
    MasquagePoints (tokens masque du predicteur, cf. masquage.py), pour que
    les deux utilisent exactement le meme encodage de position."""
    dim_par_axe = dim // 4
    return torch.cat([
        embedding_position_continue(centres[..., 0], dim_par_axe),
        embedding_position_continue(centres[..., 1], dim_par_axe),
        embedding_position_continue(centres[..., 2], dim_par_axe),
        embedding_position_continue(t_idx, dim_par_axe),
    ], dim=-1)


class TokenizerPoints(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        assert dim % 4 == 0, "dim divise en 4 (x,y,z,t) pour le positional embedding"
        self.dim = dim
        self.mini_pointnet = MiniPointNet(dim)

    def forward(self, groupes, centres):
        """groupes: (B,T,K,k,3) normalises, centres: (B,T,K,3) coordonnees
        reelles (repere capteur brut). Retourne (B, T*K, dim)."""
        B, T, K, k, _ = groupes.shape

        embeddings = self.mini_pointnet(groupes)  # (B,T,K,dim)

        t_idx = torch.arange(T, device=groupes.device, dtype=torch.float32).view(1, T, 1).expand(B, T, K)
        pe = positional_embedding_points(centres, t_idx, self.dim)  # (B,T,K,dim)

        return (embeddings + pe).reshape(B, T * K, self.dim)
