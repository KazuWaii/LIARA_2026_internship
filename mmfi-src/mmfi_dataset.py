"""Chargement des sequences LiDAR de MM-Fi (nuages de points + verite terrain)."""

from pathlib import Path
import numpy as np

RACINE_PAR_DEFAUT = Path("data/MMFi_lidar_only/MMFi_Dataset")

# Table 5 du papier MM-Fi (arXiv:2305.10345) : A02-A05/A13-A14/A17-A23/A27
# sont des activites quotidiennes, le reste des activites de readaptation.
ACTIONS = {
    "A01": "Stretching and relaxing",
    "A02": "Chest expansion (horizontal)",
    "A03": "Chest expansion (vertical)",
    "A04": "Twist (left)",
    "A05": "Twist (right)",
    "A06": "Mark time",
    "A07": "Limb extension (left)",
    "A08": "Limb extension (right)",
    "A09": "Lunge (toward left-front)",
    "A10": "Lunge (toward right-front)",
    "A11": "Limb extension (both)",
    "A12": "Squat",
    "A13": "Raising hand (left)",
    "A14": "Raising hand (right)",
    "A15": "Lunge (toward left)",
    "A16": "Lunge (toward right)",
    "A17": "Waving hand (left)",
    "A18": "Waving hand (right)",
    "A19": "Picking up things",
    "A20": "Throwing (toward left)",
    "A21": "Throwing (toward right)",
    "A22": "Kicking (toward left)",
    "A23": "Kicking (toward right)",
    "A24": "Body extension (left)",
    "A25": "Body extension (right)",
    "A26": "Jumping up",
    "A27": "Bowing",
}


def lister_sequences(racine=RACINE_PAR_DEFAUT):
    """Retourne la liste (env, sujet, action) de toutes les sequences avec du lidar."""
    racine = Path(racine)
    sequences = []
    for env_dir in sorted(racine.glob("E*")):
        for sujet_dir in sorted(env_dir.glob("S*")):
            for action_dir in sorted(sujet_dir.glob("A*")):
                if (action_dir / "lidar").is_dir():
                    sequences.append((env_dir.name, sujet_dir.name, action_dir.name))
    return sequences


def chemin_sequence(env, sujet, action, racine=RACINE_PAR_DEFAUT):
    return Path(racine) / env / sujet / action


def charger_frame_lidar(chemin_bin):
    """Lit un fichier frame*.bin -> tableau (N, 3) de points x,y,z (float64)."""
    return np.fromfile(chemin_bin, dtype=np.float64).reshape(-1, 3)


def lister_frames(env, sujet, action, racine=RACINE_PAR_DEFAUT):
    dossier = chemin_sequence(env, sujet, action, racine) / "lidar"
    return sorted(dossier.glob("frame*.bin"))


def charger_sequence_lidar(env, sujet, action, racine=RACINE_PAR_DEFAUT):
    """Charge toutes les frames d'une sequence -> liste de tableaux (N_i, 3)."""
    return [charger_frame_lidar(f) for f in lister_frames(env, sujet, action, racine)]


def charger_verite_terrain(env, sujet, action, racine=RACINE_PAR_DEFAUT):
    """Squelette 3D (T, 17, 3) associe a la sequence."""
    chemin = chemin_sequence(env, sujet, action, racine) / "ground_truth.npy"
    return np.load(chemin, allow_pickle=True)


def nom_action(code_action):
    return ACTIONS.get(code_action, code_action)
