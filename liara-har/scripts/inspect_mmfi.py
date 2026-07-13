from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def find_first_lidar_directory(dataset_root: Path) -> Path:
    """Trouve le premier dossier nommé 'lidar' contenant des frames binaires."""
    for directory in dataset_root.rglob("lidar"):
        if directory.is_dir() and any(directory.glob("frame*.bin")):
            return directory

    raise FileNotFoundError(
        f"Aucun dossier 'lidar' contenant des fichiers frame*.bin "
        f"n'a été trouvé dans : {dataset_root}"
    )

def load_lidar_frame(frame_path: Path) -> np.ndarray:
    """
    Charge une frame LiDAR MM-Fi.

    Le format officiel est une suite de float64 restructurée en N x 3 :
    x, y et z.
    """
    raw_data = np.fromfile(frame_path, dtype=np.float64)

    if raw_data.size == 0:
        raise ValueError(f"Le fichier est vide : {frame_path}")

    if raw_data.size % 3 != 0:
        raise ValueError(
            f"Le fichier contient {raw_data.size} valeurs, "
            "ce qui n'est pas divisible par 3."
        )

    points = raw_data.reshape(-1, 3)

    if not np.isfinite(points).all():
        invalid_count = np.size(points) - np.isfinite(points).sum()
        raise ValueError(
            f"La frame contient {invalid_count} valeur(s) NaN ou infinie(s)."
        )

    return points

def print_dataset_structure(lidar_directory: Path) -> None:
    """Affiche les identifiants environnement, sujet et activité."""
    try:
        action = lidar_directory.parent.name
        subject = lidar_directory.parent.parent.name
        environment = lidar_directory.parent.parent.parent.name

        print("\nSéquence détectée")
        print("-----------------")
        print(f"Environnement : {environment}")
        print(f"Sujet          : {subject}")
        print(f"Activité       : {action}")
        print(f"Dossier LiDAR  : {lidar_directory}")
    except IndexError:
        print(f"Dossier LiDAR : {lidar_directory}")

def print_frame_statistics(points: np.ndarray, frame_path: Path) -> None:
    """Affiche les principales statistiques de la frame."""
    minimums = points.min(axis=0)
    maximums = points.max(axis=0)
    means = points.mean(axis=0)
    standard_deviations = points.std(axis=0)

    distances = np.linalg.norm(points, axis=1)

    print("\nFrame analysée")
    print("--------------")
    print(f"Fichier              : {frame_path.name}")
    print(f"Taille du fichier    : {frame_path.stat().st_size / 1024:.2f} Ko")
    print(f"Forme du tableau     : {points.shape}")
    print(f"Type                  : {points.dtype}")
    print(f"Nombre de points     : {len(points)}")
    print(f"Valeurs toutes finies: {np.isfinite(points).all()}")

    print("\nStatistiques par axe")
    print("--------------------")
    for index, axis in enumerate(("x", "y", "z")):
        print(
            f"{axis} : "
            f"min={minimums[index]:8.3f}, "
            f"max={maximums[index]:8.3f}, "
            f"moyenne={means[index]:8.3f}, "
            f"écart-type={standard_deviations[index]:8.3f}"
        )

    print("\nDistances au capteur")
    print("--------------------")
    print(f"Minimum : {distances.min():.3f}")
    print(f"Maximum : {distances.max():.3f}")
    print(f"Moyenne : {distances.mean():.3f}")
    print(f"Médiane : {np.median(distances):.3f}")

def set_axes_equal(ax: plt.Axes, points: np.ndarray) -> None:
    """Impose une échelle comparable sur les trois axes."""
    minimums = points.min(axis=0)
    maximums = points.max(axis=0)

    centers = (minimums + maximums) / 2
    radius = np.max(maximums - minimums) / 2

    if radius == 0:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)

def plot_point_cloud(
    points: np.ndarray,
    frame_path: Path,
    output_path: Path,
    max_points: int = 30_000,
) -> None:
    """Affiche et sauvegarde le nuage de points."""
    if len(points) > max_points:
        rng = np.random.default_rng(seed=42)
        selected_indices = rng.choice(
            len(points),
            size=max_points,
            replace=False,
        )
        displayed_points = points[selected_indices]
    else:
        displayed_points = points

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")

    axis.scatter(
        displayed_points[:, 0],
        displayed_points[:, 1],
        displayed_points[:, 2],
        s=1,
    )

    axis.set_title(f"MM-Fi — {frame_path.name}")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")

    set_axes_equal(axis, displayed_points)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)

    print(f"\nVisualisation sauvegardée : {output_path.resolve()}")

    plt.show()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspecte la première frame LiDAR disponible dans MM-Fi."
    )

    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Chemin vers le dossier racine de MM-Fi.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/mmfi_first_frame.png"),
        help="Chemin de sortie de la visualisation.",
    )

    return parser.parse_args()

def main() -> int:
    args = parse_arguments()
    dataset_root = args.dataset_root.expanduser().resolve()

    if not dataset_root.exists():
        print(
            f"Erreur : le chemin n'existe pas : {dataset_root}",
            file=sys.stderr,
        )
        return 1

    if not dataset_root.is_dir():
        print(
            f"Erreur : le chemin n'est pas un dossier : {dataset_root}",
            file=sys.stderr,
        )
        return 1

    try:
        lidar_directory = find_first_lidar_directory(dataset_root)
        frame_files = sorted(lidar_directory.glob("frame*.bin"))

        print_dataset_structure(lidar_directory)
        print(f"Nombre de frames dans la séquence : {len(frame_files)}")

        first_frame_path = frame_files[0]
        points = load_lidar_frame(first_frame_path)

        print_frame_statistics(points, first_frame_path)
        plot_point_cloud(points, first_frame_path, args.output)

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())