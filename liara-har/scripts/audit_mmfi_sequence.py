from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def frame_number(path: Path) -> int:
    """Extrait le numéro depuis un nom comme frame001.bin."""
    match = re.search(r"(\d+)", path.stem)

    if match is None:
        raise ValueError(f"Numéro de frame introuvable : {path.name}")

    return int(match.group(1))


def load_frame(path: Path) -> np.ndarray:
    """Charge une frame MM-Fi au format N x 3 en float64."""
    values = np.fromfile(path, dtype=np.float64)

    if values.size == 0:
        raise ValueError(f"Frame vide : {path}")

    if values.size % 3 != 0:
        raise ValueError(
            f"Nombre de valeurs incorrect dans {path.name} : "
            f"{values.size}"
        )

    points = values.reshape(-1, 3)

    if not np.isfinite(points).all():
        raise ValueError(f"Valeurs non finies dans {path.name}")

    return points


def load_sequence(lidar_directory: Path) -> tuple[list[Path], list[np.ndarray]]:
    """Charge toutes les frames d'un dossier LiDAR."""
    frame_paths = sorted(
        lidar_directory.glob("frame*.bin"),
        key=frame_number,
    )

    if not frame_paths:
        raise FileNotFoundError(
            f"Aucune frame trouvée dans : {lidar_directory}"
        )

    frames = [load_frame(path) for path in frame_paths]
    return frame_paths, frames


def compute_statistics(
    frame_paths: list[Path],
    frames: list[np.ndarray],
) -> list[dict[str, float | int | str]]:
    """Calcule les statistiques de chaque frame."""
    rows: list[dict[str, float | int | str]] = []

    for path, points in zip(frame_paths, frames):
        minimums = points.min(axis=0)
        maximums = points.max(axis=0)
        centroid = points.mean(axis=0)

        rows.append(
            {
                "frame": path.name,
                "frame_number": frame_number(path),
                "number_of_points": len(points),
                "x_min": minimums[0],
                "x_max": maximums[0],
                "y_min": minimums[1],
                "y_max": maximums[1],
                "z_min": minimums[2],
                "z_max": maximums[2],
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "centroid_z": centroid[2],
            }
        )

    return rows


def save_statistics(
    rows: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    """Sauvegarde les statistiques dans un fichier CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_point_counts(
    rows: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    """Trace l'évolution du nombre de points."""
    frame_indices = [
        int(row["frame_number"])
        for row in rows
    ]
    point_counts = [
        int(row["number_of_points"])
        for row in rows
    ]

    figure, axis = plt.subplots(figsize=(12, 5))

    axis.plot(frame_indices, point_counts)
    axis.set_xlabel("Numéro de frame")
    axis.set_ylabel("Nombre de points")
    axis.set_title("MM-Fi — Nombre de points par frame")
    axis.grid(True)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def animate_sequence(
    frames: list[np.ndarray],
    frame_paths: list[Path],
    interval_ms: int,
) -> None:
    """Anime la séquence avec une vue 3D et une vue de dessus."""
    all_minimums = np.min(
        np.vstack([points.min(axis=0) for points in frames]),
        axis=0,
    )
    all_maximums = np.max(
        np.vstack([points.max(axis=0) for points in frames]),
        axis=0,
    )

    figure = plt.figure(figsize=(14, 7))

    axis_3d = figure.add_subplot(121, projection="3d")
    axis_bev = figure.add_subplot(122)

    def update(frame_index: int):
        points = frames[frame_index]

        axis_3d.clear()
        axis_bev.clear()

        axis_3d.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=2,
        )

        axis_3d.set_xlim(all_minimums[0], all_maximums[0])
        axis_3d.set_ylim(all_minimums[1], all_maximums[1])
        axis_3d.set_zlim(all_minimums[2], all_maximums[2])

        axis_3d.set_xlabel("x")
        axis_3d.set_ylabel("y")
        axis_3d.set_zlabel("z")
        axis_3d.set_title(
            f"Vue 3D — {frame_paths[frame_index].name}"
        )

        axis_bev.scatter(
            points[:, 0],
            points[:, 1],
            s=2,
        )

        axis_bev.set_xlim(all_minimums[0], all_maximums[0])
        axis_bev.set_ylim(all_minimums[1], all_maximums[1])
        axis_bev.set_aspect("equal", adjustable="box")
        axis_bev.set_xlabel("x")
        axis_bev.set_ylabel("y")
        axis_bev.set_title("Vue de dessus")
        axis_bev.grid(True)

        figure.suptitle(
            f"Frame {frame_index + 1}/{len(frames)}"
        )

        return []

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=interval_ms,
        repeat=True,
        blit=False,
    )

    # Conserver la référence jusqu'à la fermeture de la fenêtre.
    _ = animation
    plt.tight_layout()
    plt.show()


def print_summary(frames: list[np.ndarray]) -> None:
    point_counts = np.asarray(
        [len(points) for points in frames],
        dtype=np.int64,
    )

    print("\nRésumé de la séquence")
    print("---------------------")
    print(f"Nombre de frames       : {len(frames)}")
    print(f"Points minimum         : {point_counts.min()}")
    print(f"Points maximum         : {point_counts.max()}")
    print(f"Points moyens          : {point_counts.mean():.2f}")
    print(f"Écart-type             : {point_counts.std():.2f}")
    print(f"Médiane                : {np.median(point_counts):.2f}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse et anime une séquence LiDAR MM-Fi."
    )

    parser.add_argument(
        "lidar_directory",
        type=Path,
        help="Chemin vers un dossier E../S../A../lidar.",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=100,
        help="Durée entre deux frames en millisecondes.",
    )

    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Calcule uniquement les statistiques.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    lidar_directory = args.lidar_directory.expanduser().resolve()

    if not lidar_directory.is_dir():
        raise NotADirectoryError(
            f"Dossier invalide : {lidar_directory}"
        )

    frame_paths, frames = load_sequence(lidar_directory)
    rows = compute_statistics(frame_paths, frames)

    sequence_name = "_".join(
        [
            lidar_directory.parent.parent.parent.name,
            lidar_directory.parent.parent.name,
            lidar_directory.parent.name,
        ]
    )

    csv_output = Path(
        f"outputs/{sequence_name}_statistics.csv"
    )
    plot_output = Path(
        f"outputs/{sequence_name}_point_counts.png"
    )

    save_statistics(rows, csv_output)
    plot_point_counts(rows, plot_output)
    print_summary(frames)

    print(f"\nStatistiques : {csv_output.resolve()}")
    print(f"Courbe       : {plot_output.resolve()}")

    if not args.no_animation:
        animate_sequence(
            frames=frames,
            frame_paths=frame_paths,
            interval_ms=args.interval,
        )


if __name__ == "__main__":
    main()