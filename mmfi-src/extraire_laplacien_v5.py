"""Comme extraire_laplacien_v4.py, mais avec `nettoyage=True` (RANSAC
retrait de plans sol/mur + retrait de bruit statistique AVANT DBSCAN, cf.
laplacien_spectral._isoler_personne) -- teste l'hypothese de la recherche du
5 septembre (rang 3 du plan) : l'isolation de la personne (soustraction de
fond + DBSCAN seuls) est moins fidele que celle du papier de reference
(detecteur 3D supervise), et un nettoyage bon marche (RANSAC+outlier
removal, sans entrainement/labels) pourrait recuperer une partie de l'ecart
avant de tenter la piste plus couteuse (detecteur appris). Meme structure
que v4 (1080 sequences, 8 fenetres/sequence, parallelise) pour comparaison
directe : entrainer_laplacien_v5.py reevalue avec les hyperparametres deja
regles (SVM C=3/gamma=scale, HGB tune) sur les DEUX splits (sujet,
environnement) contre les references v4 (0,659/0,653 sans nettoyage)."""

import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import mmfi_dataset as mmfi
from laplacien_spectral import extraire_fenetres_sequence

CODES_ACTIONS = sorted(mmfi.ACTIONS.keys())
INDEX_ACTION = {code: i for i, code in enumerate(CODES_ACTIONS)}

NB_FENETRES = 8
NB_FRAMES_ECHANTILLON = 8


def _traiter_sequence(args):
    env, sujet, action = args
    frames = mmfi.lister_frames(env, sujet, action)
    pts = [mmfi.charger_frame_lidar(f) for f in frames]
    if len(pts) < 5:
        return None
    try:
        feats = extraire_fenetres_sequence(
            pts, nb_fenetres=NB_FENETRES, nb_frames_echantillon=NB_FRAMES_ECHANTILLON,
            quadrants=True, nettoyage=True)
    except Exception as e:
        print(f"  [!] sequence {env}/{sujet}/{action} echouee: {e}", flush=True)
        return None
    label = INDEX_ACTION[action]
    return feats, label, env, sujet


if __name__ == "__main__":
    sequences = mmfi.lister_sequences()
    print(f"sequences totales: {len(sequences)}  nb_fenetres={NB_FENETRES}  "
          f"nb_frames_echantillon={NB_FRAMES_ECHANTILLON}  nettoyage=True")

    nb_procs = max(1, (os.cpu_count() or 4) - 2)
    print(f"extraction parallele sur {nb_procs} processus...")

    X, y, envs, sujets = [], [], [], []
    t0 = time.time()
    with Pool(processes=nb_procs) as pool:
        for i, res in enumerate(pool.imap(_traiter_sequence, sequences, chunksize=4)):
            if res is not None:
                feats, label, env, sujet = res
                X.extend(feats)
                y.extend([label] * len(feats))
                envs.extend([env] * len(feats))
                sujets.extend([sujet] * len(feats))
            if (i + 1) % 50 == 0:
                duree = time.time() - t0
                restant = duree / (i + 1) * (len(sequences) - i - 1)
                print(f"  {i + 1}/{len(sequences)} sequences  [{duree:.1f}s ecoulees, "
                      f"~{restant:.0f}s restantes]", flush=True)

    X = np.stack(X)
    y = np.array(y)
    envs = np.array(envs)
    sujets = np.array(sujets)
    duree_totale = time.time() - t0
    print(f"\nextraction terminee en {duree_totale:.1f}s -- X: {X.shape}  y: {y.shape}")

    np.savez("mmfi-src/features_laplacien_v5_all.npz",
             X=X, y=y, envs=envs, sujets=sujets)
    print("features sauvegardees -> mmfi-src/features_laplacien_v5_all.npz")
    print("EXTRACTION_LAPLACIEN_V5_OK")
