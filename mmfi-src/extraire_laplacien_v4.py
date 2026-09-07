"""Re-extraction des features Laplacien (quadrants, cf. laplacien_spectral.py)
sur les 1080 sequences MM-Fi, EN UNE SEULE PASSE, pour deux ablations en
suite de la piste B (README/tracking 4 septembre, meilleur resultat 0,617
SVM avec le split par sujet) :

1. Split par ENVIRONNEMENT (E01-E03 train / E04 val) -- reproduit EXACTEMENT
   le protocole du papier arXiv:2509.23255 ("30 sujets E1-E3 train / 10
   sujets E4 val"), verifie ici que les sujets sont deja partitionnes par
   environnement dans MM-Fi (10 sujets/environnement, aucun chevauchement)
   -- ce split est donc a la fois plus comparable a leurs 81-90% ET un test
   de generalisation inter-environnement (pas seulement inter-sujet).
2. Fenetrage plus dense (nb_fenetres=8 au lieu de 4) -- plus de couverture
   temporelle par sequence, cout ~2x (mais parallelise sur les coeurs
   disponibles, contrairement aux scripts v1-v3 qui tournaient sequentiel).

Sauvegarde TOUTES les fenetres de TOUTES les sequences avec leurs metadonnees
(environnement, sujet) dans un seul fichier -- entrainer_laplacien_v4.py
reconstruit ensuite les deux splits (sujet et environnement) SANS re-extraire,
pour comparer les deux protocoles sur exactement les memes features."""

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
            quadrants=True)
    except Exception as e:
        print(f"  [!] sequence {env}/{sujet}/{action} echouee: {e}", flush=True)
        return None
    label = INDEX_ACTION[action]
    return feats, label, env, sujet


if __name__ == "__main__":
    sequences = mmfi.lister_sequences()
    print(f"sequences totales: {len(sequences)}  nb_fenetres={NB_FENETRES}  "
          f"nb_frames_echantillon={NB_FRAMES_ECHANTILLON}")

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

    np.savez("mmfi-src/features_laplacien_v4_all.npz",
             X=X, y=y, envs=envs, sujets=sujets)
    print("features sauvegardees -> mmfi-src/features_laplacien_v4_all.npz")
    print("EXTRACTION_LAPLACIEN_V4_OK")
