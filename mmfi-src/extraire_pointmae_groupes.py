"""Precalcule les GROUPES BRUTS (pas encore passes dans l'encodeur, cf.
extraire_pointmae.py qui precalcule les features POOLEES d'un encodeur
FIGE) -- necessaire pour le FINE-TUNING complet de l'encodeur Point-MAE
(entrainer_pointmae_finetune.py) : contrairement au linear probing, les
poids de l'encodeur changent a chaque pas, donc ses features de sortie ne
peuvent PAS etre precalculees une fois pour toutes -- seul le pretraitement
CPU (isolation personne + FPS/kNN, ~17s/pas mesure "a la volee", cf.
Tracking) est independant des poids et peut etre precalcule.

Meme fenetrage que extraire_pointmae.py (4 fenetres/sequence, 8 frames
echantillonnees/fenetre) pour rester comparable au linear probing (0,934)."""

import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import mmfi_dataset as mmfi
from extraire_pointmae import _extraire_groupes_sequence, NB_FENETRES, NB_FRAMES_ECHANTILLON

if __name__ == "__main__":
    sequences = mmfi.lister_sequences()
    print(f"sequences totales: {len(sequences)}  nb_fenetres={NB_FENETRES}  "
          f"nb_frames_echantillon={NB_FRAMES_ECHANTILLON}", flush=True)

    nb_procs = max(1, (os.cpu_count() or 4) - 2)
    print(f"extraction CPU (isolation+FPS/kNN) parallele sur {nb_procs} processus...", flush=True)

    t0 = time.time()
    tous_groupes, tous_centres, labels, envs, sujets = [], [], [], [], []
    with Pool(processes=nb_procs) as pool:
        for i, res in enumerate(pool.imap(_extraire_groupes_sequence, sequences, chunksize=2)):
            if res is not None:
                groupes_tout, centres_tout, label, env, sujet = res
                tous_groupes.append(groupes_tout.astype(np.float16))  # float16 -- divise la taille par 2
                tous_centres.append(centres_tout.astype(np.float16))
                labels.extend([label] * NB_FENETRES)
                envs.extend([env] * NB_FENETRES)
                sujets.extend([sujet] * NB_FENETRES)
            if (i + 1) % 100 == 0:
                duree = time.time() - t0
                restant = duree / (i + 1) * (len(sequences) - i - 1)
                print(f"  {i + 1}/{len(sequences)} sequences  [{duree:.1f}s ecoulees, "
                      f"~{restant:.0f}s restantes]", flush=True)
    print(f"extraction CPU terminee en {time.time()-t0:.1f}s", flush=True)

    tous_groupes = np.concatenate(tous_groupes, axis=0)  # (N_fenetres, T, G, K, 3)
    tous_centres = np.concatenate(tous_centres, axis=0)
    labels, envs, sujets = np.array(labels), np.array(envs), np.array(sujets)
    print(f"groupes: {tous_groupes.shape} ({tous_groupes.nbytes/1e9:.2f} Go)  centres: {tous_centres.shape}")

    np.savez("mmfi-src/groupes_pointmae.npz",
             groupes=tous_groupes, centres=tous_centres, y=labels, envs=envs, sujets=sujets)
    print("groupes sauvegardes -> mmfi-src/groupes_pointmae.npz")
    print("EXTRACTION_GROUPES_POINTMAE_OK")
