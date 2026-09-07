"""Precalcule les features Point-MAE (encodeur ShapeNet-pretrained, FIGE)
pour toutes les sequences MM-Fi, EN UNE SEULE PASSE -- separe le
pretraitement COUTEUX (isolation personne + FPS/kNN, CPU, ~0.5s/sequence)
et le forward de l'encodeur (GPU) de l'ENTRAINEMENT (qui devient alors
quasi instantane, juste un pooling temporel + une tete lineaire sur des
features DEJA calculees) -- meme principe que extraire_laplacien_v4/v5.py
pour le pipeline classique. Necessaire : un premier essai d'entrainement
"a la volee" (isolation+FPS/kNN+encodeur recalcules a chaque pas) prenait
~15-17s/pas meme avec 6 workers CPU en parallele -- bien trop lent pour un
sweep + protocole 3 graines (cf. session 5 septembre).

Comme l'encodeur est FIGE (geler_encodeur=True, linear probing -- seule
config testee ici par contrainte de temps, cf. README/tracking : PointGST
recommande justement le linear probing plutot que le fine-tuning complet a
cette echelle de donnees), ses features ne dependent JAMAIS des poids
entraines -- les precalculer une fois est strictement equivalent a les
recalculer a chaque pas, juste ~800x plus rapide en pratique (calcule une
fois au lieu de 800 pas x nb graines).

4 fenetres/sequence, 8 frames sous-echantillonnees/fenetre (meme convention
que le pipeline Laplacien) -- feature = pooling mean+max sur les 64 groupes
Point-MAE (768 dims) PAR FRAME, PAS encore poolee sur le temps (le pooling
temporel reste appris/variable, fait a l'entrainement)."""

import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import mmfi_dataset as mmfi
import tokenisation as tok
from laplacien_spectral import _voxels_statiques, _isoler_personne
from modele_pointmae import EncodeurPointMAE, charger_poids_pretrain, NUM_GROUP, GROUP_SIZE

CODES_ACTIONS = sorted(mmfi.ACTIONS.keys())
INDEX_ACTION = {code: i for i, code in enumerate(CODES_ACTIONS)}

# cf. entrainer_pointmae.py pour la provenance/URL de ce checkpoint.
CHECKPOINT_PRETRAIN = r"D:\pretrained_checkpoints\point_mae\pretrain.pth"
NB_FENETRES = 4
NB_FRAMES_ECHANTILLON = 8


def _extraire_groupes_sequence(args):
    """CPU seul (multiprocessing) : isolation + FPS/kNN pour toutes les
    frames echantillonnees d'une sequence -> (NB_FENETRES, NB_FRAMES_ECHANTILLON,
    NUM_GROUP, GROUP_SIZE, 3) groupes + centres correspondants, en numpy
    (pas de tenseurs CUDA dans les sous-processus)."""
    env, sujet, action = args
    frames = mmfi.charger_sequence_lidar(env, sujet, action)
    nb_total = len(frames)
    if nb_total < 5:
        return None
    longueur_fenetre = min(40, nb_total)
    voxels = _voxels_statiques(frames)

    debut_max = nb_total - longueur_fenetre
    debuts = np.linspace(0, max(debut_max, 0), NB_FENETRES).astype(int)

    groupes_tout, centres_tout = [], []
    for debut in debuts:
        indices_frames = np.linspace(debut, debut + longueur_fenetre - 1, NB_FRAMES_ECHANTILLON).astype(int)
        for i in indices_frames:
            personne = _isoler_personne(frames[i], voxels)
            if personne.shape[0] == 0:
                personne = np.zeros((1, 3), dtype=np.float32)
            g, c = tok.tokeniser_frame_points(personne, NUM_GROUP, GROUP_SIZE)
            groupes_tout.append(g.numpy())
            centres_tout.append(c.numpy())

    groupes_tout = np.stack(groupes_tout).reshape(NB_FENETRES, NB_FRAMES_ECHANTILLON, NUM_GROUP, GROUP_SIZE, 3)
    centres_tout = np.stack(centres_tout).reshape(NB_FENETRES, NB_FRAMES_ECHANTILLON, NUM_GROUP, 3)
    label = INDEX_ACTION[action]
    return groupes_tout, centres_tout, label, env, sujet


if __name__ == "__main__":
    sequences = mmfi.lister_sequences()
    print(f"sequences totales: {len(sequences)}  nb_fenetres={NB_FENETRES}  "
          f"nb_frames_echantillon={NB_FRAMES_ECHANTILLON}", flush=True)

    nb_procs = max(1, (os.cpu_count() or 4) - 2)
    print(f"extraction CPU (isolation+FPS/kNN) parallele sur {nb_procs} processus...", flush=True)

    t0 = time.time()
    resultats = []
    with Pool(processes=nb_procs) as pool:
        for i, res in enumerate(pool.imap(_extraire_groupes_sequence, sequences, chunksize=2)):
            if res is not None:
                resultats.append(res)
            if (i + 1) % 100 == 0:
                duree = time.time() - t0
                restant = duree / (i + 1) * (len(sequences) - i - 1)
                print(f"  {i + 1}/{len(sequences)} sequences  [{duree:.1f}s ecoulees, "
                      f"~{restant:.0f}s restantes]", flush=True)
    print(f"extraction CPU terminee en {time.time()-t0:.1f}s -- {len(resultats)} sequences valides", flush=True)

    print("chargement encodeur Point-MAE (ShapeNet pretrain, fige)...", flush=True)
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    encodeur = EncodeurPointMAE().to(appareil)
    charger_poids_pretrain(encodeur, CHECKPOINT_PRETRAIN, appareil)
    encodeur.eval()

    print("forward GPU (encodage + pooling mean+max par frame)...", flush=True)
    t0 = time.time()
    toutes_features, labels, envs, sujets = [], [], [], []
    with torch.no_grad():
        for groupes_tout, centres_tout, label, env, sujet in resultats:
            g = torch.from_numpy(groupes_tout).float().to(appareil)  # (F,T,G,K,3)
            c = torch.from_numpy(centres_tout).float().to(appareil)  # (F,T,G,3)
            F, T, G, K, _ = g.shape
            tokens = encodeur(g.reshape(F * T, G, K, 3), c.reshape(F * T, G, 3))  # (F*T,G,384)
            pool = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)  # (F*T,768)
            toutes_features.append(pool.reshape(F, T, -1).cpu().numpy())
            labels.extend([label] * F)
            envs.extend([env] * F)
            sujets.extend([sujet] * F)
    print(f"forward GPU termine en {time.time()-t0:.1f}s", flush=True)

    toutes_features = np.concatenate(toutes_features, axis=0)  # (N_fenetres, T, 768)
    labels = np.array(labels)
    envs = np.array(envs)
    sujets = np.array(sujets)
    print(f"features: {toutes_features.shape}  labels: {labels.shape}")

    np.savez("mmfi-src/features_pointmae.npz",
             X=toutes_features, y=labels, envs=envs, sujets=sujets)
    print("features sauvegardees -> mmfi-src/features_pointmae.npz")
    print("EXTRACTION_POINTMAE_OK")
