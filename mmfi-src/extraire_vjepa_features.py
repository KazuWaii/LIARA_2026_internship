"""Autre "add-in" JEPA demande explicitement par l'utilisateur : "get a vit
trained... downloading a pre-trained one on the web related to our project".
Contrairement a modele_pointmae_jepa.py (continued pretraining, PARTANT de
Point-MAE), ceci teste un ViT COMPLETEMENT DIFFERENT : le vrai checkpoint
V-JEPA officiel (facebookresearch/jepa, ViT-L/16, vitl16.pth.tar, 5,1 Go,
pre-entraine sur video naturelle -- Kinetics-710/SSv2/HowTo100M), applique
sur la methode GRILLE (pas points) -- la grille (T,4,H,W) a EXACTEMENT la
forme d'un clip video (cf. README, note 4/5 : "la grille empilee dans le
temps a exactement la forme d'un clip video"), donc structurellement le
candidat naturel pour un ViT video pre-entraine.

REUTILISE LE CODE SOURCE REEL de reference_vjepa/ (deja telecharge, verbatim,
PAS de reimplementation par retro-ingenierie ici -- contrairement a
Point-MAE, ce code est pur PyTorch, aucune extension CUDA a compiler) :
vision_transformer.vit_large, avec interpolation automatique de la
position embedding (interpolate_pos_encoding, deja dans leur code) pour
notre grille 4x6 spatiale x 8 temporelle (192 tokens) au lieu de leur
14x14x8=1568 natifs -- pas de modification necessaire.

ADAPTATIONS OBLIGATOIRES (domaine tres different -- video RGB naturelle
vs grille d'occupation LiDAR) :
  - 3 canaux au lieu de 4 : on garde x,y,z (positions reelles, pas RGB),
    on retire le canal de validite -- le conv3D pre-entraine attend
    exactement in_chans=3, poids inchanges.
  - Standardisation par-fenetre (moyenne/ecart-type sur les points VALIDES
    seulement) des 3 canaux avant le patch embedding -- les valeurs xyz
    brutes (metres, echelle ~1-6m) sont d'un ordre de grandeur tres
    different des pixels RGB normalises (~[-2,2]) sur lesquels le conv3D
    pre-entraine a ete calibre.

Meme structure de pipeline que extraire_pointmae.py : extraction CPU
parallele (projection grille par fenetre) PUIS forward GPU par lots
(modele fige, ViT-L 300M parametres -- plus lourd que Point-MAE, lots plus
petits)."""

import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_vjepa"))

import numpy as np
import torch

import mmfi_dataset as mmfi
import tokenisation as tok
from src.models.vision_transformer import vit_large

CHECKPOINT_VJEPA = r"D:\pretrained_checkpoints\vjepa\vitl16.pth.tar"
NB_FENETRES = 4
LONGUEUR_FENETRE = 16  # = num_frames du ViT-L pretrained (T=16, tubelet=2 -> 8 patches temporels)


def _extraire_grilles_sequence(args):
    """CPU seul : projette NB_FENETRES fenetres d'une sequence -> grilles
    (T,3,H,W) (x,y,z seulement, validite retiree). Retourne liste de
    (grille, label, env, sujet)."""
    env, sujet, action = args
    frames = mmfi.charger_sequence_lidar(env, sujet, action)
    nb_total = len(frames)
    if nb_total < LONGUEUR_FENETRE:
        return None
    debut_max = nb_total - LONGUEUR_FENETRE
    debuts = np.linspace(0, max(debut_max, 0), NB_FENETRES).astype(int)

    resultats = []
    for debut in debuts:
        fenetre = frames[debut:debut + LONGUEUR_FENETRE]
        grille = tok.empiler_fenetre_temporelle(fenetre, profil=tok.PROFIL_MMFI)  # (T,4,H,W)
        grille_xyz = grille[:, :3]  # (T,3,H,W) -- retire le canal de validite
        resultats.append((grille_xyz.astype(np.float32), action, env, sujet))
    return resultats


def _standardiser(grille_xyz):
    """grille_xyz: (B,3,T,H,W) -- standardise chaque canal (moyenne/ecart-type
    sur les valeurs NON NULLES, une cellule vide = (0,0,0) par construction
    de projeter_multicanaux -- ne doit pas biaiser la normalisation)."""
    masque_valide = (grille_xyz != 0).any(dim=1, keepdim=True)  # (B,1,T,H,W)
    n_valide = masque_valide.sum(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1)
    moyenne = (grille_xyz * masque_valide).sum(dim=(2, 3, 4), keepdim=True) / n_valide
    ecart = torch.sqrt(((grille_xyz - moyenne) ** 2 * masque_valide).sum(dim=(2, 3, 4), keepdim=True) / n_valide)
    return (grille_xyz - moyenne) / (ecart + 1e-4)


if __name__ == "__main__":
    sequences = mmfi.lister_sequences()
    print(f"sequences totales: {len(sequences)}  nb_fenetres={NB_FENETRES}  longueur_fenetre={LONGUEUR_FENETRE}",
          flush=True)

    nb_procs = max(1, (os.cpu_count() or 4) - 2)
    print(f"extraction CPU (projection grille) parallele sur {nb_procs} processus...", flush=True)
    t0 = time.time()
    CODES_ACTIONS = sorted(mmfi.ACTIONS.keys())
    INDEX_ACTION = {code: i for i, code in enumerate(CODES_ACTIONS)}

    grilles, labels, envs, sujets = [], [], [], []
    with Pool(processes=nb_procs) as pool:
        for i, res in enumerate(pool.imap(_extraire_grilles_sequence, sequences, chunksize=2)):
            if res is not None:
                for grille, action, env, sujet in res:
                    grilles.append(grille)
                    labels.append(INDEX_ACTION[action])
                    envs.append(env)
                    sujets.append(sujet)
            if (i + 1) % 100 == 0:
                duree = time.time() - t0
                restant = duree / (i + 1) * (len(sequences) - i - 1)
                print(f"  {i + 1}/{len(sequences)} sequences  [{duree:.1f}s ecoulees, "
                      f"~{restant:.0f}s restantes]", flush=True)
    print(f"extraction CPU terminee en {time.time()-t0:.1f}s -- {len(grilles)} fenetres", flush=True)

    labels, envs, sujets = np.array(labels), np.array(envs), np.array(sujets)

    print("chargement V-JEPA ViT-L/16 (checkpoint officiel, video naturelle)...", flush=True)
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    modele = vit_large(img_size=224, patch_size=16, num_frames=LONGUEUR_FENETRE, tubelet_size=2,
                        in_chans=3, uniform_power=True).to(appareil)

    ckpt = torch.load(CHECKPOINT_VJEPA, map_location=appareil, weights_only=False)
    print("cles checkpoint:", list(ckpt.keys()))
    # target_encoder = copie EMA (moyenne mobile, stop-gradient) -- choix standard
    # pour l'extraction de features en aval dans la litterature JEPA, cf. README
    # (mettre_a_jour_cible/encodeur_cible, meme convention que ce projet).
    sd = ckpt["target_encoder"]
    prefixe = "module.backbone."
    sd = {k[len(prefixe):]: v for k, v in sd.items() if k.startswith(prefixe)}
    manquantes, inattendues = modele.load_state_dict(sd, strict=False)
    print(f"chargement: {len(manquantes)} manquantes, {len(inattendues)} inattendues")
    if manquantes:
        print("  manquantes (extrait):", manquantes[:10])
    if inattendues:
        print("  inattendues (extrait):", inattendues[:10])
    modele.eval()

    print("forward GPU (encodage + pooling mean+max)...", flush=True)
    t0 = time.time()
    toutes_features = []
    taille_lot = 8  # ViT-L (300M) -- lots plus petits que Point-MAE (12M)
    N = len(grilles)
    with torch.no_grad():
        for debut in range(0, N, taille_lot):
            lot = grilles[debut:debut + taille_lot]
            g = torch.from_numpy(np.stack(lot)).to(appareil)  # (B,T,3,H,W)
            g = g.permute(0, 2, 1, 3, 4).contiguous()  # (B,3,T,H,W) -- attendu par PatchEmbed3D
            g = _standardiser(g)
            tokens = modele(g)  # (B, 192, 1024)
            pool = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)  # (B,2048)
            toutes_features.append(pool.cpu().numpy())
            if (debut // taille_lot) % 20 == 0:
                print(f"  {debut}/{N}  [{time.time()-t0:.0f}s]", flush=True)
    print(f"forward GPU termine en {time.time()-t0:.1f}s")

    toutes_features = np.concatenate(toutes_features, axis=0)
    print(f"features: {toutes_features.shape}")

    np.savez("mmfi-src/features_vjepa.npz",
             X=toutes_features, y=labels, envs=envs, sujets=sujets)
    print("features sauvegardees -> mmfi-src/features_vjepa.npz")
    print("EXTRACTION_VJEPA_OK")
