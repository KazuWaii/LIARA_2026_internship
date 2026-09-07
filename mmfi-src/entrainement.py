"""Boucle d'entrainement JEPA sur MM-Fi lidar -- une seule fonction
parametree par `methode` ("grille" ou "points") plutot que deux scripts
dupliques, pour garantir que les deux methodes tournent avec exactement les
memes hyperparametres d'entrainement (nb_etapes, taille_lot, taux
d'apprentissage, decroissance EMA, ecretage de gradient, graine) -- condition
necessaire pour que leur comparaison soit pertinente (cf. discussion du
31 aout). Cf. mmfi-src/README.md pour la documentation complete du pipeline.

Grille reecrite le 3 septembre pour mirroirer les scheduleurs de V-JEPA
(reference_vjepa/app/vjepa/train.py, init_opt + momentum_scheduler) :
warmup lineaire puis decroissance cosinus du taux d'apprentissage, poids de
regularisation (weight decay) croissant lineairement, momentum EMA croissant
lineairement, ecretage de gradient actif seulement apres le warmup -- tout
etait fixe/constant auparavant. Formes identiques a V-JEPA, ampleurs reduites
a notre echelle (nb_etapes plutot que ipe*epochs*ipe_scale)."""

import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Python "embeddable" : sys.path est controle par un fichier ._pth qui
# n'ajoute pas automatiquement le dossier du script.

import torch
from torch.utils.data import DataLoader, ConcatDataset

import tokenisation as tok
import jrdb_dataset as jrdb
from dataset_jepa import (
    MMFiDatasetJEPA, MMFiDatasetJEPAPoints, decouper_train_val,
    OnceDatasetJEPA, decouper_train_val_once,
    JrdbDatasetJEPA, decouper_train_val_jrdb,
)
from modele import ModeleJEPA, ModeleJEPAPoints
from perte import perte_jepa, perte_jepa_points


def _pas_forward(modele, lot, methode, appareil, poids_reg=None):
    """Une passe avant + perte, pour l'une ou l'autre methode. Retourne
    (perte, colonnes_csv (liste), texte_log, contexte_encode) -- contexte_encode
    est deja calcule pour la perte, le renvoyer evite un second forward pass
    quand diagnostic_effondrement_poolee en a besoin (cf. boucle d'entrainement).

    poids_reg: dict optionnel (poids_variance, seuil_variance,
    poids_uniformite_poolee, temperature_uniformite) -- absent = defauts de
    perte_jepa (0.0, mirroir de reg_coeff: 0.0, cf. perte.py). Sert a
    reactiver la regularisation prouvee (3 septembre : le mirroir V-JEPA pur
    ne suffit pas a eviter l'effondrement sur notre donnee, cf. README)."""
    poids_reg = poids_reg or {}
    if methode == "points":
        groupes, centres = lot[0].to(appareil), lot[1].to(appareil)
        predictions, cibles, contexte_encode = modele(groupes, centres)
        perte, perte_pred, perte_reg = perte_jepa_points(predictions, cibles, contexte_encode)
        return perte, [perte_pred, perte_reg], f"prediction={perte_pred:.4f} reg={perte_reg:.4f}", contexte_encode
    else:
        lot = lot.to(appareil)
        predictions, cibles, contexte_encode = modele(lot)
        perte, perte_jepa_val, perte_reg = perte_jepa(predictions, cibles, contexte_encode, **poids_reg)
        return perte, [perte_jepa_val, perte_reg], f"jepa={perte_jepa_val:.4f} reg={perte_reg:.4f}", contexte_encode


@torch.no_grad()
def diagnostic_effondrement(modele, lot, methode, appareil):
    """Similarite cosinus moyenne entre tokens cible independants -- proche de
    1 = effondrement, cf. discussion du 31-08 (0.94 avant regularisation de
    variance, 0.75 apres sur le smoke test grille a 200 pas)."""
    if methode == "points":
        patches = modele.tokenizer(lot[0].to(appareil), lot[1].to(appareil))
    else:
        patches = modele.tokenizer(lot.to(appareil))
    cible = modele.encodeur_cible(patches).reshape(-1, modele.tokenizer.dim)
    a = cible[torch.randperm(cible.shape[0])[:500]]
    b = cible[torch.randperm(cible.shape[0])[:500]]
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


@torch.no_grad()
def diagnostic_effondrement_poolee(contexte_encode):
    """Similarite cosinus moyenne entre features POOLEES (mean+max sur les
    patches, comme pooler_mean_max dans evaluation.py) d'echantillons
    DIFFERENTS du lot -- la quantite qui s'est reveleee effondree en linear
    probing (~0.999) le 2 septembre malgre un effondrement par-patch
    (diagnostic_effondrement ci-dessus) raisonnable. Ajoutee le 3 septembre en
    meme temps que perte_uniformite_poolee (perte.py) pour suivre cette
    quantite EN DIRECT pendant l'entrainement plutot que de decouvrir
    l'effondrement seulement apres coup en linear probing. Reste utile meme
    perte_uniformite_poolee desactivee par defaut (cf. perte.py) : c'est le
    signal qui dirait si le mirroir V-JEPA (EMA+stop-grad+masquage seuls,
    sans regularisation explicite) suffit sur notre donnee, comme chez eux.

    contexte_encode : un seul tenseur (B,N,dim) (methode points, inchangee),
    OU une LISTE de tenseurs (B,K_i,dim) -- un par groupe de masque (methode
    grille depuis le 3 septembre, K differe d'un groupe a l'autre). Le
    pooling (mean+max sur dim=1) fait disparaitre la dependance a K, les
    groupes redeviennent concatenables sur le batch apres coup."""
    groupes = contexte_encode if isinstance(contexte_encode, list) else [contexte_encode]
    pool = torch.cat([torch.cat([g.mean(dim=1), g.max(dim=1).values], dim=-1) for g in groupes], dim=0)
    if pool.shape[0] < 2:
        return float("nan")
    pool = torch.nn.functional.normalize(pool, dim=-1)
    sims = pool @ pool.T
    masque = ~torch.eye(pool.shape[0], dtype=torch.bool, device=pool.device)
    return sims[masque].mean().item()


@torch.no_grad()
def evaluer(modele, chargeur_val, methode, appareil, nb_lots=5, poids_reg=None):
    modele.eval()
    pertes = []
    for i, lot in enumerate(chargeur_val):
        if i >= nb_lots:
            break
        perte, _, _, _ = _pas_forward(modele, lot, methode, appareil, poids_reg)
        pertes.append(perte.item())
    modele.train()
    return sum(pertes) / len(pertes) if pertes else float("nan")


def _lr_a_l_etape(etape, nb_etapes, nb_etapes_warmup, lr_debut, lr_pic, lr_final):
    """Warmup lineaire (lr_debut -> lr_pic) puis decroissance cosinus
    (lr_pic -> lr_final) -- mirroir de init_opt (WarmupCosineSchedule),
    reference_vjepa/app/vjepa/train.py."""
    if etape < nb_etapes_warmup:
        return lr_debut + (lr_pic - lr_debut) * etape / max(1, nb_etapes_warmup)
    progression = (etape - nb_etapes_warmup) / max(1, nb_etapes - nb_etapes_warmup)
    return lr_final + 0.5 * (lr_pic - lr_final) * (1 + math.cos(math.pi * min(progression, 1.0)))


def _wd_a_l_etape(etape, nb_etapes, wd_debut, wd_final):
    """Poids de regularisation croissant lineairement -- mirroir de
    CosineWDSchedule -> simplifie en lineaire, meme principe (wd_debut a
    wd_final sur l'entrainement complet)."""
    return wd_debut + (wd_final - wd_debut) * etape / max(1, nb_etapes)


def _ema_a_l_etape(etape, nb_etapes, ema_debut, ema_fin):
    """Momentum EMA croissant lineairement -- mirroir de momentum_scheduler,
    reference_vjepa/app/vjepa/train.py (generateur -> forme close ici, pas
    besoin de rejouer l'etat pour une reprise de checkpoint qu'on ne fait pas)."""
    return ema_debut + (ema_fin - ema_debut) * etape / max(1, nb_etapes)


def entrainer(
    source="mmfi", methode="grille", nb_etapes=3000, taille_lot=8,
    taux_apprentissage=1e-4, lr_debut=1e-5, lr_final=1e-6, fraction_warmup=0.1,
    wd_debut=0.01, wd_final=0.1, clip_grad=1.0,
    decroissance_ema=0.996, decroissance_ema_fin=0.9999,
    intervalle_log=25, intervalle_eval=200,
    intervalle_checkpoint=500, racine=None, graine=0, suffixe=None,
    poids_variance=0.0, seuil_variance=0.2,
    poids_uniformite_poolee=0.0, temperature_uniformite=2.0,
):
    assert source in ("mmfi", "once", "jrdb", "mmfi_jrdb", "mmfi_jrdb2x")
    assert methode in ("grille", "points")
    if source in ("once", "jrdb", "mmfi_jrdb", "mmfi_jrdb2x"):
        assert methode == "grille", f"methode points pas encore calibree pour {source} (BORNES/echelles specifiques a MM-Fi)"
    torch.manual_seed(graine)  # reproductibilite + comparaison controlee entre methodes

    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    print("appareil:", appareil, "-- source:", source, "-- methode:", methode)

    kwargs_racine = {"racine": racine} if racine else {}
    if source in ("mmfi_jrdb", "mmfi_jrdb2x"):
        # Pre-entrainement combine (3 septembre soir) : JRDB projete via
        # PROFIL_MMFI (meme forme de grille que MM-Fi, verifie mecaniquement
        # -- cf. README "Dataset 3 -- JRDB") pour pouvoir melanger les deux
        # sources dans un seul ConcatDataset/DataLoader.
        #
        # mmfi_jrdb2x (4 septembre matin) : la comparaison multi-graines a
        # montre MM-Fi+JRDB (25 sequences JRDB, <3% du corpus MM-Fi)
        # statistiquement indiscernable de from-scratch -- hypothese : volume
        # JRDB trop marginal. Ce mode double le volume JRDB en reutilisant le
        # split "test" de JRDB (D:/jrdb/test/, deja extrait, jamais utilise
        # pour le pre-entrainement) -- legitime car aucune etiquette JRDB
        # n'est utilisee nulle part dans ce pipeline (JEPA auto-supervise,
        # seul MM-Fi fournit les etiquettes HAR pour le fine-tuning aval).
        sequences_train_mmfi, sequences_val_mmfi = decouper_train_val()
        sequences_train_jrdb, sequences_val_jrdb = decouper_train_val_jrdb()
        print(f"sequences MM-Fi train: {len(sequences_train_mmfi)}  val: {len(sequences_val_mmfi)}")
        print(f"sequences JRDB (train split) train: {len(sequences_train_jrdb)}  val: {len(sequences_val_jrdb)}")
        datasets_train = [
            MMFiDatasetJEPA(sequences_train_mmfi),
            JrdbDatasetJEPA(sequences_train_jrdb, profil=tok.PROFIL_MMFI),
        ]
        datasets_val = [
            MMFiDatasetJEPA(sequences_val_mmfi),
            JrdbDatasetJEPA(sequences_val_jrdb, profil=tok.PROFIL_MMFI),
        ]
        if source == "mmfi_jrdb2x":
            racine_jrdb_test = jrdb.RACINE_PAR_DEFAUT.parent.parent.parent / "test" / "pointclouds" / "lower_velodyne"
            sequences_train_jrdb2, sequences_val_jrdb2 = decouper_train_val_jrdb(racine=racine_jrdb_test)
            print(f"sequences JRDB (test split, reutilise sans etiquette) train: {len(sequences_train_jrdb2)}  val: {len(sequences_val_jrdb2)}")
            datasets_train.append(JrdbDatasetJEPA(sequences_train_jrdb2, racine=racine_jrdb_test, profil=tok.PROFIL_MMFI))
            datasets_val.append(JrdbDatasetJEPA(sequences_val_jrdb2, racine=racine_jrdb_test, profil=tok.PROFIL_MMFI))
        dataset_train = ConcatDataset(datasets_train)
        dataset_val = ConcatDataset(datasets_val)
        print(f"dataset combine -- train: {len(dataset_train)}  val: {len(dataset_val)}")
        kwargs_modele = {}
    else:
        if source == "once":
            sequences_train, sequences_val = decouper_train_val_once(**kwargs_racine)
        elif source == "jrdb":
            sequences_train, sequences_val = decouper_train_val_jrdb(**kwargs_racine)
        else:
            sequences_train, sequences_val = decouper_train_val(**kwargs_racine)
        print(f"sequences train: {len(sequences_train)}  val: {len(sequences_val)}")

        if source == "once":
            DatasetClasse = OnceDatasetJEPA
            kwargs_modele = dict(hauteur=40, largeur=180, noyau_spatial=10)
        elif source == "jrdb":
            DatasetClasse = JrdbDatasetJEPA
            kwargs_modele = dict(hauteur=16, largeur=180, noyau_spatial=4)
        else:
            DatasetClasse = MMFiDatasetJEPAPoints if methode == "points" else MMFiDatasetJEPA
            kwargs_modele = {}
        dataset_train = DatasetClasse(sequences_train, **kwargs_racine)
        dataset_val = DatasetClasse(sequences_val, **kwargs_racine)

    ModeleClasse = ModeleJEPAPoints if methode == "points" else ModeleJEPA
    colonnes_perte = ["perte_prediction", "perte_reg"] if methode == "points" else ["perte_jepa", "perte_reg"]

    chargeur_train = DataLoader(dataset_train, batch_size=taille_lot, shuffle=True, num_workers=0)
    chargeur_val = DataLoader(dataset_val, batch_size=taille_lot, shuffle=True, num_workers=0)

    modele = ModeleClasse(**kwargs_modele).to(appareil)
    # modele.masquage inclus explicitement : sans ca, le(s) token(s) masque(s)
    # appris ne sont jamais mis a jour par l'optimiseur (bug trouve et
    # corrige le 31 aout, present sur les runs precedents des deux methodes).
    parametres_encodeur = (
        list(modele.tokenizer.parameters())
        + list(modele.masquage.parameters())
        + list(modele.encodeur_contexte.parameters())
    )
    parametres_predicteur = list(modele.predicteur.parameters())
    optimiseur = torch.optim.AdamW(parametres_encodeur + parametres_predicteur, lr=taux_apprentissage)

    nb_etapes_warmup = max(1, int(fraction_warmup * nb_etapes))
    poids_reg = dict(
        poids_variance=poids_variance, seuil_variance=seuil_variance,
        poids_uniformite_poolee=poids_uniformite_poolee, temperature_uniformite=temperature_uniformite,
    )

    suffixe = suffixe or (methode if source == "mmfi" else f"{source}_{methode}")
    dossier_ckpt = f"mmfi-src/checkpoints_{suffixe}"
    os.makedirs(dossier_ckpt, exist_ok=True)
    fichier_historique = open(f"mmfi-src/historique_entrainement_{suffixe}.csv", "w", newline="")
    ecrivain = csv.writer(fichier_historique)
    ecrivain.writerow(["etape", "perte_train", *colonnes_perte, "perte_val",
                        "similarite_effondrement", "similarite_effondrement_poolee", "lr", "wd", "ema"])

    modele.train()
    etape = 0
    t0 = time.time()

    while etape < nb_etapes:
        for lot in chargeur_train:
            if etape >= nb_etapes:
                break

            lr = _lr_a_l_etape(etape, nb_etapes, nb_etapes_warmup, lr_debut, taux_apprentissage, lr_final)
            wd = _wd_a_l_etape(etape, nb_etapes, wd_debut, wd_final)
            ema = _ema_a_l_etape(etape, nb_etapes, decroissance_ema, decroissance_ema_fin)
            for groupe in optimiseur.param_groups:
                groupe["lr"] = lr
                groupe["weight_decay"] = wd

            perte, details, texte_details, contexte_encode = _pas_forward(modele, lot, methode, appareil, poids_reg)

            optimiseur.zero_grad()
            perte.backward()
            # Ecretage de gradient actif seulement apres le warmup (mirroir
            # de train_step, reference_vjepa/app/vjepa/train.py) -- les tout
            # premiers pas ont des gradients naturellement plus bruites/larges
            # pendant que le taux d'apprentissage monte, ecreter des ce moment
            # genererait plus de bruit que d'aide.
            if etape >= nb_etapes_warmup:
                torch.nn.utils.clip_grad_norm_(parametres_encodeur, clip_grad)
                torch.nn.utils.clip_grad_norm_(parametres_predicteur, clip_grad)
            optimiseur.step()
            modele.mettre_a_jour_cible(ema)

            perte_val, similarite, similarite_poolee = "", "", ""
            if etape % intervalle_eval == 0:
                perte_val = evaluer(modele, chargeur_val, methode, appareil, poids_reg=poids_reg)
                similarite = diagnostic_effondrement(modele, lot, methode, appareil)
                similarite_poolee = diagnostic_effondrement_poolee(contexte_encode)

            ecrivain.writerow([etape, perte.item(), *details, perte_val, similarite, similarite_poolee, lr, wd, ema])
            fichier_historique.flush()

            if etape % intervalle_log == 0:
                duree = time.time() - t0
                extra = (f"  val={perte_val:.4f} sim_effondrement={similarite:.4f} sim_poolee={similarite_poolee:.4f}"
                          if perte_val != "" else "")
                print(f"etape {etape:4d}/{nb_etapes}  perte={perte.item():.4f} ({texte_details}){extra}"
                      f"  lr={lr:.2e} ema={ema:.5f}  [{duree:.1f}s]")

            if etape > 0 and etape % intervalle_checkpoint == 0:
                torch.save(modele.state_dict(), f"{dossier_ckpt}/etape_{etape}.pt")

            etape += 1

    fichier_historique.close()
    torch.save(modele.state_dict(), f"{dossier_ckpt}/final.pt")
    return modele


if __name__ == "__main__":
    # usage: python entrainement.py [grille|points] [mmfi|once]
    methode = sys.argv[1] if len(sys.argv) > 1 else "grille"
    source = sys.argv[2] if len(sys.argv) > 2 else "mmfi"
    entrainer(methode=methode, source=source)
    suffixe = methode if source == "mmfi" else f"{source}_{methode}"
    print(f"\nentrainement termine -- poids -> mmfi-src/checkpoints_{suffixe}/final.pt")
    print(f"historique complet -> mmfi-src/historique_entrainement_{suffixe}.csv")
