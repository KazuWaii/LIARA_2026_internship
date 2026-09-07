"""Perte JEPA (grille) reecrite le 3 septembre pour mirroirer fidelement
V-JEPA (reference_vjepa/app/vjepa/train.py, loss_fn) : L1 (erreur absolue
moyenne) dans l'espace de representation, PAS de similarite cosinus, PAS de
scission vide/occupe -- ces deux mecanismes etaient des adaptations
specifiques a AD-L-JEPA/notre grille eparse, retirees en meme temps que le
token "vide" (cf. tokenisation.py) car V-JEPA n'en a pas besoin (video dense,
pas de notion de cellule vide).

Trouvaille cle qui motive cette reecriture (3 septembre) : le vitl16.yaml de
V-JEPA a `reg_coeff: 0.0` -- leur regularisateur de variance (equivalent a
perte_variance ci-dessous) est calcule mais NE CONTRIBUE AUCUN GRADIENT en
production. V-JEPA compte entierement sur EMA + stop-gradient + son masquage
multi-blocs a deux echelles pour eviter l'effondrement, sans terme de
regularisation explicite. `perte_variance`/`perte_uniformite_poolee` restent
definies ci-dessous (testees, fonctionnelles -- cf. historique du 2-3
septembre : la seconde a resolu un effondrement poole a 0.9933 -> ~0.0-0.1)
mais ne sont PAS branchees par defaut dans perte_jepa, pour mirroirer
`reg_coeff: 0.0` fidelement. Reactivation = changer poids_variance/
poids_uniformite_poolee a une valeur non nulle, pas une reecriture."""

import torch
from torch import nn


def perte_variance(embeddings, seuil=0.2, eps=1e-4):
    """Penalise les dimensions dont l'ecart-type (sur le lot aplati) tombe
    sous `seuil` -- empeche l'effondrement dimensionnel. embeddings: (B,N,dim).

    Retourne 0 si moins de 2 lignes (variance non definie) -- garde-fou en
    plus de la garantie de contexte non vide dans masquage.py, pour ne jamais
    laisser un NaN corrompre les gradients."""
    plat = embeddings.reshape(-1, embeddings.shape[-1])
    if plat.shape[0] < 2:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    ecart_type = torch.sqrt(plat.var(dim=0) + eps)
    return torch.relu(seuil - ecart_type).mean()


def perte_uniformite_poolee(embeddings, temperature=2.0):
    """Remplace perte_variance_poolee (ajoutee le 2 septembre, invalidee le 3)
    : la variance BRUTE (non normalisee) d'une representation poolee peut
    etre satisfaite par une simple variation d'ECHELLE d'un vecteur
    globalement partage -- chaque echantillon pointe dans quasiment la meme
    DIRECTION, seule sa norme varie -- sans que la similarite cosinus (ce
    qu'evalue reellement le linear probing, cf. pooler_mean_max dans
    evaluation.py) ne bouge. Confirme empiriquement le 3 septembre : sur 600
    pas d'entrainement, perte_variance_poolee tombait a ~0 (ecart-type par
    dimension au-dessus du seuil -- "variance suffisante" du point de vue de
    la perte) alors que la similarite cosinus entre echantillons du lot
    restait a 0.9933 (quasi inchangee vs ~0.999 sans aucune regularisation
    poolee). La regularisation n'agissait simplement pas sur la quantite
    mesuree en aval.

    Perte d'uniformite hyperspherique (Wang & Isola 2020, "Understanding
    Contrastive Representation Learning through Alignment and Uniformity on
    the Hypersphere") a la place : normalise chaque representation poolee en
    norme unite (seule la direction compte pour un cosinus), puis penalise
    directement la proximite angulaire moyenne entre paires d'echantillons
    DIFFERENTS du lot -- cible exactement la quantite que mesure le
    diagnostic d'effondrement en aval, plutot qu'un proxy (l'ecart-type brut)
    satisfiable sans que la direction ne change.

    Pool mean+max (comme pooler_mean_max dans evaluation.py, pas seulement la
    moyenne) : la representation evaluee en aval concatene mean ET max sur les
    patches -- la reguler sur la moyenne seule laisserait la moitie du
    vecteur (le max) totalement libre de s'effondrer. embeddings: (B, N, dim)."""
    if embeddings.shape[0] < 2:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    pool = torch.cat([embeddings.mean(dim=1), embeddings.max(dim=1).values], dim=-1)  # (B, 2*dim)
    pool = nn.functional.normalize(pool, dim=-1)
    distances_carrees = torch.cdist(pool, pool).pow(2)  # (B, B)
    masque = ~torch.eye(pool.shape[0], dtype=torch.bool, device=pool.device)
    return torch.log(torch.exp(-temperature * distances_carrees[masque]).mean() + 1e-8)


def perte_uniformite_poolee_liste(liste_embeddings, temperature=2.0):
    """Variante grille de perte_uniformite_poolee (3 septembre) :
    contexte_encode y est une LISTE (un tenseur par groupe de masque, K
    different d'un groupe a l'autre -- cf. modele.py), pas concatenable
    directement contrairement a la methode points. Le pooling (mean+max sur
    dim=1) fait disparaitre la dependance a K -- chaque groupe devient
    (B, 2*dim), memes dimensions quel que soit K -- donc concatenable sur le
    batch APRES pooling, avant le calcul d'uniformite."""
    pool = torch.cat([
        torch.cat([e.mean(dim=1), e.max(dim=1).values], dim=-1)
        for e in liste_embeddings
    ], dim=0)
    if pool.shape[0] < 2:
        return torch.zeros((), device=pool.device, dtype=pool.dtype)
    pool = nn.functional.normalize(pool, dim=-1)
    distances_carrees = torch.cdist(pool, pool).pow(2)
    masque = ~torch.eye(pool.shape[0], dtype=torch.bool, device=pool.device)
    return torch.log(torch.exp(-temperature * distances_carrees[masque]).mean() + 1e-8)


def perte_jepa(
    predictions, cibles, contexte_encode=None,
    poids_variance=0.0, seuil_variance=0.2,
    poids_uniformite_poolee=0.0, temperature_uniformite=2.0,
):
    """predictions, cibles: LISTES de tenseurs (B, |cible_i|, dim), une entree
    par groupe de masque (courte/longue portee, cf. masquage.py -- modele.py
    renvoie une liste, pas un tenseur concatene, expres pour cette perte).

    Mirroir exact de loss_fn(z, h) dans reference_vjepa/app/vjepa/train.py :
    erreur absolue moyenne (L1) PAR GROUPE, puis moyenne NON PONDEREE sur les
    groupes -- pas une moyenne aplatie sur tous les groupes concatenes, qui
    ponderait chaque groupe par sa taille de cible plutot que de les traiter
    a egalite (les deux groupes ont des tailles de cible tres differentes).

    poids_variance/poids_uniformite_poolee = 0.0 par defaut, mirroir de
    `reg_coeff: 0.0` dans vitl16.yaml -- cf. docstring du module. Reactiver
    necessite contexte_encode: LISTE de tenseurs (B, |ctx_i|, dim), meme
    forme que predictions/cibles -- sortie de modele.py, un tenseur par
    groupe (tailles de contexte differentes par construction, pas
    concatenables directement, cf. rassembler dans masquage.py).

    Retourne (perte_totale, perte_jepa_scalaire, perte_reg_scalaire).
    """
    perte = sum(torch.abs(z - h).mean() for z, h in zip(predictions, cibles)) / len(predictions)

    reg = torch.zeros((), device=perte.device, dtype=perte.dtype)
    if poids_variance > 0 or poids_uniformite_poolee > 0:
        assert contexte_encode is not None, "contexte_encode requis si un poids de regularisation est non nul"
        # Chaque groupe a sa propre taille (K differe) -- aplati par groupe
        # avant de concatener (perte_variance aplatit de toute facon en
        # interne, seule la derniere dimension -- le nombre de features --
        # doit correspondre entre groupes, ce qui est toujours le cas ici).
        predictions_plat = torch.cat([p.reshape(-1, p.shape[-1]) for p in predictions], dim=0)
        contexte_plat = torch.cat([c.reshape(-1, c.shape[-1]) for c in contexte_encode], dim=0)
        v_reg = 0.5 * (
            perte_variance(predictions_plat, seuil_variance)
            + perte_variance(contexte_plat, seuil_variance)
        )
        u_reg = perte_uniformite_poolee_liste(contexte_encode, temperature_uniformite)
        reg = poids_variance * v_reg + poids_uniformite_poolee * u_reg

    return perte + reg, perte.item(), reg.item()


def perte_jepa_points(
    predictions, cibles, contexte_encode, poids_variance=1.0, seuil_variance=0.2,
    poids_uniformite_poolee=1.0, temperature_uniformite=2.0,
):
    """Meme perte que perte_jepa, sans le split vide/occupe -- la methode
    point-based (FPS) ne selectionne que des points reels, pas de notion de
    patch structurellement vide (cf. masquage_points.py).

    Retourne (perte_totale, perte_prediction_scalaire, perte_regularisation_scalaire).
    """
    predictions_norm = nn.functional.normalize(predictions, dim=-1)
    cibles_norm = nn.functional.normalize(cibles.detach(), dim=-1)

    perte_prediction = (1.0 - (predictions_norm * cibles_norm).sum(dim=-1)).mean()

    v_reg = 0.5 * (
        perte_variance(predictions, seuil_variance)
        + perte_variance(contexte_encode, seuil_variance)
    )
    u_reg = perte_uniformite_poolee(contexte_encode, temperature_uniformite)

    perte_totale = perte_prediction + poids_variance * v_reg + poids_uniformite_poolee * u_reg
    return perte_totale, perte_prediction.item(), (v_reg + u_reg).item()
