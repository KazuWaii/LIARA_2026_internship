"""Protocole d'evaluation COMMUN a toutes les pipelines de features gelees
du projet (Laplacien, Point-MAE, fusion).

Raison d'etre : jusqu'au 11 septembre, chaque script d'evaluation
reimplementait le meme enchainement (masquage des lignes selon le split,
StandardScaler, SVM/RF/HGB, GridSearchCV GroupKFold sujet) avec de legeres
variations, et `_masque`/`_pool_temporel` etaient importes depuis des
scripts d'entrainement (`entrainer_laplacien_v4`, `tuner_pointmae_classique`)
plutot que depuis un module partage -- ce qui a eu une consequence concrete :
les scripts Point-MAE commites n'evaluaient QUE le split sujet, alors que le
README et les notes de suivi rapportent aussi des chiffres split
environnement (0,908 / 0,926 / 0,937) qu'aucun code du depot ne reproduit.

Ce module regroupe donc :
  - les deux decoupages train/val (deplaces depuis dataset_jepa, qui importe
    torch inutilement pour du sklearn -- dataset_jepa les re-exporte),
  - le masquage des lignes de features par split,
  - les poolings (temporel, par sequence),
  - les classifieurs classiques et leur reglage,
  - `evaluer_deux_splits`, qui fait passer N'IMPORTE QUELLE fonction
    d'evaluation sur LES DEUX splits et resume les deux.

Toute nouvelle pipeline de features doit passer par `evaluer_deux_splits`
pour qu'un resultat split sujet ne puisse plus etre publie sans son
equivalent split environnement.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

import mmfi_dataset as mmfi

# Meilleurs hyperparametres trouves par GridSearchCV (GroupKFold sujet) sur
# les features Laplacien v2 (cf. tuner_laplacien_v2.py / log_tuner_laplacien_v2.txt)
# puis reconfirmes tels quels sur les features Point-MAE
# (log_tuner_pointmae_classique.txt : C=3.0, gamma=scale).
SVM_TUNE_PARAMS = dict(kernel="rbf", C=3.0, gamma="scale")
HGB_TUNE_PARAMS = dict(learning_rate=0.1, max_depth=6, max_iter=400,
                       l2_regularization=0.0, random_state=0)

# Grille de reference du projet (tuner_laplacien_v2.py et
# tuner_pointmae_classique.py utilisaient deja exactement celle-ci ;
# tuner_fusion.py en utilisait un SOUS-ENSEMBLE -- il passe desormais sur la
# grille complete, qui la contient, donc son cv_acc ne peut que monter ou
# rester egal).
GRILLE_SVM = {
    "clf__kernel": ["rbf"],
    "clf__C": [0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
    "clf__gamma": ["scale", "auto", 0.001, 0.003, 0.01, 0.03],
}

NB_FOLDS = 5


# --------------------------------------------------------------------------
# Decoupages train/val
# --------------------------------------------------------------------------

def decouper_train_val(racine=mmfi.RACINE_PAR_DEFAUT, nb_sujets_val=8):
    """Decoupage par sujet (meme principe que le split cross-subject suggere
    par le toolbox MM-Fi officiel : sujets disjoints entre train et val)."""
    sequences = mmfi.lister_sequences(racine)
    sujets = sorted({s for _, s, _ in sequences})
    sujets_val = set(sujets[-nb_sujets_val:])

    train = [s for s in sequences if s[1] not in sujets_val]
    val = [s for s in sequences if s[1] in sujets_val]
    return train, val


def decouper_train_val_environnement(racine=mmfi.RACINE_PAR_DEFAUT, env_val="E04"):
    """Decoupage par ENVIRONNEMENT -- reproduit exactement le protocole du
    papier Laplacien (arXiv:2509.23255) : "30 sujets E1-E3 train / 10 sujets
    E4 val". Verifie (4 septembre) que MM-Fi partitionne deja les 40 sujets
    par environnement (10 sujets/environnement, aucun chevauchement), donc
    ce split par environnement EST un split par sujet, en plus d'etre un
    test de generalisation inter-environnement -- different de
    decouper_train_val() qui retient les 8 derniers sujets (S33-S40, tous
    dans E04) plutot que les 10 sujets complets d'un environnement."""
    sequences = mmfi.lister_sequences(racine)
    train = [s for s in sequences if s[0] != env_val]
    val = [s for s in sequences if s[0] == env_val]
    return train, val


# Les deux splits sur lesquels TOUT resultat du projet doit etre rapporte.
# "sujet" est le split historique de toutes les comparaisons internes ;
# "environnement" est le seul protocole homogene avec la litterature de
# reference (arXiv:2509.23255, 81-90%).
SPLITS = (
    ("sujet", decouper_train_val),
    ("environnement", decouper_train_val_environnement),
)


# --------------------------------------------------------------------------
# Masquage et poolings
# --------------------------------------------------------------------------

def masque_lignes(sequences, envs, sujets):
    """sequences: liste de (env,sujet,action) du split voulu -> masque
    booleen sur les lignes de X/y/envs/sujets dont (env,sujet) correspond a
    une sequence du split (l'action n'est pas necessaire: (env,sujet) suffit
    a distinguer les groupes de lignes ici puisque chaque sequence a un
    (env,sujet,action) unique, mais on ne stocke pas l'action par ligne --
    on filtre donc par (env,sujet) ET on verifie que ca correspond a la
    partition voulue, ce qui est le cas ici car les deux splits utilises
    (sujet et environnement) sont des unions completes de (env,sujet))."""
    paires_voulues = {(env, sujet) for env, sujet, _action in sequences}
    paires_lignes = set(zip(envs.tolist(), sujets.tolist()))
    manquantes = paires_voulues - paires_lignes
    if manquantes:
        print(f"  [!] {len(manquantes)} paires (env,sujet) du split absentes des features "
              f"(sequences trop courtes, ignorees a l'extraction): {list(manquantes)[:5]}")
    m = np.zeros(len(envs), dtype=bool)
    for env, sujet in paires_voulues:
        m |= (envs == env) & (sujets == sujet)
    return m


def pool_temporel(X):
    """X: (N,T,D) -> (N,2D) mean+max sur les T frames (pooling NON APPRIS,
    celui de toutes les references Point-MAE du projet)."""
    return np.concatenate([X.mean(axis=1), X.max(axis=1)], axis=1)


def moyenne_par_sequence(X, y, envs, sujets):
    """Regroupe les lignes par sequence unique (env,sujet,y) et moyenne les
    fenetres de chaque sequence. Retourne (X_seq, y_seq, envs_seq, sujets_seq)
    -- UNE ligne par sequence, ordre deterministe (trie par cle). Sert a
    fusionner deux caches de features qui n'ont pas le meme nombre de
    fenetres par sequence."""
    cles = list(zip(envs.tolist(), sujets.tolist(), y.tolist()))
    index = {}
    for i, cle in enumerate(cles):
        index.setdefault(cle, []).append(i)
    cles_triees = sorted(index.keys())
    X_seq = np.stack([X[index[cle]].mean(axis=0) for cle in cles_triees])
    y_seq = np.array([cle[2] for cle in cles_triees])
    envs_seq = np.array([cle[0] for cle in cles_triees])
    sujets_seq = np.array([cle[1] for cle in cles_triees])
    return X_seq, y_seq, envs_seq, sujets_seq


# --------------------------------------------------------------------------
# Classifieurs classiques
# --------------------------------------------------------------------------

def _scores(preds, y_val):
    return (preds == y_val).mean(), f1_score(y_val, preds, average="macro", zero_division=0)


def regler_svm(X_train, y_train, groupes_train, grille=None, n_splits=NB_FOLDS):
    """GridSearchCV sur le TRAIN SEUL, GroupKFold par sujet (chaque sequence
    produit plusieurs fenetres correlees : un K-Fold aveugle les repartirait
    entre folds et surestimerait la generalisation). Le scaler est DANS le
    pipeline, donc refite a l'interieur de chaque fold -- pas de fuite du
    fold de validation dans la normalisation.

    Retourne l'estimateur ajuste (a utiliser directement pour predire sur la
    validation, qui n'est jamais vue ici)."""
    cv = GroupKFold(n_splits=n_splits)
    pipe = Pipeline([("scale", StandardScaler()), ("clf", SVC())])
    gs = GridSearchCV(pipe, grille or GRILLE_SVM, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train, groups=groupes_train)
    return gs


def evaluer_classifieurs(X_train, y_train, X_val, y_val, groupes_train,
                         regler=True, grille=None):
    """SVM par defaut / RandomForest par defaut / HGB regle / SVM regle.

    `regler=True`  -> le SVM "regle" passe par un GridSearchCV complet
                      (GroupKFold sujet sur le train).
    `regler=False` -> il reutilise directement SVM_TUNE_PARAMS, sans nouvelle
                      recherche (choix d'origine de entrainer_laplacien_v4 :
                      un grid search par split coutait trop cher avec le
                      one-vs-one 27 classes, et on voulait isoler l'effet du
                      split a hyperparametres FIXES).

    La validation ne sert QU'A predire, jamais a selectionner quoi que ce
    soit."""
    print(f"   train: {X_train.shape}  val: {X_val.shape}  "
          f"sujets_train={len(set(groupes_train))}")

    scaler = StandardScaler().fit(X_train)
    Xtr, Xva = scaler.transform(X_train), scaler.transform(X_val)

    acc, f1 = _scores(SVC(kernel="rbf", C=1.0).fit(Xtr, y_train).predict(Xva), y_val)
    print(f"   SVM defaut (C=1.0)        accuracy={acc:.3f} f1_macro={f1:.3f}")

    acc_rf, f1_rf = _scores(
        RandomForestClassifier(random_state=0).fit(Xtr, y_train).predict(Xva), y_val)
    print(f"   RandomForest defaut       accuracy={acc_rf:.3f} f1_macro={f1_rf:.3f}")

    acc_h, f1_h = _scores(
        HistGradientBoostingClassifier(**HGB_TUNE_PARAMS).fit(Xtr, y_train).predict(Xva), y_val)
    print(f"   HGB regle                 accuracy={acc_h:.3f} f1_macro={f1_h:.3f}")

    if regler:
        gs = regler_svm(X_train, y_train, groupes_train, grille=grille)
        acc_t, f1_t = _scores(gs.predict(X_val), y_val)
        params, cv_acc = gs.best_params_, gs.best_score_
        print(f"   SVM regle (GridSearchCV)  accuracy={acc_t:.3f} f1_macro={f1_t:.3f}  "
              f"cv_acc={cv_acc:.3f} params={params}")
    else:
        acc_t, f1_t = _scores(SVC(**SVM_TUNE_PARAMS).fit(Xtr, y_train).predict(Xva), y_val)
        params, cv_acc = dict(SVM_TUNE_PARAMS), None
        print(f"   SVM regle (params fixes)  accuracy={acc_t:.3f} f1_macro={f1_t:.3f}  "
              f"params={params}")

    return dict(svm_defaut=acc, svm_defaut_f1=f1,
                rf_defaut=acc_rf, rf_defaut_f1=f1_rf,
                hgb_tune=acc_h, hgb_tune_f1=f1_h,
                svm_tune=acc_t, svm_tune_f1=f1_t,
                svm_params=params, svm_cv_acc=cv_acc)


# --------------------------------------------------------------------------
# Driver : les deux splits, toujours
# --------------------------------------------------------------------------

def evaluer_deux_splits(X, y, envs, sujets, fn_evaluer, nom_experience=""):
    """Fait passer `fn_evaluer` sur LES DEUX splits du projet.

    fn_evaluer(X_train, y_train, X_val, y_val, groupes_train) -> dict de
    metriques. Volontairement generique : `evaluer_classifieurs` pour les
    pipelines sklearn, mais aussi bien une boucle d'entrainement torch
    multi-graines (cf. entrainer_pointmae_seeds.py).

    Retourne {"sujet": <dict>, "environnement": <dict>}."""
    resultats = {}
    for nom_split, decouper in SPLITS:
        seq_train, seq_val = decouper()
        m_train = masque_lignes(seq_train, envs, sujets)
        m_val = masque_lignes(seq_val, envs, sujets)
        titre = f"SPLIT {nom_split.upper()}"
        if nom_split == "environnement":
            titre += " (E01-E03 train / E04 val, protocole du papier)"
        print(f"\n########## {nom_experience} -- {titre} ##########")
        resultats[nom_split] = fn_evaluer(
            X[m_train], y[m_train], X[m_val], y[m_val], sujets[m_train])
    return resultats


def resumer(nom_experience, resultats, cles=("svm_defaut", "rf_defaut", "hgb_tune", "svm_tune")):
    """Resume compact des deux splits, cote a cote."""
    print(f"\n\n=== RESUME {nom_experience} (les deux splits) ===")
    entete = "split".ljust(16) + "".join(c.ljust(14) for c in cles)
    print(entete)
    for nom_split in ("sujet", "environnement"):
        if nom_split not in resultats:
            continue
        r = resultats[nom_split]
        ligne = nom_split.ljust(16)
        for c in cles:
            v = r.get(c)
            ligne += (f"{v:.3f}" if isinstance(v, float) else str(v)).ljust(14)
        print(ligne)
