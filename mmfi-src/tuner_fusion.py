"""Autre "add-in" demande (broader comparison, PAS du JEPA) : fusionne les
features du spectre Laplacien (classique, 0,659/0,659 seul) et les features
Point-MAE (ShapeNet, 0,949 seul) -- teste si les deux sont COMPLEMENTAIRES
(le spectre Laplacien capture la geometrie du squelette/pose via les
valeurs propres du graphe de proximite ; Point-MAE capture des features
apprises generiques sur ~50k formes 3D -- deux points de vue differents sur
le meme nuage de points isole).

Les deux caches (features_laplacien_v4_all.npz : 8 fenetres/sequence ;
features_pointmae.npz : 4 fenetres/sequence) n'ont PAS le meme nombre de
fenetres par sequence -- fusion au niveau de la SEQUENCE (pas de la
fenetre) : moyenne des fenetres de chaque sequence pour les deux jeux de
features (cle unique (env,sujet,y) -- action encodee dans y, cf. mmfi_dataset.ACTIONS),
puis concatenation des deux vecteurs moyennes."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from dataset_jepa import decouper_train_val
from tuner_pointmae_classique import _pool_temporel


def _moyenne_par_sequence(X, y, envs, sujets):
    """Regroupe les lignes par sequence unique (env,sujet,y) et moyenne les
    fenetres de chaque sequence. Retourne (X_seq, y_seq, envs_seq, sujets_seq)
    -- UNE ligne par sequence, ordre deterministe (trie par cle)."""
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


def _masque_seq(sequences, envs_seq, sujets_seq):
    paires_voulues = {(env, sujet) for env, sujet, _action in sequences}
    m = np.zeros(len(envs_seq), dtype=bool)
    for env, sujet in paires_voulues:
        m |= (envs_seq == env) & (sujets_seq == sujet)
    return m


if __name__ == "__main__":
    dlap = np.load("mmfi-src/features_laplacien_v4_all.npz")
    dpm = np.load("mmfi-src/features_pointmae.npz")

    Xlap_seq, ylap_seq, elap_seq, slap_seq = _moyenne_par_sequence(
        dlap["X"], dlap["y"], dlap["envs"], dlap["sujets"])
    Xpm_flat = _pool_temporel(dpm["X"])
    Xpm_seq, ypm_seq, epm_seq, spm_seq = _moyenne_par_sequence(
        Xpm_flat, dpm["y"], dpm["envs"], dpm["sujets"])

    print(f"sequences Laplacien: {Xlap_seq.shape}  sequences Point-MAE: {Xpm_seq.shape}")
    assert np.array_equal(ylap_seq, ypm_seq) and np.array_equal(elap_seq, epm_seq) and np.array_equal(slap_seq, spm_seq), \
        "les deux jeux de features doivent couvrir exactement les memes sequences, dans le meme ordre"

    X_fusion = np.concatenate([Xlap_seq, Xpm_seq], axis=1)
    print(f"X_fusion: {X_fusion.shape}")

    seq_train, seq_val = decouper_train_val()
    m_train, m_val = _masque_seq(seq_train, elap_seq, slap_seq), _masque_seq(seq_val, elap_seq, slap_seq)
    X_train, X_val = X_fusion[m_train], X_fusion[m_val]
    y_train, y_val = ylap_seq[m_train], ylap_seq[m_val]
    groupes_train = slap_seq[m_train]
    print(f"train: {X_train.shape}  val: {X_val.shape}")

    scaler = StandardScaler().fit(X_train)
    Xtr, Xva = scaler.transform(X_train), scaler.transform(X_val)

    print("\n=== SVM (C=3.0, gamma=scale, meme reglage que Point-MAE seul) ===")
    svm = SVC(kernel="rbf", C=3.0, gamma="scale").fit(Xtr, y_train)
    preds = svm.predict(Xva)
    print(f"   accuracy={(preds==y_val).mean():.3f} f1_macro={f1_score(y_val,preds,average='macro'):.3f}")

    print("\n=== SVM regle sur la fusion (GridSearchCV, GroupKFold sujet) ===")
    cv = GroupKFold(n_splits=5)
    pipe = Pipeline([("scale", StandardScaler()), ("clf", SVC())])
    grille = {"clf__kernel": ["rbf"], "clf__C": [0.1, 1.0, 3.0, 10.0, 30.0],
              "clf__gamma": ["scale", "auto", 0.001, 0.01]}
    gs = GridSearchCV(pipe, grille, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train, groups=groupes_train)
    preds_t = gs.predict(X_val)
    acc_t = (preds_t == y_val).mean()
    print(f"   meilleurs params={gs.best_params_} cv_acc={gs.best_score_:.3f} "
          f"VAL accuracy={acc_t:.3f} f1_macro={f1_score(y_val,preds_t,average='macro'):.3f}")

    print("\n\n=== RESUME ===")
    print("Laplacien seul (0,659) | Point-MAE seul (0,949) | Fusion (ci-dessus)")
    print("TUNER_FUSION_OK")
