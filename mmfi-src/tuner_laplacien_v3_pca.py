"""Reprise de la piste v3 (statistiques de vecteurs propres, 6200 dims,
cf. laplacien_spectral.descripteur_pose_quadrants_vecteurs) qui avait
SUR-APPRIS avec les hyperparametres par defaut (README : SVM 0,590 / RF
0,611, PAS de gain net sur v2=0,617 malgre 50x plus de dimensions).
Hypothese non testee a l'epoque (note 3) : sur-ajustement, pas absence de
signal -- teste ici avec une REDUCTION DE DIMENSION (PCA) avant
classification, sur les features v3 deja extraites (features_laplacien_v3.npz,
split sujet, 4 fenetres) + les hyperparametres deja regles (tuner_laplacien_v2.py :
SVM C=3/gamma=scale, HistGradientBoosting tune).

Deux variantes testees :
1. PCA(v3) seul, plusieurs nombres de composantes -- est-ce que compresser
   les 6200 dims de v3 recupere le signal des vecteurs propres sans le
   sur-ajustement ?
2. concat[v2, PCA(v3)] -- est-ce que l'information des vecteurs propres
   AJOUTE quelque chose au-dessus de v2 (valeurs propres seules, notre
   meilleure feature a ce jour, 0,659 avec SVM tune) une fois compressee ?

GroupKFold par sujet (comme tuner_laplacien_v2.py) pour choisir le nombre
de composantes PCA sans fuite (chaque sequence produit 4 lignes correlees)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from dataset_jepa import decouper_train_val

SVM_TUNE_PARAMS = dict(kernel="rbf", C=3.0, gamma="scale")
HGB_TUNE_PARAMS = dict(learning_rate=0.1, max_depth=6, max_iter=400, l2_regularization=0.0, random_state=0)
N_COMPOSANTES = [10, 20, 50, 100, 200, 400]


def _groupes(sequences, nb_fenetres=4):
    g = []
    for (_env, sujet, _action) in sequences:
        g.extend([sujet] * nb_fenetres)
    return np.array(g)


def _evaluer_config(nom, X_train, y_train, X_val, y_val, groupes, n_comp):
    pipe_svm = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=n_comp, random_state=0)),
                          ("clf", SVC(**SVM_TUNE_PARAMS))])
    pipe_hgb = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=n_comp, random_state=0)),
                          ("clf", HistGradientBoostingClassifier(**HGB_TUNE_PARAMS))])

    cv = GroupKFold(n_splits=5)
    cv_svm = cross_val_score(pipe_svm, X_train, y_train, cv=cv, groups=groupes, scoring="accuracy", n_jobs=-1).mean()
    cv_hgb = cross_val_score(pipe_hgb, X_train, y_train, cv=cv, groups=groupes, scoring="accuracy", n_jobs=-1).mean()

    pipe_svm.fit(X_train, y_train)
    preds_svm = pipe_svm.predict(X_val)
    acc_svm = (preds_svm == y_val).mean()
    f1_svm = f1_score(y_val, preds_svm, average="macro", zero_division=0)

    pipe_hgb.fit(X_train, y_train)
    preds_hgb = pipe_hgb.predict(X_val)
    acc_hgb = (preds_hgb == y_val).mean()
    f1_hgb = f1_score(y_val, preds_hgb, average="macro", zero_division=0)

    print(f"[{nom}] n_comp={n_comp:4d}  SVM: cv={cv_svm:.3f} VAL_acc={acc_svm:.3f} VAL_f1={f1_svm:.3f}  |  "
          f"HGB: cv={cv_hgb:.3f} VAL_acc={acc_hgb:.3f} VAL_f1={f1_hgb:.3f}", flush=True)
    return dict(n_comp=n_comp, cv_svm=cv_svm, acc_svm=acc_svm, f1_svm=f1_svm,
                cv_hgb=cv_hgb, acc_hgb=acc_hgb, f1_hgb=f1_hgb)


if __name__ == "__main__":
    print("chargement features v2 (quadrants, 600 dims) et v3 (quadrants+vecteurs, 6200 dims)...")
    d2 = np.load("mmfi-src/features_laplacien_v2.npz")
    d3 = np.load("mmfi-src/features_laplacien_v3.npz")
    assert np.array_equal(d2["y_train"], d3["y_train"]) and np.array_equal(d2["y_val"], d3["y_val"]), \
        "v2 et v3 doivent avoir ete extraites sur le meme ordre de sequences"

    y_train, y_val = d3["y_train"], d3["y_val"]
    X3_train, X3_val = d3["X_train"], d3["X_val"]
    X2_train, X2_val = d2["X_train"], d2["X_val"]
    Xcat_train = np.concatenate([X2_train, X3_train], axis=1)
    Xcat_val = np.concatenate([X2_val, X3_val], axis=1)

    seq_train, _ = decouper_train_val()
    groupes = _groupes(seq_train)
    assert len(groupes) == X3_train.shape[0]

    print("\n=== reference (sans PCA, README) === v2 SVM_tune=0.638 HGB_tune=0.649 | "
          "v3 SVM_defaut=0.590 RF_defaut=0.611 (aucun gain net vs v2)\n")

    resultats = {"v3_pca": [], "concat_v2_v3_pca": []}

    print("--- PCA(v3) seul ---")
    for n in N_COMPOSANTES:
        resultats["v3_pca"].append(_evaluer_config("v3_pca", X3_train, y_train, X3_val, y_val, groupes, n))

    print("\n--- PCA(concat[v2,v3]) ---")
    for n in N_COMPOSANTES:
        resultats["concat_v2_v3_pca"].append(
            _evaluer_config("concat_pca", Xcat_train, y_train, Xcat_val, y_val, groupes, n))

    print("\n\n=== RESUME ===")
    print("reference sans PCA : v2 SVM_tune=0.638 HGB_tune=0.649 (meilleur sur split sujet 4 fenetres) ; "
          "v2 8-fenetres SVM_tune=0.659 (meilleur absolu, features differentes, non comparable directement ici)")
    for nom, liste in resultats.items():
        meilleur_svm = max(liste, key=lambda r: r["acc_svm"])
        meilleur_hgb = max(liste, key=lambda r: r["acc_hgb"])
        print(f"{nom}: meilleur SVM -> n_comp={meilleur_svm['n_comp']} VAL_acc={meilleur_svm['acc_svm']:.3f} "
              f"VAL_f1={meilleur_svm['f1_svm']:.3f}")
        print(f"{nom}: meilleur HGB -> n_comp={meilleur_hgb['n_comp']} VAL_acc={meilleur_hgb['acc_hgb']:.3f} "
              f"VAL_f1={meilleur_hgb['f1_hgb']:.3f}")
    print("TUNER_LAPLACIEN_V3_PCA_OK")
