"""Essaie des classifieurs CLASSIQUES (SVM/RandomForest/HistGradientBoosting,
memes hyperparametres regles que tuner_laplacien_v2.py) sur les features
Point-MAE DEJA EXTRAITES (features_pointmae.npz -- encodeur ShapeNet-pretrained
FIGE, 768 dims/frame, 8 frames/fenetre) au lieu de la tete lineaire+pooling
appris (entrainer_pointmae_seeds.py, 0,934+-0,006). Meme pooling mean+max
TEMPOREL (8 frames -> 1536 dims) qu'avant, mais classifieur classique a la
place d'un entrainement par descente de gradient -- teste si un classifieur
non-parametrique/a marge (SVM) ou un ensemble d'arbres extrait davantage du
signal deja tres riche de ces features que la tete lineaire simple.

GroupKFold par sujet (comme tuner_laplacien_v2.py) pour le reglage
d'hyperparametres SVM, PAS un K-Fold aveugle (chaque sequence produit
plusieurs fenetres correlees)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from dataset_jepa import decouper_train_val
from entrainer_laplacien_v4 import _masque


def _pool_temporel(X):
    """X: (N,8,768) -> (N,1536) mean+max sur les 8 frames."""
    return np.concatenate([X.mean(axis=1), X.max(axis=1)], axis=1)


if __name__ == "__main__":
    data = np.load("mmfi-src/features_pointmae.npz")
    X, y, envs, sujets = data["X"], data["y"], data["envs"], data["sujets"]
    seq_train, seq_val = decouper_train_val()
    m_train, m_val = _masque(seq_train, envs, sujets), _masque(seq_val, envs, sujets)

    X_train = _pool_temporel(X[m_train])
    X_val = _pool_temporel(X[m_val])
    y_train, y_val = y[m_train], y[m_val]
    groupes_train = sujets[m_train]
    print(f"X_train: {X_train.shape}  X_val: {X_val.shape}  sujets_train={len(set(groupes_train))}")

    scaler = StandardScaler().fit(X_train)
    Xtr, Xva = scaler.transform(X_train), scaler.transform(X_val)

    print("\n=== SVM par defaut (C=1.0, reference) ===")
    svm = SVC(kernel="rbf", C=1.0).fit(Xtr, y_train)
    preds = svm.predict(Xva)
    print(f"   accuracy={(preds==y_val).mean():.3f} f1_macro={f1_score(y_val,preds,average='macro'):.3f}")

    print("\n=== RandomForest par defaut ===")
    rf = RandomForestClassifier(random_state=0).fit(Xtr, y_train)
    preds_rf = rf.predict(Xva)
    print(f"   accuracy={(preds_rf==y_val).mean():.3f} f1_macro={f1_score(y_val,preds_rf,average='macro'):.3f}")

    print("\n=== HistGradientBoosting (hyperparametres deja regles pour Laplacien) ===")
    hgb = HistGradientBoostingClassifier(learning_rate=0.1, max_depth=6, max_iter=400,
                                          l2_regularization=0.0, random_state=0).fit(Xtr, y_train)
    preds_h = hgb.predict(Xva)
    print(f"   accuracy={(preds_h==y_val).mean():.3f} f1_macro={f1_score(y_val,preds_h,average='macro'):.3f}")

    cv = GroupKFold(n_splits=5)
    print("\n=== SVM regle (GridSearchCV, GroupKFold sujet) ===")
    pipe = Pipeline([("scale", StandardScaler()), ("clf", SVC())])
    grille = {"clf__kernel": ["rbf"], "clf__C": [0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
              "clf__gamma": ["scale", "auto", 0.001, 0.003, 0.01, 0.03]}
    gs = GridSearchCV(pipe, grille, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train, groups=groupes_train)
    preds_t = gs.predict(X_val)
    acc_t = (preds_t == y_val).mean()
    print(f"   meilleurs params={gs.best_params_} cv_acc={gs.best_score_:.3f} "
          f"VAL accuracy={acc_t:.3f} f1_macro={f1_score(y_val,preds_t,average='macro'):.3f}")

    print("\n\n=== RESUME (reference : tete lineaire+pooling appris = 0.934) ===")
    print(f"SVM defaut: {(preds==y_val).mean():.3f}  RF defaut: {(preds_rf==y_val).mean():.3f}  "
          f"HGB (hyperparams Laplacien): {(preds_h==y_val).mean():.3f}  SVM regle: {acc_t:.3f}")
    print("TUNER_POINTMAE_CLASSIQUE_OK")
