"""Recherche d'hyperparametres (SVM/RF/HistGradientBoosting) sur les features
Laplacien v2 (quadrants) DEJA extraites (features_laplacien_v2.npz, split par
sujet, cf. entrainer_laplacien_v2.py) -- objectif : verifier si le 0,617 SVM
(hyperparametres par defaut C=1.0) peut etre ameliore par un simple reglage,
avant de payer le cout de re-extraction pour d'autres ablations (fenetres,
split par environnement, cf. tuner_laplacien_v4.py).

Validation croisee GROUPEE PAR SUJET (GroupKFold) sur le train, PAS un
K-Fold aveugle : chaque sequence produit 4 lignes (fenetres temporelles)
tres correlees entre elles (meme sequence/sujet) -- un K-Fold aveugle les
repartirait entre folds et surestimerait le score de generalisation. Les
groupes sont reconstruits a partir de l'ordre deterministe de
decouper_train_val() (verifie : 864 sequences x 4 fenetres = 3456 lignes,
aucune sequence ignoree)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, f1_score

from dataset_jepa import decouper_train_val

NB_FENETRES = 4


def _groupes(sequences, nb_fenetres=NB_FENETRES):
    """Reconstruit le sujet associe a chaque ligne de features (nb_fenetres
    lignes par sequence, meme ordre que construire_dataset)."""
    g = []
    for (_env, sujet, _action) in sequences:
        g.extend([sujet] * nb_fenetres)
    return np.array(g)


if __name__ == "__main__":
    data = np.load("mmfi-src/features_laplacien_v2.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    print(f"X_train: {X_train.shape}  X_val: {X_val.shape}")

    sequences_train, _ = decouper_train_val()
    groupes = _groupes(sequences_train)
    assert len(groupes) == X_train.shape[0], (len(groupes), X_train.shape[0])
    nb_sujets_train = len(set(groupes))
    print(f"sujets train: {nb_sujets_train} -- GroupKFold(5)")

    cv = GroupKFold(n_splits=5)

    resultats = {}

    print("\n=== SVM (grid: C x gamma x kernel) ===")
    pipe_svm = Pipeline([("scale", StandardScaler()), ("clf", SVC())])
    grille_svm = {
        "clf__kernel": ["rbf"],
        "clf__C": [0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
        "clf__gamma": ["scale", "auto", 0.001, 0.003, 0.01, 0.03],
    }
    gs_svm = GridSearchCV(pipe_svm, grille_svm, cv=cv,
                           scoring="accuracy", n_jobs=-1, verbose=1)
    gs_svm.fit(X_train, y_train, groups=groupes)
    print(f"meilleurs params SVM: {gs_svm.best_params_}  cv_accuracy={gs_svm.best_score_:.3f}")
    preds = gs_svm.predict(X_val)
    acc = (preds == y_val).mean()
    prec = precision_score(y_val, preds, average="macro", zero_division=0)
    f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    print(f"SVM tune -> VAL accuracy={acc:.3f} precision_macro={prec:.3f} f1_macro={f1:.3f}")
    resultats["svm_tune"] = dict(cv_acc=gs_svm.best_score_, val_acc=acc, val_f1=f1, params=gs_svm.best_params_)

    print("\n=== RandomForest (grid) ===")
    pipe_rf = Pipeline([("scale", StandardScaler()), ("clf", RandomForestClassifier(random_state=0))])
    grille_rf = {
        "clf__n_estimators": [200, 500, 800],
        "clf__max_depth": [None, 10, 20, 30],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
    }
    gs_rf = GridSearchCV(pipe_rf, grille_rf, cv=cv,
                          scoring="accuracy", n_jobs=-1, verbose=1)
    gs_rf.fit(X_train, y_train, groups=groupes)
    print(f"meilleurs params RF: {gs_rf.best_params_}  cv_accuracy={gs_rf.best_score_:.3f}")
    preds_rf = gs_rf.predict(X_val)
    acc_rf = (preds_rf == y_val).mean()
    prec_rf = precision_score(y_val, preds_rf, average="macro", zero_division=0)
    f1_rf = f1_score(y_val, preds_rf, average="macro", zero_division=0)
    print(f"RF tune -> VAL accuracy={acc_rf:.3f} precision_macro={prec_rf:.3f} f1_macro={f1_rf:.3f}")
    resultats["rf_tune"] = dict(cv_acc=gs_rf.best_score_, val_acc=acc_rf, val_f1=f1_rf, params=gs_rf.best_params_)

    print("\n=== HistGradientBoosting (grid) ===")
    pipe_hgb = Pipeline([("scale", StandardScaler()), ("clf", HistGradientBoostingClassifier(random_state=0))])
    grille_hgb = {
        "clf__max_iter": [200, 400],
        "clf__learning_rate": [0.03, 0.1, 0.3],
        "clf__max_depth": [None, 6, 12],
        "clf__l2_regularization": [0.0, 1.0],
    }
    gs_hgb = GridSearchCV(pipe_hgb, grille_hgb, cv=cv,
                           scoring="accuracy", n_jobs=-1, verbose=1)
    gs_hgb.fit(X_train, y_train, groups=groupes)
    print(f"meilleurs params HGB: {gs_hgb.best_params_}  cv_accuracy={gs_hgb.best_score_:.3f}")
    preds_hgb = gs_hgb.predict(X_val)
    acc_hgb = (preds_hgb == y_val).mean()
    prec_hgb = precision_score(y_val, preds_hgb, average="macro", zero_division=0)
    f1_hgb = f1_score(y_val, preds_hgb, average="macro", zero_division=0)
    print(f"HGB tune -> VAL accuracy={acc_hgb:.3f} precision_macro={prec_hgb:.3f} f1_macro={f1_hgb:.3f}")
    resultats["hgb_tune"] = dict(cv_acc=gs_hgb.best_score_, val_acc=acc_hgb, val_f1=f1_hgb, params=gs_hgb.best_params_)

    print("\n\n=== RESUME (reference sans tuning : SVM=0.617 RF=0.601, cf. README) ===")
    for nom, r in resultats.items():
        print(f"{nom}: cv_acc={r['cv_acc']:.3f}  VAL acc={r['val_acc']:.3f}  VAL f1={r['val_f1']:.3f}  params={r['params']}")
    print("TUNER_LAPLACIEN_V2_OK")
