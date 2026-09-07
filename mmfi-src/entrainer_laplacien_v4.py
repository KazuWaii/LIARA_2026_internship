"""Consomme features_laplacien_v4_all.npz (extraire_laplacien_v4.py --
quadrants, 8 fenetres/sequence, 1080 sequences) pour comparer DEUX
ablations de la piste B (README, meilleur resultat 0,617 SVM, split par
sujet, 4 fenetres) SANS re-extraire :

1. Meme split que jusqu'ici (8 derniers sujets, decouper_train_val) mais
   avec le fenetrage plus dense (8 au lieu de 4) -- isole l'effet du
   fenetrage seul.
2. Split par ENVIRONNEMENT (E01-E03 train / E04 val, decouper_train_val_
   environnement) -- reproduit le protocole exact du papier, seul point de
   comparaison vraiment homogene avec leurs 81-90%.

Pour chaque split : SVM/RF par defaut (reference) PUIS tunes (GridSearchCV,
GroupKFold par sujet sur le train -- memes grilles que tuner_laplacien_v2.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from dataset_jepa import decouper_train_val, decouper_train_val_environnement

# Meilleurs hyperparametres deja trouves par GridSearchCV (GroupKFold sujet,
# cf. tuner_laplacien_v2.py / log_tuner_laplacien_v2.txt, split sujet
# original 4 fenetres) -- reutilises ici DIRECTEMENT (pas de nouveau grid
# search, trop couteux x2 avec le one-vs-one 27-classes, cf. session) pour
# isoler l'effet du split/fenetrage sur DES HYPERPARAMETRES FIXES. Un grid
# search dedie a chaque nouveau split resterait a faire si ces resultats
# sont prometteurs.
SVM_TUNE_PARAMS = dict(kernel="rbf", C=3.0, gamma="scale")
HGB_TUNE_PARAMS = dict(learning_rate=0.1, max_depth=6, max_iter=400, l2_regularization=0.0, random_state=0)


def _masque(sequences, envs, sujets):
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


def _evaluer(nom, X_train, y_train, X_val, y_val, groupes_train):
    print(f"\n########## {nom} ##########")
    print(f"train: {X_train.shape}  val: {X_val.shape}  sujets_train={len(set(groupes_train))}")

    scaler = StandardScaler().fit(X_train)
    Xtr, Xva = scaler.transform(X_train), scaler.transform(X_val)

    print(f"-- SVM par defaut (C=1.0, reference) --")
    svm = SVC(kernel="rbf", C=1.0).fit(Xtr, y_train)
    preds = svm.predict(Xva)
    acc = (preds == y_val).mean()
    f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    print(f"   accuracy={acc:.3f} f1_macro={f1:.3f}")

    print(f"-- RF par defaut --")
    rf = RandomForestClassifier(random_state=0).fit(Xtr, y_train)
    preds_rf = rf.predict(Xva)
    acc_rf = (preds_rf == y_val).mean()
    f1_rf = f1_score(y_val, preds_rf, average="macro", zero_division=0)
    print(f"   accuracy={acc_rf:.3f} f1_macro={f1_rf:.3f}")

    print(f"-- SVM (hyperparametres deja tunes: {SVM_TUNE_PARAMS}) --")
    svm_t = SVC(**SVM_TUNE_PARAMS).fit(Xtr, y_train)
    preds_t = svm_t.predict(Xva)
    acc_t = (preds_t == y_val).mean()
    f1_t = f1_score(y_val, preds_t, average="macro", zero_division=0)
    print(f"   accuracy={acc_t:.3f} f1_macro={f1_t:.3f}")

    print(f"-- HistGradientBoosting (hyperparametres deja tunes: {HGB_TUNE_PARAMS}) --")
    hgb_t = HistGradientBoostingClassifier(**HGB_TUNE_PARAMS).fit(Xtr, y_train)
    preds_h = hgb_t.predict(Xva)
    acc_h = (preds_h == y_val).mean()
    f1_h = f1_score(y_val, preds_h, average="macro", zero_division=0)
    print(f"   accuracy={acc_h:.3f} f1_macro={f1_h:.3f}")

    return dict(svm_defaut=acc, rf_defaut=acc_rf, svm_tune=acc_t, svm_tune_f1=f1_t,
                hgb_tune=acc_h, hgb_tune_f1=f1_h)


if __name__ == "__main__":
    data = np.load("mmfi-src/features_laplacien_v4_all.npz")
    X, y, envs, sujets = data["X"], data["y"], data["envs"], data["sujets"]
    print(f"features totales: X={X.shape}")

    resultats = {}

    seq_train, seq_val = decouper_train_val()
    m_train, m_val = _masque(seq_train, envs, sujets), _masque(seq_val, envs, sujets)
    resultats["split_sujet_8fenetres"] = _evaluer(
        "SPLIT SUJET (8 derniers sujets val, comme avant) -- 8 fenetres/sequence",
        X[m_train], y[m_train], X[m_val], y[m_val], sujets[m_train])

    seq_train_e, seq_val_e = decouper_train_val_environnement()
    m_train_e, m_val_e = _masque(seq_train_e, envs, sujets), _masque(seq_val_e, envs, sujets)
    resultats["split_environnement"] = _evaluer(
        "SPLIT ENVIRONNEMENT (E01-E03 train / E04 val, protocole du papier) -- 8 fenetres/sequence",
        X[m_train_e], y[m_train_e], X[m_val_e], y[m_val_e], sujets[m_train_e])

    print("\n\n=== RESUME FINAL ===")
    print("reference anterieure (4 fenetres, split sujet) : SVM_defaut=0.617 "
          "SVM_tune=0.638 HGB_tune=0.649 (meilleur a ce jour)")
    for nom, r in resultats.items():
        print(f"{nom}: SVM_defaut={r['svm_defaut']:.3f}  RF_defaut={r['rf_defaut']:.3f}  "
              f"SVM_tune={r['svm_tune']:.3f} (f1={r['svm_tune_f1']:.3f})  "
              f"HGB_tune={r['hgb_tune']:.3f} (f1={r['hgb_tune_f1']:.3f})")
    print("ENTRAINEMENT_LAPLACIEN_V4_OK")
