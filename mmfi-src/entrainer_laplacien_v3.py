"""Version complete de entrainer_laplacien.py -- quadrants (v2, SVM 0,617)
+ statistiques de vecteurs propres, deuxieme ablation la plus impactante du
papier (arXiv:2509.23255, +6pts sur le 13-classes AU-DESSUS des quadrants
seuls). Feature vector 6200 dims (5 parties x (30 valeurs propres + 40
vecteurs propres x 7 stats) x 4 stats temporelles)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, f1_score

import mmfi_dataset as mmfi
from dataset_jepa import decouper_train_val
from laplacien_spectral import extraire_fenetres_sequence

CODES_ACTIONS = sorted(mmfi.ACTIONS.keys())
INDEX_ACTION = {code: i for i, code in enumerate(CODES_ACTIONS)}


def construire_dataset(sequences, nb_fenetres=4, nb_frames_echantillon=8, verbeux_tous=50):
    import time
    X, y = [], []
    t0 = time.time()
    for i, (env, sujet, action) in enumerate(sequences):
        frames = mmfi.lister_frames(env, sujet, action)
        pts = [mmfi.charger_frame_lidar(f) for f in frames]
        if len(pts) < 5:
            continue
        try:
            feats = extraire_fenetres_sequence(pts, nb_fenetres=nb_fenetres,
                                                nb_frames_echantillon=nb_frames_echantillon,
                                                quadrants=True, vecteurs=True)
        except Exception as e:
            print(f"  [!] sequence {env}/{sujet}/{action} echouee: {e}")
            continue
        label = INDEX_ACTION[action]
        X.extend(feats)
        y.extend([label] * len(feats))
        if (i + 1) % verbeux_tous == 0:
            duree = time.time() - t0
            print(f"  {i + 1}/{len(sequences)} sequences  [{duree:.1f}s]")
    return np.stack(X), np.array(y)


if __name__ == "__main__":
    sequences_train, sequences_val = decouper_train_val()
    print(f"sequences train: {len(sequences_train)}  val: {len(sequences_val)}")

    print("extraction features train (quadrants+vecteurs)...")
    X_train, y_train = construire_dataset(sequences_train)
    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")

    print("extraction features val (quadrants+vecteurs)...")
    X_val, y_val = construire_dataset(sequences_val)
    print(f"X_val: {X_val.shape}  y_val: {y_val.shape}")

    np.savez("mmfi-src/features_laplacien_v3.npz",
             X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    print("features sauvegardees -> mmfi-src/features_laplacien_v3.npz")

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

    print("\n=== SVM (rbf, C=1.0) ===")
    svm = SVC(kernel="rbf", C=1.0).fit(X_train_s, y_train)
    preds = svm.predict(X_val_s)
    precision = precision_score(y_val, preds, average="macro", zero_division=0)
    f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    acc = (preds == y_val).mean()
    print(f"SVM: accuracy={acc:.3f} precision_macro={precision:.3f} f1_macro={f1:.3f}")

    print("\n=== RandomForest (defaut) ===")
    rf = RandomForestClassifier(random_state=0).fit(X_train_s, y_train)
    preds_rf = rf.predict(X_val_s)
    precision_rf = precision_score(y_val, preds_rf, average="macro", zero_division=0)
    f1_rf = f1_score(y_val, preds_rf, average="macro", zero_division=0)
    acc_rf = (preds_rf == y_val).mean()
    print(f"RF: accuracy={acc_rf:.3f} precision_macro={precision_rf:.3f} f1_macro={f1_rf:.3f}")

    print("\n\n=== RESUME ===")
    print(f"SVM: accuracy(=notre 'precision')={acc:.3f} precision_macro_sklearn={precision:.3f} f1_macro={f1:.3f}")
    print(f"RF:  accuracy(=notre 'precision')={acc_rf:.3f} precision_macro_sklearn={precision_rf:.3f} f1_macro={f1_rf:.3f}")
    print("ENTRAINEMENT_LAPLACIEN_V3_OK")
