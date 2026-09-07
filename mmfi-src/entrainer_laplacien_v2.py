"""Version enrichie de entrainer_laplacien.py -- ajoute la partition en 4
quadrants spatiaux (lateral x vertical) + corps entier, cf.
laplacien_spectral.descripteur_pose_quadrants. Suite au premier resultat
decevant (v1 sans quadrants : SVM 0,218, RF 0,205, tres en-dessous du 0,90
rapporte par le papier ET de notre baseline deep learning 0,51) -- le papier
identifie cette partition comme l'ablation la PLUS impactante (61%->94% sur
le 13-classes), testee en priorite avant les statistiques de vecteurs
propres (plus couteuses)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, f1_score

from dataset_jepa import decouper_train_val
from entrainer_laplacien import construire_dataset

if __name__ == "__main__":
    sequences_train, sequences_val = decouper_train_val()
    print(f"sequences train: {len(sequences_train)}  val: {len(sequences_val)}")

    print("extraction features train (quadrants)...")
    X_train, y_train = construire_dataset(sequences_train, quadrants=True)
    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")

    print("extraction features val (quadrants)...")
    X_val, y_val = construire_dataset(sequences_val, quadrants=True)
    print(f"X_val: {X_val.shape}  y_val: {y_val.shape}")

    np.savez("mmfi-src/features_laplacien_v2.npz",
             X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    print("features sauvegardees -> mmfi-src/features_laplacien_v2.npz")

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
    print("ENTRAINEMENT_LAPLACIEN_V2_OK")
