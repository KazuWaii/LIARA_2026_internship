"""Diagnostic (3 septembre soir) : pre-entrainement combine MM-Fi+JRDB
(source="mmfi_jrdb", JRDB projete via PROFIL_MMFI) -- teste si ajouter un
corpus domaine-proche plus grand aide, suite au fine-tuning qui a montre que
MM-Fi seul ne bat pas from-scratch (cf. README, section fine-tuning)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entrainement import entrainer

entrainer(
    methode="grille", source="mmfi_jrdb", nb_etapes=500, suffixe="mmfi_jrdb_diag",
    poids_uniformite_poolee=1.0, temperature_uniformite=2.0,
)
print("DIAG_COMBINE_OK")
