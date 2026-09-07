"""Diagnostic (3 septembre soir) : valide l'entrainement JRDB standalone
(architecture V-JEPA mirroir + regularisation, source="jrdb", PROFIL_JRDB
natif) avant un run complet -- meme methodologie que les diagnostics grille
precedents (500 pas, verifie absence de NaN et les deux diagnostics
d'effondrement)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entrainement import entrainer

entrainer(
    methode="grille", source="jrdb", nb_etapes=500, suffixe="jrdb_diag",
    poids_uniformite_poolee=1.0, temperature_uniformite=2.0,
)
print("DIAG_JRDB_OK")
