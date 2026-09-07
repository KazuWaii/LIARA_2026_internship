"""Diagnostic (3 septembre) : valide le mirroir V-JEPA (masquage par-echantillon
deux groupes, perte L1, regularisation OFF par defaut, scheduleurs LR/WD/EMA)
sur un entrainement court avant de lancer les 3000 pas complets. Verifie
absence de NaN, tendance de la perte, et les deux diagnostics d'effondrement
(par-patch et poole) -- sans aucune regularisation explicite active, mirroir
fidele de reg_coeff: 0.0."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entrainement import entrainer

entrainer(methode="grille", source="mmfi", nb_etapes=500, suffixe="grille_vjepa_diag")
print("DIAG_VJEPA_OK")
