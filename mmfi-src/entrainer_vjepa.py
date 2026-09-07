"""Lancement de l'entrainement complet grille mirroir V-JEPA (3 septembre),
regularisation reactivee (poids_uniformite_poolee=1.0) suite au diagnostic
qui a confirme que le mirroir pur (reg_coeff: 0.0) n'evite pas l'effondrement
sur notre donnee (cf. README/diag_vjepa_mirror.log) alors que la combiner
avec cette regularisation le fait (cf. diag_vjepa_mirror_reg.log)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entrainement import entrainer

entrainer(
    methode="grille", source="mmfi", nb_etapes=3000, suffixe="grille_vjepa",
    poids_uniformite_poolee=1.0, temperature_uniformite=2.0,
)
print("ENTRAINEMENT_VJEPA_OK")
