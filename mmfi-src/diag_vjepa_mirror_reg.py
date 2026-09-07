"""Diagnostic (3 septembre) : mirroir V-JEPA + perte_uniformite_poolee
reactivee (poids_uniformite_poolee=1.0) -- teste si la regularisation
prouvee la veille (0.9933 -> ~0.0-0.1 de similarite poolee) resout aussi
l'effondrement observe sur le mirroir pur (reg_coeff: 0.0, cf.
diag_vjepa_mirror.py : sim_poolee restee ~0.93-0.96 tout le run)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entrainement import entrainer

entrainer(
    methode="grille", source="mmfi", nb_etapes=500, suffixe="grille_vjepa_diag_reg",
    poids_uniformite_poolee=1.0, temperature_uniformite=2.0,
)
print("DIAG_VJEPA_REG_OK")
