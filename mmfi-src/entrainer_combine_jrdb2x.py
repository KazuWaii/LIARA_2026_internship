"""Suite du 4 septembre matin -- la comparaison multi-graines a montre que
MM-Fi+JRDB (25 sequences JRDB train, <3% du corpus MM-Fi) est statistiquement
indiscernable de from-scratch en fine-tuning (0,498+-0,044 vs 0,509+-0,044,
n=3 graines chacune). Conclusion honnete a l'epoque : le volume de JRDB
ajoute est probablement trop marginal pour produire un effet mesurable, pas
necessairement l'idee elle-meme.

Ce script teste directement cette hypothese via `source="mmfi_jrdb2x"``
(entrainement.py) qui DOUBLE le volume JRDB en reutilisant son split "test"
(D:/jrdb/test/, deja extrait le 3 septembre, jamais utilise) -- legitime car
JEPA est auto-supervise, aucune etiquette JRDB n'est utilisee nulle part
(seul MM-Fi fournit les etiquettes HAR pour le fine-tuning aval). ~50
sequences JRDB train au lieu de 25 (~5-6% du corpus combine au lieu de
<3%)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entrainement import entrainer

entrainer(
    methode="grille", source="mmfi_jrdb2x", nb_etapes=3000, suffixe="mmfi_jrdb2x",
    poids_uniformite_poolee=1.0, temperature_uniformite=2.0,
)
print("ENTRAINEMENT_COMBINE_JRDB2X_OK")
