"""Entrainement complet combine MM-Fi+JRDB (3 septembre soir) -- teste si
ajouter un corpus domaine-proche plus grand (JRDB, 27 sequences campus)
ferme l'ecart trouve en fine-tuning (MM-Fi seul ne bat pas from-scratch,
cf. README). Diagnostic 500 pas confirme sain (diag_combine.log)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entrainement import entrainer

entrainer(
    methode="grille", source="mmfi_jrdb", nb_etapes=3000, suffixe="mmfi_jrdb",
    poids_uniformite_poolee=1.0, temperature_uniformite=2.0,
)
print("ENTRAINEMENT_COMBINE_OK")
