"""Trace les courbes d'entrainement (historique_entrainement_{methode}.csv)
et de linear probing (log_evaluation_v2.txt) -- genere les figures
mmfi-src/courbes_entrainement_v2.png et mmfi-src/comparaison_probing_v2.png,
pour documenter visuellement le correctif du 3 septembre (perte_uniformite_poolee)."""

import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RACINE = os.path.dirname(os.path.abspath(__file__))


def _lire_historique(methode):
    chemin = os.path.join(RACINE, f"historique_entrainement_{methode}.csv")
    rows = list(csv.DictReader(open(chemin)))
    etapes = [int(r["etape"]) for r in rows]
    perte_train = [float(r["perte_train"]) for r in rows]
    eval_rows = [(int(r["etape"]), r) for r in rows if r["similarite_effondrement"] != ""]
    etapes_eval = [e for e, _ in eval_rows]
    sim = [float(r["similarite_effondrement"]) for _, r in eval_rows]
    sim_poolee = [float(r["similarite_effondrement_poolee"]) for _, r in eval_rows]
    return etapes, perte_train, etapes_eval, sim, sim_poolee


fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for col, methode in enumerate(["grille", "points"]):
    etapes, perte_train, etapes_eval, sim, sim_poolee = _lire_historique(methode)

    ax = axes[0, col]
    ax.plot(etapes, perte_train, linewidth=0.7, color="tab:blue")
    ax.set_title(f"{methode} — perte d'entraînement")
    ax.set_xlabel("pas")
    ax.axhline(0, color="grey", linewidth=0.5)

    ax = axes[1, col]
    ax.plot(etapes_eval, sim, marker="o", markersize=3, label="par patch", color="tab:orange")
    ax.plot(etapes_eval, sim_poolee, marker="o", markersize=3, label="poolée (mean+max)", color="tab:red")
    ax.axhline(0.999, color="grey", linestyle="--", linewidth=0.8, label="effondrement total (avant correctif)")
    ax.set_title(f"{methode} — similarité cosinus (effondrement)")
    ax.set_xlabel("pas")
    ax.set_ylim(-0.15, 1.05)
    ax.legend(fontsize=7)

fig.suptitle("Entraînement JEPA (3000 pas, MM-Fi) — avec perte_uniformite_poolee (3 septembre)")
fig.tight_layout()
fig.savefig(os.path.join(RACINE, "courbes_entrainement_v2.png"), dpi=130)
print("-> courbes_entrainement_v2.png")


def _lire_probing(chemin):
    """Parse le format '[methode, checkpoint] etape N/M  ... precision_val=X ...'
    -> dict condition -> (etapes, precisions)."""
    motif = re.compile(
        r"\[(\w+), (\S+)\] etape\s+(\d+)/\d+.*?precision_val=([\d.]+)"
    )
    par_condition = {}
    for ligne in open(chemin, encoding="utf-8", errors="ignore"):
        m = motif.search(ligne)
        if not m:
            continue
        methode, ckpt, etape, precision = m.groups()
        nom = f"{methode}_{'preentraine' if ckpt != 'ALEATOIRE' else 'aleatoire'}"
        par_condition.setdefault(nom, ([], []))
        par_condition[nom][0].append(int(etape))
        par_condition[nom][1].append(float(precision))
    return par_condition


par_condition = _lire_probing(os.path.join(RACINE, "log_evaluation_v2.txt"))
chance = 1 / 27

fig, ax = plt.subplots(figsize=(8, 5))
couleurs = {
    "grille_preentraine": "tab:blue", "grille_aleatoire": "tab:cyan",
    "points_preentraine": "tab:red", "points_aleatoire": "tab:orange",
}
for nom, (etapes, precisions) in par_condition.items():
    ax.plot(etapes, precisions, label=nom, color=couleurs.get(nom), linewidth=1.3)
ax.axhline(chance, color="grey", linestyle="--", linewidth=0.8, label=f"hasard (1/27={chance:.3f})")
ax.set_xlabel("pas de linear probing")
ax.set_ylabel("précision (validation)")
ax.set_title("Linear probing — 27 actions MM-Fi (3 septembre, post-correctif)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(RACINE, "comparaison_probing_v2.png"), dpi=130)
print("-> comparaison_probing_v2.png")
