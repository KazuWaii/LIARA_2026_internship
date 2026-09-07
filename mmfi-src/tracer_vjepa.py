"""Trace les courbes du mirroir V-JEPA (entrainement + linear probing) --
genere mmfi-src/courbes_vjepa.png, meme esprit que tracer_courbes.py mais
pour le run grille_vjepa (un seul mask, deux conditions de probing)."""

import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RACINE = os.path.dirname(os.path.abspath(__file__))

rows = list(csv.DictReader(open(os.path.join(RACINE, "historique_entrainement_grille_vjepa.csv"))))
etapes = [int(r["etape"]) for r in rows]
perte_train = [float(r["perte_jepa"]) for r in rows]
eval_rows = [(int(r["etape"]), r) for r in rows if r["similarite_effondrement"] != ""]
etapes_eval = [e for e, _ in eval_rows]
sim = [float(r["similarite_effondrement"]) for _, r in eval_rows]
sim_poolee = [float(r["similarite_effondrement_poolee"]) for _, r in eval_rows]

motif = re.compile(r"\[(\w+), (\S+)\] etape\s+(\d+)/\d+.*?precision_val=([\d.]+)")
par_condition = {}
for ligne in open(os.path.join(RACINE, "log_eval_vjepa.txt"), encoding="utf-8", errors="ignore"):
    m = motif.search(ligne)
    if not m:
        continue
    methode, ckpt, etape, precision = m.groups()
    nom = "vjepa_preentraine" if ckpt != "ALEATOIRE" else "vjepa_aleatoire"
    par_condition.setdefault(nom, ([], []))
    par_condition[nom][0].append(int(etape))
    par_condition[nom][1].append(float(precision))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

axes[0].plot(etapes, perte_train, linewidth=0.6, color="tab:blue")
axes[0].set_title("perte L1 (jepa) -- entrainement")
axes[0].set_xlabel("pas")
axes[0].axhline(0, color="grey", linewidth=0.5)

axes[1].plot(etapes_eval, sim, marker="o", markersize=3, label="par patch", color="tab:orange")
axes[1].plot(etapes_eval, sim_poolee, marker="o", markersize=3, label="poolée", color="tab:red")
axes[1].axhline(0.999, color="grey", linestyle="--", linewidth=0.8, label="effondrement total")
axes[1].set_title("similarité cosinus (effondrement)")
axes[1].set_xlabel("pas")
axes[1].set_ylim(-0.15, 1.05)
axes[1].legend(fontsize=8)

chance = 1 / 27
for nom, (e, p) in par_condition.items():
    axes[2].plot(e, p, label=nom, linewidth=1.3)
axes[2].axhline(chance, color="grey", linestyle="--", linewidth=0.8, label=f"hasard ({chance:.3f})")
axes[2].set_title("linear probing -- 27 actions MM-Fi")
axes[2].set_xlabel("pas de probing")
axes[2].set_ylabel("précision (validation)")
axes[2].legend(fontsize=8)

fig.suptitle("Mirroir V-JEPA (grille, MM-Fi) -- masquage 2 groupes + perte L1 + uniformité poolée")
fig.tight_layout()
fig.savefig(os.path.join(RACINE, "courbes_vjepa.png"), dpi=130)
print("-> courbes_vjepa.png")
