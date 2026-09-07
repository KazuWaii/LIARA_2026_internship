"""Verification de forme (3 septembre) : passe avant complete sur donnees
factices avant de lancer un vrai entrainement -- valide le nouveau masquage
(par-echantillon, deux groupes), la nouvelle perte L1 (listes), le nouveau
tokenizer (sans token_vide), sans depenser de temps GPU/disque reel."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from modele import ModeleJEPA
from perte import perte_jepa

torch.manual_seed(0)

B, T, H, W = 4, 16, 64, 96
modele = ModeleJEPA()
print(f"parametres entraines: {sum(p.numel() for p in modele.parameters() if p.requires_grad):,}")

tenseur = torch.zeros(B, T, 4, H, W)
tenseur[:, :, :3] = torch.randn(B, T, 3, H, W) * 0.5
tenseur[:, :, 3] = (torch.rand(B, T, H, W) > 0.85).float()  # ~15% d'occupation, plausible pour MM-Fi

predictions, cibles, contexte_encode = modele(tenseur)
print(f"nb groupes: {len(predictions)}")
for i, (p, c) in enumerate(zip(predictions, cibles)):
    print(f"  groupe {i}: predictions {tuple(p.shape)}  cibles {tuple(c.shape)}")
print(f"contexte_encode: {[tuple(c.shape) for c in contexte_encode]}")

perte, perte_jepa_val, perte_reg = perte_jepa(predictions, cibles, contexte_encode)
print(f"perte={perte.item():.4f}  perte_jepa={perte_jepa_val:.4f}  perte_reg={perte_reg:.4f}")

optimiseur = torch.optim.AdamW(
    list(modele.tokenizer.parameters()) + list(modele.masquage.parameters())
    + list(modele.encodeur_contexte.parameters()) + list(modele.predicteur.parameters()),
    lr=1e-4,
)
optimiseur.zero_grad()
perte.backward()

sans_gradient = [n for n, p in modele.named_parameters()
                  if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
print(f"parametres entraines SANS gradient non-nul ({len(sans_gradient)}): {sans_gradient[:10]}")
assert "masquage.tokens_masque.0" not in sans_gradient, "token masque groupe 0 sans gradient"
assert "masquage.tokens_masque.1" not in sans_gradient, "token masque groupe 1 sans gradient"

grad_cible = [n for n, p in modele.encodeur_cible.named_parameters() if p.grad is not None]
assert not grad_cible, f"l'encodeur cible ne devrait JAMAIS recevoir de gradient : {grad_cible}"

optimiseur.step()
modele.mettre_a_jour_cible(0.996)

print("SMOKE_TEST_OK")
