from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def getWeights(model, verbose=False):
    """Renvoie {nom_couche: noyaux} pour chaque Conv2D.
    Forme des noyaux : (kh, kw, canaux_entree, n_filtres)."""
    weights = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            weights[layer.name] = layer.get_weights()[0]
    if verbose:
        for nom, w in weights.items():
            print(f"Layer: {nom}, Weights shape: {w.shape}")
            for j in range(w.shape[-1]):
                print(f"Filter {j} weights (canal 0):\n{w[:, :, 0, j]}\n")
    return weights


def plot_filtres(model, name="filtres", max_canaux=8, max_filtres=16,
                 annoter=True, titre=None):
    """Affiche tous les filtres de toutes les couches Conv2D sur une seule figure.

    - une section par couche (titre = nom de la couche + forme)
    - colonnes = filtres de sortie, lignes = canaux d'entrée
    - bleu = positif, rouge = négatif, blanc = 0 ; intensité ∝ |valeur|
    - échelle symétrique [-m, m] commune à toute la couche (m = max |poids|),
      pour comparer les filtres d'une même couche entre eux
    - max_canaux / max_filtres : limite l'affichage pour les grosses couches
    """
    weights = getWeights(model)
    if not weights:
        print("Aucune couche Conv2D dans le modèle.")
        return

    # Taille de chaque section (en nombre de vignettes)
    tailles = []
    for w in weights.values():
        kh, kw, cin, cout = w.shape
        tailles.append((min(cin, max_canaux), min(cout, max_filtres)))

    n_col_max = max(c for _, c in tailles)
    hauteurs = [r + 0.6 for r, _ in tailles]            # + place pour le titre
    cote = 1.1                                          # pouces par vignette
    fig = plt.figure(figsize=(cote * n_col_max + 1.5, cote * sum(hauteurs) + 0.5),
                     layout="constrained")
    if titre:
        fig.suptitle(titre, fontsize=14)
    sous_figs = fig.subfigures(len(weights), 1, height_ratios=hauteurs)
    sous_figs = np.atleast_1d(sous_figs)

    for sf, (nom, w), (n_lig, n_col) in zip(sous_figs, weights.items(), tailles):
        kh, kw, cin, cout = w.shape
        m = np.abs(w).max() or 1.0
        tronque = " (tronqué)" if (cin > n_lig or cout > n_col) else ""
        sf.suptitle(f"{nom} — noyaux {kh}x{kw}, {cin} canal(aux) d'entrée, "
                    f"{cout} filtres{tronque}   |   échelle ±{m:.3f}", fontsize=10)

        axes = sf.subplots(n_lig, n_col, squeeze=False)
        for i in range(n_lig):
            for j in range(n_col):
                ax = axes[i, j]
                f = w[:, :, i, j]
                im = ax.imshow(f, cmap="RdBu", vmin=-m, vmax=m)   # rouge < 0 < bleu
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"filtre {j}", fontsize=8)
                if j == 0 and n_lig > 1:
                    ax.set_ylabel(f"canal {i}", fontsize=8)
                # Valeur écrite dans chaque case (seulement si lisible)
                if annoter and kh * kw <= 25:
                    for (y, x), v in np.ndenumerate(f):
                        ax.text(x, y, f"{v:.2f}", ha="center", va="center",
                                fontsize=7 if kh <= 3 else 5,
                                color="white" if abs(v) > 0.6 * m else "black")
        sf.colorbar(im, ax=axes, shrink=0.8, label="poids")

    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure enregistrée : figures/{name}.png")
    return weights