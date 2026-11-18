import sys, inspect
import matplotlib
import matplotlib.pyplot as plt
import  numpy as np
from scipy import fft, stats
from utils import get_filter, getSymAntiSymTF, getDominantAngle
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from os import path

POS = [(0,0),(0,1),(1,0),(1,1),(0,2),(2,0),(1,2),(2,1),(2,2)]       # même ordre que random_full.ipynb
NAMES = ["dc","c1","c2","c3","c4","c5","c6","c7","c8"]
coords = np.array([-1, 0, 1])
XX, YY = np.meshgrid(coords, coords) 


def layer_stats(m, l):
    F = np.asarray(get_filter(m, l))                       # (3, 3, in, out)
    sym, anti = getSymAntiSymTF(F)
    sym, anti = np.asarray(sym), np.asarray(anti)
    nF, ns, na = [np.sqrt((A**2).sum((0, 1))) for A in (F, sym, anti)]   # (in, out)
    assert np.allclose(nF**2, ns**2 + na**2, atol=1e-4)
    return dict(
        F=F,
        beta=na / (nF + 1e-12),                            # ||anti|| / ||F||
        sigma=F.sum((0, 1)),                               # Σ (masse)
        fs_signed=ns * np.sign(F.sum((0, 1))),             # ||fs||·sign(Σ)
        gx=(F * XX[:, :, None, None]).sum((0, 1)),         # gradient horizontal
        gy=(F * YY[:, :, None, None]).sum((0, 1)),         # gradient vertical
        dct=fft.dctn(F, axes=(0, 1), norm="ortho"),        # coefficients DCT 3×3
    )

def plot_beta_hist(stats0, stats, N_LAYERS, name="beta_model2", titre="Modèle 2 — régression d'angle"):
    """Histogrammes de β = ||fa|| / ||F|| par couche, avant (init) et après entraînement.
    La figure est enregistrée dans figures/<name>.png."""
    fig, ax = plt.subplots(1, N_LAYERS, figsize=(4 * N_LAYERS, 3.2),
                           sharey=True, squeeze=False)   # squeeze=False : marche aussi avec 1 couche
    ax = ax[0]
    for l in range(N_LAYERS):
        b0 = np.ravel(stats0[l]["beta"])
        b = np.ravel(stats[l]["beta"])
        ax[l].hist(b0, bins=30, range=(0, 1), alpha=.5, label=f"init (moy {b0.mean():.2f})")
        ax[l].hist(b,  bins=30, range=(0, 1), alpha=.5, label=f"entraîné (moy {b.mean():.2f})")
        ax[l].axvline(b0.mean(), color="C0", ls="--", lw=1)
        ax[l].axvline(b.mean(),  color="C1", ls="--", lw=1)
        ax[l].set_title(f"Couche {l} : β moy = {b.mean():.2f}")
        ax[l].set_xlabel("β = ||anti|| / ||F||")
        ax[l].legend(fontsize=8)
    ax[0].set_ylabel("nombre de filtres")
    if titre:
        fig.suptitle(titre)
    fig.tight_layout()

    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure enregistrée : figures/{name}.png")