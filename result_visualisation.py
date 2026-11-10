import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
from pathlib import Path


# matplotlib.use("TkAgg")
def save_image(fig, name):
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure enregistrée : figures/{name}.png")

def printLossAcc(hist, n_epoch, prefix=""):
    xc = range(1, n_epoch + 1)
    h = hist.history
 
    for metrique, legende_loc in (("loss", "upper right"), ("accuracy", "lower right")):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(xc, h[metrique], marker="o", label="train")
        ax.plot(xc, h[f"val_{metrique}"], marker="o", label="val")
        ax.set_xlabel("Époque")
        ax.set_ylabel(metrique)
        ax.set_title(f"train vs val — {metrique}")
        ax.set_xticks(list(xc))
        ax.grid(True)
        ax.legend(loc=legende_loc)
        save_image(fig, f"{prefix}{metrique}")

def plot_confusion_matrix(cm, classes, normalize=False,
                          title="Confusion matrix", cmap=plt.cm.Blues,
                          name="confusion_matrix"):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"
 
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks, classes, rotation=45)
    ax.set_yticks(ticks, classes)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
 
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
 
    fig.tight_layout()
    save_image(fig, name)