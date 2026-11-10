"""
Étape 1 — Génération des images d'entraînement (motifs noir et blanc 32x32)

Classes :
  0 = bandes horizontales
  1 = bandes verticales
  2 = damier

Diversité :
  - bandes : largeur du trait (L) et espacement (E) tirés indépendamment
      L = E      -> bandes régulières 50/50
      L = 1, E = 9 -> un trait fin tous les 10 px (motif très asymétrique)
  - damier : cases rectangulaires (largeur cx ≠ hauteur cy possible)
  - phase aléatoire (décalage) pour chaque image
  - bruit (sauf pour PROBA_PROPRE des images) :
      * bavure : chaque pixel du trait contamine un de ses 8 voisins avec une proba p
      * sel et poivre : quelques pixels isolés inversés
  - inversion aléatoire noir/blanc (trait noir sur fond blanc)

Sortie :
  dataset/
    train/ val/ test/
      bandes_h/xxxxx.png, bandes_v/xxxxx.png, damier/xxxxx.png
      labels.csv   (fichier, classe, label, largeur, espace, periode,
                    phase_x, phase_y, bavure, sel_poivre, inverse)
    apercu.png, apercu_bruit.png

Utilisation :
  python generer_motifs.py                     # 70 / 10 / 20 par défaut
  python generer_motifs.py --split 60 20 20
  python generer_motifs.py --split 60 20 20 --n 2000   # 2000 images par classe
"""

import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

# ---------------- Paramètres ----------------
TAILLE = 32
N_PAR_CLASSE = 1000
LARGEUR_MIN, LARGEUR_MAX = 1, 8      # largeur du trait (bandes) / taille de case (damier)
ESPACE_MIN, ESPACE_MAX = 1, 14       # espace entre deux traits
PERIODE_MAX = 20                     # pour garder au moins ~1,5 période visible dans 32 px
PROBA_INVERSION = 0.5                # proportion d'images en trait noir sur fond blanc

# Bruit
PROBA_PROPRE = 0.3                   # part des images laissées sans aucun bruit
BAVURE_MAX = 0.25                    # proba max qu'un pixel du trait contamine un voisin
SEL_POIVRE_MAX = 0.03                # proba max qu'un pixel quelconque soit inversé

SPLITS = ["train", "val", "test"]
PROPORTIONS = [70, 10, 20]           
DOSSIER = Path("dataset")
GRAINE = 0

CLASSES = {"bandes_h": 0, "bandes_v": 1, "damier": 2}


def bandes(orientation, largeur, espace, phase, taille=TAILLE):
    img = Image.new("L", (taille, taille), 0)
    d = ImageDraw.Draw(img)
    periode = largeur + espace
    for debut in range(phase - periode, taille, periode):
        fin = debut + largeur - 1
        if orientation == "h":
            d.rectangle([0, debut, taille - 1, fin], fill=255)
        else:
            d.rectangle([debut, 0, fin, taille - 1], fill=255)
    return img


def damier(cx, cy, phase_x, phase_y, taille=TAILLE):
    """Damier de cases cx (largeur) x cy (hauteur), décalé de (phase_x, phase_y)."""
    img = Image.new("L", (taille, taille), 0)
    d = ImageDraw.Draw(img)
    for i in range(taille // cx + 3):
        for j in range(taille // cy + 3):
            if (i + j) % 2 == 0:
                x0 = i * cx - phase_x
                y0 = j * cy - phase_y
                d.rectangle([x0, y0, x0 + cx - 1, y0 + cy - 1], fill=255)
    return img


# ---------------- Bruit ----------------
VOISINS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def bavure(img, p, rng):
    """Pour chaque pixel blanc (le trait), avec une probabilité p,
    un de ses 8 voisins tiré au hasard devient blanc lui aussi.
    La contamination est calculée sur l'image d'origine (pas d'effet boule de neige)."""
    src = img.load()
    out = img.copy()
    dst = out.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if src[x, y] == 255 and rng.random() < p:
                dx, dy = rng.choice(VOISINS)
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    dst[nx, ny] = 255
    return out


def sel_poivre(img, p, rng):
    """Chaque pixel est inversé (noir <-> blanc) avec une probabilité p."""
    out = img.copy()
    px = out.load()
    w, h = out.size
    for y in range(h):
        for x in range(w):
            if rng.random() < p:
                px[x, y] = 255 - px[x, y]
    return out


# ---------------- Tirage des paramètres ----------------
def tirer_bandes(rng):
    while True:
        L = rng.randint(LARGEUR_MIN, LARGEUR_MAX)
        E = rng.randint(ESPACE_MIN, ESPACE_MAX)
        if L + E <= PERIODE_MAX:
            return L, E


def tirer_damier(rng):
    cx = rng.randint(LARGEUR_MIN, LARGEUR_MAX)
    # moitié des damiers carrés, moitié rectangulaires
    cy = cx if rng.random() < 0.5 else rng.randint(LARGEUR_MIN, LARGEUR_MAX)
    return cx, cy


def creer_motif(nom, rng):
    """Renvoie (image, dict de paramètres)."""
    if nom in ("bandes_h", "bandes_v"):
        L, E = tirer_bandes(rng)
        phase = rng.randrange(L + E)
        orient = "h" if nom == "bandes_h" else "v"
        img = bandes(orient, L, E, phase)
        p = dict(largeur=L, espace=E, periode=L + E,
                 phase_x=phase if orient == "v" else 0,
                 phase_y=phase if orient == "h" else 0)
    else:
        cx, cy = tirer_damier(rng)
        px, py = rng.randrange(2 * cx), rng.randrange(2 * cy)
        img = damier(cx, cy, px, py)
        # pour le damier : largeur = cx, espace = cy (dimensions de la case)
        p = dict(largeur=cx, espace=cy, periode=f"{2*cx}x{2*cy}",
                 phase_x=px, phase_y=py)

    # Bruit appliqué AVANT l'inversion : le blanc est alors toujours le trait,
    # donc la bavure étale bien le trait, qu'il finisse noir ou blanc.
    if rng.random() < PROBA_PROPRE:
        p_bav, p_sp = 0.0, 0.0
    else:
        p_bav = round(rng.uniform(0, BAVURE_MAX), 3)
        p_sp = round(rng.uniform(0, SEL_POIVRE_MAX), 3)
        img = bavure(img, p_bav, rng)
        img = sel_poivre(img, p_sp, rng)
    p["bavure"] = p_bav
    p["sel_poivre"] = p_sp

    inverse = rng.random() < PROBA_INVERSION
    if inverse:
        img = ImageOps.invert(img)
    p["inverse"] = int(inverse)
    return img, p


# ---------------- Génération du jeu de données ----------------
def tailles_split(n, proportions):
    """Répartit n images selon les proportions (ex. [70, 10, 20]).
    Les arrondis sont corrigés pour que la somme fasse exactement n."""
    total = sum(proportions)
    tailles = [int(n * p / total) for p in proportions]
    tailles[0] += n - sum(tailles)        # le reste va dans train
    return tailles


def generer(proportions=PROPORTIONS):
    if len(proportions) != len(SPLITS):
        raise ValueError(f"Il faut {len(SPLITS)} proportions (train, val, test)")
    rng = random.Random(GRAINE)

    champs = ["fichier", "classe", "label", "largeur", "espace", "periode",
              "phase_x", "phase_y", "bavure", "sel_poivre", "inverse"]
    lignes = {s: [] for s in SPLITS}
    for s in SPLITS:
        for nom in CLASSES:
            (DOSSIER / s / nom).mkdir(parents=True, exist_ok=True)

    # Découpage stratifié : chaque split a la même proportion de chaque classe
    tailles = tailles_split(N_PAR_CLASSE, proportions)
    for nom, label in CLASSES.items():
        k = 0
        for s, n in zip(SPLITS, tailles):
            for _ in range(n):
                img, p = creer_motif(nom, rng)
                fichier = f"{nom}/{k:05d}.png"          # chemin relatif au dossier du split
                img.save(DOSSIER / s / fichier)
                lignes[s].append(dict(fichier=fichier, classe=nom, label=label, **p))
                k += 1

    for s in SPLITS:
        with open(DOSSIER / s / "labels.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=champs)
            w.writeheader()
            w.writerows(lignes[s])

    for s, n in zip(SPLITS, tailles):
        print(f"{s:>5} : {n * len(CLASSES):5d} images  ({n} par classe)")


# ---------------- Aperçu visuel ----------------
def apercu(n_col=10, zoom=4, marge=4, graine=1):
    """Mosaïque : une ligne par classe, n_col exemples tirés au hasard."""
    rng = random.Random(graine)
    cote = TAILLE * zoom
    W = n_col * (cote + marge) + marge
    H = len(CLASSES) * (cote + marge) + marge
    mosaique = Image.new("L", (W, H), 128)
    for r, nom in enumerate(CLASSES):
        for c in range(n_col):
            img, _ = creer_motif(nom, rng)
            img = img.resize((cote, cote), Image.NEAREST)
            mosaique.paste(img, (marge + c * (cote + marge), marge + r * (cote + marge)))
    mosaique.save(DOSSIER / "apercu.png")
    print(f"Aperçu : {DOSSIER / 'apercu.png'}")


def apercu_bruit(niveaux=(0, 0.05, 0.1, 0.15, 0.25, 0.4), zoom=4, marge=4, graine=2):
    """Un même motif par ligne, avec une bavure croissante de gauche à droite."""
    rng = random.Random(graine)
    motifs = [bandes("h", 1, 7, 3), bandes("v", 3, 3, 0), damier(4, 4, 0, 0)]
    cote = TAILLE * zoom
    W = len(niveaux) * (cote + marge) + marge
    H = len(motifs) * (cote + marge) + marge
    mosaique = Image.new("L", (W, H), 128)
    for r, m in enumerate(motifs):
        for c, p in enumerate(niveaux):
            img = bavure(m, p, rng).resize((cote, cote), Image.NEAREST)
            mosaique.paste(img, (marge + c * (cote + marge), marge + r * (cote + marge)))
    mosaique.save(DOSSIER / "apercu_bruit.png")
    print(f"Aperçu du bruit : {DOSSIER / 'apercu_bruit.png'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Génère le jeu de motifs train/val/test")
    parser.add_argument("--split", type=float, nargs=3, default=PROPORTIONS,
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="proportions, ex. --split 60 20 20 (défaut : 70 10 20)")
    parser.add_argument("--n", type=int, default=N_PAR_CLASSE,
                        help="nombre total d'images par classe")
    args = parser.parse_args()
    N_PAR_CLASSE = args.n

    if DOSSIER.exists():
        shutil.rmtree(DOSSIER)        
    generer(args.split)
    apercu()
    apercu_bruit()