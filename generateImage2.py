"""
Jeu de données de RÉGRESSION — angle d'une demi-droite (images 32x32)

Chaque image contient une demi-droite qui part d'un point proche du centre
et va jusqu'au bord. L'angle est dans [0, 360[ : la demi-droite a un sens,
donc 0° (vers la droite) et 180° (vers la gauche) sont deux images différentes.

Convention d'angle (comme en maths, tel qu'on voit l'image) :
    0°   -> vers la droite
    90°  -> vers le haut
    180° -> vers la gauche
    270° -> vers le bas

Diversité :
  - angle uniforme dans [0, 360[
  - épaisseur variable (EPAISSEUR_MIN à EPAISSEUR_MAX px, valeurs non entières possibles)
  - origine légèrement décalée du centre (± ORIGINE_JITTER px)
  - bruit : bavure + sel et poivre (sauf pour PROBA_PROPRE des images)
  - inversion noir/blanc aléatoire

Sortie :
  dataset2/
    train/  images/00000.png, ...   metadata.json
    val/    images/00000.png, ...   metadata.json
    test/   images/00000.png, ...   metadata.json
    apercu.png

Utilisation :
  python generer_angles.py
  python generer_angles.py --n 10000 --split 60 20 20
"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

# ---------------- Paramètres ----------------
TAILLE = 32
N_IMAGES = 6000
EPAISSEUR_MIN, EPAISSEUR_MAX = 1.0, 4.0   # en pixels
ORIGINE_JITTER = 3                         # décalage max de l'origine par rapport au centre
SURECHANTILLONNAGE = 8                     # dessin en 256x256 puis réduction -> bords lissés
ANTICRENELAGE = True                       # False -> images strictement noir/blanc
PROBA_INVERSION = 0.5

# Bruit
PROBA_PROPRE = 0.3
BAVURE_MAX = 0.25
SEL_POIVRE_MAX = 0.03

# Découpage train / val / test (proportions, pas besoin que la somme fasse 1 ou 100)
SPLITS = ["train", "val", "test"]
PROPORTIONS = [70, 10, 20]

DOSSIER = Path("dataset2")
GRAINE = 0


# ---------------- Dessin ----------------
def demi_droite(angle_deg, epaisseur, ox, oy, taille=TAILLE):
    """Demi-droite blanche sur fond noir, partant de (ox, oy) dans la direction angle_deg.
    Dessinée en haute résolution puis réduite, pour que les épaisseurs non entières
    et les angles quelconques restent précis malgré les 32x32 pixels."""
    s = SURECHANTILLONNAGE
    grand = Image.new("L", (taille * s, taille * s), 0)
    d = ImageDraw.Draw(grand)

    a = math.radians(angle_deg)
    longueur = 2 * taille                       # assez long pour sortir de l'image
    # Centres des pixels : (x + 0.5) en coordonnées haute résolution
    x0, y0 = (ox + 0.5) * s, (oy + 0.5) * s
    x1 = x0 + longueur * s * math.cos(a)
    y1 = y0 - longueur * s * math.sin(a)        # axe y de l'image vers le bas -> signe moins
    d.line([(x0, y0), (x1, y1)], fill=255, width=max(1, round(epaisseur * s)))

    img = grand.resize((taille, taille), Image.LANCZOS)
    if not ANTICRENELAGE:
        img = img.point(lambda v: 255 if v >= 128 else 0)
    return img


# ---------------- Bruit ----------------
VOISINS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def bavure(img, p, rng):
    """Chaque pixel du trait (valeur >= 128) contamine, avec une proba p,
    un de ses 8 voisins tiré au hasard (le voisin prend sa valeur)."""
    src = img.load()
    out = img.copy()
    dst = out.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if src[x, y] >= 128 and rng.random() < p:
                dx, dy = rng.choice(VOISINS)
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    dst[nx, ny] = max(dst[nx, ny], src[x, y])
    return out


def sel_poivre(img, p, rng):
    out = img.copy()
    px = out.load()
    w, h = out.size
    for y in range(h):
        for x in range(w):
            if rng.random() < p:
                px[x, y] = 255 - px[x, y]
    return out


# ---------------- Création d'un exemple ----------------
def creer_exemple(rng):
    angle = rng.uniform(0, 360)
    epaisseur = round(rng.uniform(EPAISSEUR_MIN, EPAISSEUR_MAX), 2)
    centre = (TAILLE - 1) / 2                   # 15.5 pour 32 px
    ox = round(centre + rng.uniform(-ORIGINE_JITTER, ORIGINE_JITTER), 2)
    oy = round(centre + rng.uniform(-ORIGINE_JITTER, ORIGINE_JITTER), 2)

    img = demi_droite(angle, epaisseur, ox, oy)

    if rng.random() < PROBA_PROPRE:
        p_bav, p_sp = 0.0, 0.0
    else:
        p_bav = round(rng.uniform(0, BAVURE_MAX), 3)
        p_sp = round(rng.uniform(0, SEL_POIVRE_MAX), 3)
        img = bavure(img, p_bav, rng)
        img = sel_poivre(img, p_sp, rng)

    inverse = rng.random() < PROBA_INVERSION
    if inverse:
        img = ImageOps.invert(img)

    a = math.radians(angle)
    infos = {
        "angle_deg": round(angle, 3),
        "angle_rad": round(a, 5),
        "cos": round(math.cos(a), 5),           # cibles conseillées pour la régression :
        "sin": round(math.sin(a), 5),           # pas de saut entre 359° et 0°
        "epaisseur": epaisseur,
        "origine_x": ox,
        "origine_y": oy,
        "bavure": p_bav,
        "sel_poivre": p_sp,
        "inverse": int(inverse),
    }
    return img, infos


# ---------------- Génération ----------------
def tailles_split(n, proportions):
    total = sum(proportions)
    tailles = [int(n * p / total) for p in proportions]
    tailles[0] += n - sum(tailles)
    return tailles


def generer(n, proportions):
    if len(proportions) != len(SPLITS):
        raise ValueError(f"Il faut {len(SPLITS)} proportions (train, val, test)")
    rng = random.Random(GRAINE)
    if DOSSIER.exists():
        shutil.rmtree(DOSSIER)       # repart de zéro pour ne pas mélanger deux générations

    # Les angles sont tirés uniformément pour chaque image :
    # chaque split couvre donc tout le cercle [0, 360[.
    for s, n_split in zip(SPLITS, tailles_split(n, proportions)):
        dossier_split = DOSSIER / s
        (dossier_split / "images").mkdir(parents=True)

        exemples = []
        for i in range(n_split):
            img, infos = creer_exemple(rng)
            fichier = f"images/{i:05d}.png"      # chemin relatif au dossier du split
            img.save(dossier_split / fichier)
            exemples.append({"fichier": fichier, **infos})

        metadata = {
            "description": "Régression de l'angle d'une demi-droite",
            "split": s,
            "convention_angle": "degrés, 0 = droite, 90 = haut, sens anti-horaire, dans [0, 360[",
            "taille": TAILLE,
            "n_images": n_split,
            "parametres": {
                "epaisseur": [EPAISSEUR_MIN, EPAISSEUR_MAX],
                "origine_jitter": ORIGINE_JITTER,
                "anticrenelage": ANTICRENELAGE,
                "proba_inversion": PROBA_INVERSION,
                "proba_propre": PROBA_PROPRE,
                "bavure_max": BAVURE_MAX,
                "sel_poivre_max": SEL_POIVRE_MAX,
                "graine": GRAINE,
            },
            "images": exemples,
        }
        with open(dossier_split / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"{s:>5} : {n_split:5d} images  ->  {dossier_split}/")


# ---------------- Aperçu ----------------
def apercu(n_col=8, zoom=4, marge=4, graine=1):
    """Ligne 1 : angles 0, 45, ..., 315 sans bruit.
    Lignes 2-3 : exemples tirés au hasard (épaisseur, bruit, inversion)."""
    rng = random.Random(graine)
    cote = TAILLE * zoom
    W = n_col * (cote + marge) + marge
    H = 3 * (cote + marge) + marge
    mosaique = Image.new("L", (W, H), 128)
    c = (TAILLE - 1) / 2
    for col in range(n_col):
        imgs = [demi_droite(col * 360 / n_col, 2, c, c),
                creer_exemple(rng)[0],
                creer_exemple(rng)[0]]
        for lig, img in enumerate(imgs):
            img = img.resize((cote, cote), Image.NEAREST)
            mosaique.paste(img, (marge + col * (cote + marge), marge + lig * (cote + marge)))
    mosaique.save(DOSSIER / "apercu.png")
    print(f"Aperçu : {DOSSIER / 'apercu.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jeu de régression : angle d'une demi-droite")
    parser.add_argument("--n", type=int, default=N_IMAGES, help="nombre total d'images")
    parser.add_argument("--split", type=float, nargs=3, default=PROPORTIONS,
                        metavar=("TRAIN", "VAL", "TEST"))
    args = parser.parse_args()
    generer(args.n, args.split)
    apercu()