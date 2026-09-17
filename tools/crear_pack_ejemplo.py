"""Genera un pack de ejemplo con PNGs, para mostrar como se conectan imagenes propias.

Crea assets/packs/ejemplo_imagenes/ con una pieza por parte del cuerpo.
Reemplacen esos archivos por sus propios dibujos manteniendo el nombre y el
avatar pasa a usarlos, sin tocar codigo.

Convencion de cada PNG:
  - Fondo transparente (canal alfa).
  - Para huesos (brazo, antebrazo, muslo, pantorrilla, torso): el eje largo
    va de ARRIBA (articulacion inicial) hacia ABAJO (articulacion final), y
    la pieza va centrada horizontalmente.
  - Para cabeza y mano: la pieza va centrada, mirando al frente.

    python tools/crear_pack_ejemplo.py
"""

import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "packs", "ejemplo_imagenes")

METAL = (150, 140, 130, 255)
DARK = (60, 55, 52, 255)
ACCENT = (40, 130, 240, 255)
LIGHT = (205, 200, 195, 255)


def rounded_piece(w, h, color, edge, radius_ratio=0.42, band=None):
    """Una pieza alargada con puntas redondeadas y una banda de color."""
    img = np.zeros((h, w, 4), np.uint8)
    r = int(min(w, h) * radius_ratio)
    cv2.rectangle(img, (0, r), (w, h - r), edge, -1, cv2.LINE_AA)
    cv2.circle(img, (w // 2, r), w // 2, edge, -1, cv2.LINE_AA)
    cv2.circle(img, (w // 2, h - r), w // 2, edge, -1, cv2.LINE_AA)

    inset = max(int(w * 0.11), 2)
    cv2.rectangle(img, (inset, r), (w - inset, h - r), color, -1, cv2.LINE_AA)
    cv2.circle(img, (w // 2, r), w // 2 - inset, color, -1, cv2.LINE_AA)
    cv2.circle(img, (w // 2, h - r), w // 2 - inset, color, -1, cv2.LINE_AA)

    if band:
        y = int(h * 0.62)
        cv2.rectangle(img, (inset, y), (w - inset, y + max(int(h * 0.12), 3)), band, -1)
    return img


def head_piece(size=320):
    img = np.zeros((size, size, 4), np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), int(size * 0.44), DARK, -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), int(size * 0.39), METAL, -1, cv2.LINE_AA)
    # Visor: una franja oscura horizontal con brillo azul.
    cv2.ellipse(img, (c, int(size * 0.47)), (int(size * 0.30), int(size * 0.13)),
                0, 0, 360, DARK, -1, cv2.LINE_AA)
    cv2.ellipse(img, (c, int(size * 0.47)), (int(size * 0.26), int(size * 0.09)),
                0, 0, 360, ACCENT, -1, cv2.LINE_AA)
    cv2.circle(img, (int(size * 0.40), int(size * 0.44)), int(size * 0.035),
               LIGHT, -1, cv2.LINE_AA)
    # Antena
    cv2.line(img, (c, int(size * 0.12)), (c, int(size * 0.03)), METAL, max(size // 40, 2),
             cv2.LINE_AA)
    cv2.circle(img, (c, int(size * 0.03)), int(size * 0.045), ACCENT, -1, cv2.LINE_AA)
    return img


def hand_piece(size=140):
    img = np.zeros((size, size, 4), np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), int(size * 0.44), DARK, -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), int(size * 0.36), LIGHT, -1, cv2.LINE_AA)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)

    pieces = {
        "torso": rounded_piece(240, 300, METAL, DARK, 0.30, band=ACCENT),
        "upper_arm": rounded_piece(90, 210, METAL, DARK, 0.45),
        "forearm": rounded_piece(76, 200, ACCENT, DARK, 0.45),
        "thigh": rounded_piece(120, 250, METAL, DARK, 0.42),
        "shin": rounded_piece(100, 240, ACCENT, DARK, 0.42),
        "head": head_piece(),
        "hand": hand_piece(),
    }
    for name, img in pieces.items():
        cv2.imwrite(os.path.join(OUT, name + ".png"), img)

    # 'skin' cubre las partes que no tienen PNG (aqui el cuello): conviene
    # ponerla en el tono del pack para que no desentone.
    #
    # 'parts' es opcional: las articulaciones se deducen solas del canal
    # alfa. Se declara solo para ajustar. Aqui se agranda la cabeza porque
    # el robot queda mejor con la cabeza grande, estilo caricatura.
    theme = {
        "name": "Robot PNG",
        "skin": "#8C8882",
        "suit": "#8C8882",
        "accent": "#F0822B",
        "outline": "#3C3734",
        "glow": "#2882F0",
        "glow_strength": 0.5,
        "draw_face": False,
        "parts": {
            "head": {"size": 1.25},
        },
    }
    with open(os.path.join(OUT, "theme.json"), "w", encoding="utf-8") as fh:
        json.dump(theme, fh, indent=2, ensure_ascii=False)

    print("pack creado en " + os.path.normpath(OUT))
    for name in sorted(pieces):
        print("  " + name + ".png")
    print("Reemplacen estos PNG por los suyos manteniendo el nombre.")


if __name__ == "__main__":
    main()
