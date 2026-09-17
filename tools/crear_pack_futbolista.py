"""Genera un pack de futbolista completo, con los colores que ustedes elijan.

Dibuja las ocho partes del cuerpo como PNGs: camiseta con numero, mangas,
pantaloneta, medias, botines, manos y cabeza. La cara se deja libre a
proposito para que el motor le pinte los ojos vivos encima.

    python tools/crear_pack_futbolista.py
    python tools/crear_pack_futbolista.py --nombre "Futbolista Azul" \
        --camiseta "#1E4FD8" --pantaloneta "#FFFFFF" --medias "#1E4FD8" \
        --numero 10 --piel "#8D5524"

Cada llamada crea una carpeta en assets/packs/, asi que pueden tener varios
equipos y cambiarlos en vivo con las teclas A y D.
"""

import argparse
import json
import os
import unicodedata

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hex_a_bgr(valor):
    valor = valor.lstrip("#")
    r, g, b = int(valor[0:2], 16), int(valor[2:4], 16), int(valor[4:6], 16)
    return (b, g, r)


def mezclar(color, hacia, t):
    return tuple(int(c + (h - c) * t) for c, h in zip(color, hacia))


def rgba(color, alpha=255):
    return (int(color[0]), int(color[1]), int(color[2]), alpha)


def sombrear(img, color, oscuro=0.30):
    """Oscurece el borde izquierdo de la pieza para darle volumen.

    Sin esto las piezas se ven planas, como recortes de cartulina. Una
    sombra basta para que en el proyector se lea como un cuerpo.
    """
    h, w = img.shape[:2]
    alpha = img[:, :, 3]
    rampa = np.linspace(1.0, 0.0, w, dtype=np.float32) ** 2
    sombra = np.array(mezclar(color, (0, 0, 0), oscuro), np.float32)
    base = img[:, :, :3].astype(np.float32)
    mezcla = base * (1 - rampa[None, :, None] * 0.55) + sombra * (rampa[None, :, None] * 0.55)
    img[:, :, :3] = np.where(alpha[:, :, None] > 0, mezcla, base).astype(np.uint8)
    return img


def pieza(w, h, color, borde, radio=0.45, margen=0.12):
    """Pieza alargada con puntas redondeadas y margen transparente alrededor.

    El margen es a proposito: las articulaciones se deducen del canal alfa,
    asi que el motor encuentra los extremos del dibujo aunque sobre espacio.
    """
    mx, my = int(w * margen), int(h * margen)
    img = np.zeros((h + 2 * my, w + 2 * mx, 4), np.uint8)
    x0, y0, x1, y1 = mx, my, mx + w, my + h
    r = int(min(w, h) * radio)

    for col, grosor in ((borde, int(max(w * 0.10, 3))), (color, 0)):
        gx0, gy0 = x0 - grosor, y0 - grosor
        gx1, gy1 = x1 + grosor, y1 + grosor
        gr = r + grosor
        cv2.rectangle(img, (gx0, gy0 + gr), (gx1, gy1 - gr), rgba(col), -1, cv2.LINE_AA)
        cv2.circle(img, ((gx0 + gx1) // 2, gy0 + gr), (gx1 - gx0) // 2, rgba(col), -1, cv2.LINE_AA)
        cv2.circle(img, ((gx0 + gx1) // 2, gy1 - gr), (gx1 - gx0) // 2, rgba(col), -1, cv2.LINE_AA)
    return sombrear(img, color)


def camiseta(color, borde, cuello, numero):
    w, h = 300, 360
    img = pieza(w, h, color, borde, radio=0.22, margen=0.10)
    ih, iw = img.shape[:2]
    cx = iw // 2

    # Cuello en V
    top = int(ih * 0.10)
    puntos = np.array([[cx - int(w * 0.16), top], [cx + int(w * 0.16), top],
                       [cx, top + int(h * 0.16)]], np.int32)
    cv2.fillConvexPoly(img, puntos, rgba(cuello), cv2.LINE_AA)

    # Numero grande al pecho: es lo que se lee de lejos en el stand.
    texto = str(numero)
    escala = 4.2
    grosor = 14
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, escala, grosor)
    org = (cx - tw // 2, int(ih * 0.62) + th // 2)
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_DUPLEX, escala,
                rgba(cuello), grosor, cv2.LINE_AA)
    return img


def cabeza(piel, pelo, size=340):
    """Cabeza con pelo y orejas, pero SIN cara.

    La cara la pinta el motor encima, con los ojos siguiendo a la persona.
    Una cara fija en el PNG se ve muerta; unos ojos que te miran, no.
    """
    img = np.zeros((size, size, 4), np.uint8)
    c = size // 2
    r = int(size * 0.38)
    borde = mezclar(piel, (0, 0, 0), 0.45)

    for lado in (-1, 1):
        cv2.circle(img, (c + lado * r, c + int(r * 0.10)), int(r * 0.20),
                   rgba(borde), -1, cv2.LINE_AA)
        cv2.circle(img, (c + lado * r, c + int(r * 0.10)), int(r * 0.14),
                   rgba(piel), -1, cv2.LINE_AA)

    cv2.circle(img, (c, c), r + int(size * 0.018), rgba(borde), -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), r, rgba(piel), -1, cv2.LINE_AA)

    # Pelo: casquete sobre la mitad superior.
    cv2.ellipse(img, (c, c - int(r * 0.30)), (r, int(r * 0.78)),
                0, 180, 360, rgba(pelo), -1, cv2.LINE_AA)
    cv2.ellipse(img, (c, c - int(r * 0.18)), (r, int(r * 0.42)),
                0, 180, 360, rgba(pelo), -1, cv2.LINE_AA)
    return img


def botin(color, borde):
    w, h = 130, 190
    img = pieza(w, h, color, borde, radio=0.48, margen=0.14)
    return img


def slug(texto):
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return "".join(ch if ch.isalnum() else "_" for ch in texto.lower()).strip("_")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nombre", default="Futbolista Rojo")
    p.add_argument("--camiseta", default="#D32F2F")
    p.add_argument("--pantaloneta", default="#FFFFFF")
    p.add_argument("--medias", default="#D32F2F")
    p.add_argument("--botines", default="#1A1A1A")
    p.add_argument("--piel", default="#C68642")
    p.add_argument("--pelo", default="#2B1B12")
    p.add_argument("--numero", default="7")
    args = p.parse_args()

    jersey = hex_a_bgr(args.camiseta)
    shorts = hex_a_bgr(args.pantaloneta)
    socks = hex_a_bgr(args.medias)
    boots = hex_a_bgr(args.botines)
    piel = hex_a_bgr(args.piel)
    pelo = hex_a_bgr(args.pelo)

    def borde_de(c):
        return mezclar(c, (0, 0, 0), 0.42)

    cuello = (255, 255, 255) if sum(jersey) < 420 else (30, 30, 30)

    piezas = {
        "torso": camiseta(jersey, borde_de(jersey), cuello, args.numero),
        "upper_arm": pieza(96, 210, jersey, borde_de(jersey)),
        "forearm": pieza(80, 195, piel, borde_de(piel)),
        "thigh": pieza(124, 240, shorts, borde_de(shorts)),
        "shin": pieza(104, 235, socks, borde_de(socks)),
        "hand": pieza(92, 96, piel, borde_de(piel), radio=0.5, margen=0.10),
        "foot": botin(boots, borde_de(boots)),
        "head": cabeza(piel, pelo),
    }

    carpeta = os.path.join(ROOT, "assets", "packs", slug(args.nombre))
    os.makedirs(carpeta, exist_ok=True)
    for nombre, img in piezas.items():
        cv2.imwrite(os.path.join(carpeta, nombre + ".png"), img)

    tema = {
        "name": args.nombre,
        "skin": args.piel,          # cuello y partes sin PNG
        "suit": args.camiseta,
        "accent": args.medias,
        "outline": "#2A2520",
        "glow": args.camiseta,
        "glow_strength": 0.35,
        "draw_face": True,          # los ojos los pinta el motor encima
        "parts": {
            "head": {"size": 1.30},
        },
    }
    with open(os.path.join(carpeta, "theme.json"), "w", encoding="utf-8") as fh:
        json.dump(tema, fh, indent=2, ensure_ascii=False)

    print("pack '" + args.nombre + "' creado en " + os.path.normpath(carpeta))
    print("  " + str(len(piezas)) + " piezas + theme.json")
    print("Revisalo con: python tools/probar_pack.py " + slug(args.nombre))


if __name__ == "__main__":
    main()
