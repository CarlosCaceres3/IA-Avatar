"""Genera tres avatares de dominio publico: Bolivar, Quijote y Frankenstein.

Son personajes libres de derechos, a diferencia de cualquier famoso vivo.
Las piezas se dibujan por codigo, asi que tampoco dependen de ilustraciones
de terceros: el pack entero es original.

    python tools/crear_packs_historicos.py

Cada uno queda en assets/packs/ y aparece con las teclas A y D.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from crear_pack_futbolista import hex_a_bgr, mezclar, pieza, rgba, slug, sombrear


def borde_de(c):
    return mezclar(c, (0, 0, 0), 0.42)


def cabeza_base(piel, size=340):
    """Craneo y orejas, sin cara: el motor pinta los ojos encima."""
    img = np.zeros((size, size, 4), np.uint8)
    c = size // 2
    r = int(size * 0.37)
    borde = borde_de(piel)
    for lado in (-1, 1):
        cv2.circle(img, (c + lado * r, c + int(r * 0.10)), int(r * 0.20),
                   rgba(borde), -1, cv2.LINE_AA)
        cv2.circle(img, (c + lado * r, c + int(r * 0.10)), int(r * 0.14),
                   rgba(piel), -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), r + int(size * 0.018), rgba(borde), -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), r, rgba(piel), -1, cv2.LINE_AA)
    return img, c, r


def cabeza_bolivar(piel, pelo, size=340):
    """Pelo ondulado con patillas marcadas, el rasgo mas reconocible."""
    img, c, r = cabeza_base(piel, size)
    cv2.ellipse(img, (c, c - int(r * 0.26)), (r, int(r * 0.80)),
                0, 180, 360, rgba(pelo), -1, cv2.LINE_AA)
    # Mechones a los lados y patillas bajando hasta la mandibula
    for lado in (-1, 1):
        cv2.ellipse(img, (c + lado * int(r * 0.80), c - int(r * 0.10)),
                    (int(r * 0.30), int(r * 0.52)), lado * 14, 0, 360,
                    rgba(pelo), -1, cv2.LINE_AA)
        cv2.ellipse(img, (c + lado * int(r * 0.72), c + int(r * 0.30)),
                    (int(r * 0.15), int(r * 0.34)), 0, 0, 360,
                    rgba(pelo), -1, cv2.LINE_AA)
    return img


def cabeza_quijote(piel, metal, barba, size=340):
    """Yelmo tipo bacia y barba larga en punta."""
    img, c, r = cabeza_base(piel, size)
    # Barba: triangulo redondeado bajo la cara
    barba_pts = np.array([[c - int(r * 0.62), c + int(r * 0.22)],
                          [c + int(r * 0.62), c + int(r * 0.22)],
                          [c, c + int(r * 1.55)]], np.int32)
    cv2.fillConvexPoly(img, barba_pts, rgba(barba), cv2.LINE_AA)
    cv2.ellipse(img, (c, c + int(r * 0.34)), (int(r * 0.55), int(r * 0.34)),
                0, 0, 180, rgba(barba), -1, cv2.LINE_AA)
    # Bigote
    cv2.ellipse(img, (c, c + int(r * 0.30)), (int(r * 0.42), int(r * 0.16)),
                0, 0, 180, rgba(mezclar(barba, (255, 255, 255), 0.25)), -1, cv2.LINE_AA)
    # Yelmo: casquete metalico con ala, como una bacia de barbero invertida
    cv2.ellipse(img, (c, c - int(r * 0.30)), (int(r * 1.02), int(r * 0.78)),
                0, 180, 360, rgba(borde_de(metal)), -1, cv2.LINE_AA)
    cv2.ellipse(img, (c, c - int(r * 0.34)), (int(r * 0.92), int(r * 0.70)),
                0, 180, 360, rgba(metal), -1, cv2.LINE_AA)
    cv2.ellipse(img, (c, c - int(r * 0.30)), (int(r * 1.16), int(r * 0.16)),
                0, 0, 360, rgba(metal), -1, cv2.LINE_AA)
    cv2.ellipse(img, (c, c - int(r * 0.34)), (int(r * 1.16), int(r * 0.13)),
                0, 180, 360, rgba(mezclar(metal, (255, 255, 255), 0.35)), -1, cv2.LINE_AA)
    return img


def cabeza_frankenstein(piel, pelo, perno, size=340):
    """Craneo plano arriba y pernos en el cuello: la silueta clasica."""
    img = np.zeros((size, size, 4), np.uint8)
    c = size // 2
    r = int(size * 0.37)
    borde = borde_de(piel)

    # Pernos, uno a cada lado a la altura del cuello
    for lado in (-1, 1):
        x = c + lado * int(r * 1.22)
        y = c + int(r * 0.62)
        cv2.rectangle(img, (min(x, c), y - int(r * 0.17)),
                      (max(x, c), y + int(r * 0.17)), rgba(borde_de(perno)), -1)
        cv2.circle(img, (x, y), int(r * 0.26), rgba(borde_de(perno)), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), int(r * 0.20), rgba(perno), -1, cv2.LINE_AA)
        cv2.circle(img, (x - lado * int(r * 0.06), y - int(r * 0.06)),
                   int(r * 0.07), rgba(mezclar(perno, (255, 255, 255), 0.5)),
                   -1, cv2.LINE_AA)

    # Cabeza: rectangulo de esquinas redondeadas, plana arriba
    x0, x1 = c - r, c + r
    y0, y1 = c - int(r * 0.95), c + r
    cv2.rectangle(img, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), rgba(borde), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), rgba(piel), -1, cv2.LINE_AA)
    cv2.circle(img, (c, y1 - int(r * 0.30)), int(r * 0.95), rgba(piel), -1, cv2.LINE_AA)

    # Flequillo negro recto
    cv2.rectangle(img, (x0, y0), (x1, y0 + int(r * 0.42)), rgba(pelo), -1)
    for i in range(5):
        px = x0 + int((x1 - x0) * (i + 0.5) / 5.0)
        cv2.circle(img, (px, y0 + int(r * 0.42)), int(r * 0.13), rgba(pelo), -1, cv2.LINE_AA)
    return img


def guardar(nombre, piezas, tema):
    carpeta = os.path.join(ROOT, "assets", "packs", slug(nombre))
    os.makedirs(carpeta, exist_ok=True)
    for parte, img in piezas.items():
        cv2.imwrite(os.path.join(carpeta, parte + ".png"), img)
    with open(os.path.join(carpeta, "theme.json"), "w", encoding="utf-8") as fh:
        json.dump(tema, fh, indent=2, ensure_ascii=False)
    print("  " + nombre + " -> " + str(len(piezas)) + " piezas")


def bolivar():
    piel = hex_a_bgr("#C68642")
    azul = hex_a_bgr("#1B2A5E")
    oro = hex_a_bgr("#D4A62A")
    blanco = hex_a_bgr("#E8E4DC")
    negro = hex_a_bgr("#1A1A1A")

    torso = pieza(300, 350, azul, borde_de(azul), radio=0.22, margen=0.10)
    ih, iw = torso.shape[:2]
    cx = iw // 2
    # Banda roja cruzada y botonadura dorada: lo que lo hace reconocible
    cv2.line(torso, (int(iw * 0.22), int(ih * 0.28)), (int(iw * 0.78), int(ih * 0.70)),
             rgba(hex_a_bgr("#9E2B2B")), int(iw * 0.13), cv2.LINE_AA)
    for i in range(4):
        cv2.circle(torso, (cx, int(ih * (0.26 + i * 0.13))), int(iw * 0.035),
                   rgba(oro), -1, cv2.LINE_AA)
    # Charreteras
    for lado in (-1, 1):
        cv2.ellipse(torso, (cx + lado * int(iw * 0.34), int(ih * 0.20)),
                    (int(iw * 0.14), int(ih * 0.05)), 0, 0, 360, rgba(oro), -1, cv2.LINE_AA)

    brazo = pieza(96, 210, azul, borde_de(azul))
    antebrazo = pieza(82, 195, azul, borde_de(azul))
    ah, aw = antebrazo.shape[:2]
    cv2.rectangle(antebrazo, (int(aw * 0.12), int(ah * 0.74)),
                  (int(aw * 0.88), int(ah * 0.86)), rgba(oro), -1)

    guardar("Bolivar", {
        "torso": torso,
        "upper_arm": brazo,
        "forearm": antebrazo,
        "thigh": pieza(124, 240, blanco, borde_de(blanco)),
        "shin": pieza(104, 235, negro, borde_de(negro)),
        "hand": pieza(92, 96, piel, borde_de(piel), radio=0.5, margen=0.10),
        "foot": pieza(130, 190, negro, borde_de(negro), radio=0.48, margen=0.14),
        "head": cabeza_bolivar(piel, hex_a_bgr("#241812")),
    }, {
        "name": "Bolivar", "skin": "#C68642", "suit": "#1B2A5E", "accent": "#D4A62A",
        "outline": "#141026", "glow": "#D4A62A", "glow_strength": 0.35,
        "draw_face": True, "parts": {"head": {"size": 1.22}},
    })


def quijote():
    piel = hex_a_bgr("#D2A679")
    metal = hex_a_bgr("#A8ADB5")
    cuero = hex_a_bgr("#5A3E28")
    oscuro = hex_a_bgr("#2E2A26")

    torso = pieza(290, 345, metal, borde_de(metal), radio=0.26, margen=0.10)
    ih, iw = torso.shape[:2]
    cx = iw // 2
    # Peto: quilla central y remaches
    cv2.line(torso, (cx, int(ih * 0.18)), (cx, int(ih * 0.86)),
             rgba(mezclar(metal, (255, 255, 255), 0.45)), max(int(iw * 0.03), 2), cv2.LINE_AA)
    for lado in (-1, 1):
        for i in range(3):
            cv2.circle(torso, (cx + lado * int(iw * 0.28), int(ih * (0.30 + i * 0.18))),
                       int(iw * 0.028), rgba(borde_de(metal)), -1, cv2.LINE_AA)

    guardar("Quijote", {
        "torso": torso,
        "upper_arm": pieza(96, 210, metal, borde_de(metal)),
        "forearm": pieza(82, 195, metal, borde_de(metal)),
        "thigh": pieza(120, 240, cuero, borde_de(cuero)),
        "shin": pieza(102, 235, metal, borde_de(metal)),
        "hand": pieza(92, 96, metal, borde_de(metal), radio=0.5, margen=0.10),
        "foot": pieza(130, 190, oscuro, borde_de(oscuro), radio=0.48, margen=0.14),
        "head": cabeza_quijote(piel, metal, hex_a_bgr("#7A6E5E")),
    }, {
        "name": "Quijote", "skin": "#D2A679", "suit": "#A8ADB5", "accent": "#A8ADB5",
        "outline": "#3A362F", "glow": "#C8CEDA", "glow_strength": 0.40,
        "draw_face": True, "parts": {"head": {"size": 1.30}},
    })


def frankenstein():
    verde = hex_a_bgr("#6E8F5A")
    chaqueta = hex_a_bgr("#2B2B33")
    pantalon = hex_a_bgr("#5E5E4C")
    bota = hex_a_bgr("#2A2A2E")
    perno = hex_a_bgr("#9AA0A6")

    torso = pieza(310, 345, chaqueta, borde_de(chaqueta), radio=0.18, margen=0.10)
    ih, iw = torso.shape[:2]
    cx = iw // 2
    # Solapas de chaqueta y costuras
    for lado in (-1, 1):
        pts = np.array([[cx, int(ih * 0.12)],
                        [cx + lado * int(iw * 0.24), int(ih * 0.14)],
                        [cx, int(ih * 0.46)]], np.int32)
        cv2.fillConvexPoly(torso, pts, rgba(mezclar(chaqueta, (255, 255, 255), 0.18)),
                           cv2.LINE_AA)
    for i in range(6):
        y = int(ih * (0.55 + i * 0.05))
        cv2.line(torso, (int(iw * 0.30), y), (int(iw * 0.36), y),
                 rgba(mezclar(chaqueta, (255, 255, 255), 0.35)), 2, cv2.LINE_AA)

    antebrazo = pieza(88, 195, verde, borde_de(verde))
    ah, aw = antebrazo.shape[:2]
    cv2.line(antebrazo, (int(aw * 0.28), int(ah * 0.30)), (int(aw * 0.62), int(ah * 0.34)),
             rgba(borde_de(verde)), 2, cv2.LINE_AA)

    guardar("Frankenstein", {
        "torso": torso,
        "upper_arm": pieza(104, 210, chaqueta, borde_de(chaqueta)),
        "forearm": antebrazo,
        "thigh": pieza(132, 240, pantalon, borde_de(pantalon)),
        "shin": pieza(114, 235, pantalon, borde_de(pantalon)),
        "hand": pieza(98, 100, verde, borde_de(verde), radio=0.5, margen=0.10),
        "foot": pieza(150, 200, bota, borde_de(bota), radio=0.40, margen=0.14),
        "head": cabeza_frankenstein(verde, hex_a_bgr("#17171A"), perno),
    }, {
        "name": "Frankenstein", "skin": "#6E8F5A", "suit": "#2B2B33", "accent": "#6E8F5A",
        "outline": "#16161A", "glow": "#5A8F4A", "glow_strength": 0.40,
        "draw_face": True, "parts": {"head": {"size": 1.24}},
    })


if __name__ == "__main__":
    print("Personajes de dominio publico (sin derechos de imagen):")
    bolivar()
    quijote()
    frankenstein()
    print("Revisalos con: python tools/probar_pack.py")
