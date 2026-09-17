"""Revisa un pack de imagenes y lo muestra en varias poses.

Sirve para iterar rapido con las ilustraciones: avisa que partes faltan,
detecta los problemas tipicos y genera una hoja con el avatar en cuatro
poses para ver como quedo, sin tener que pararse frente a la camara.

    python tools/probar_pack.py                     # revisa todos los packs
    python tools/probar_pack.py mi_personaje        # revisa uno
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cv2
import numpy as np

import avatar as av
import skeleton as sk
import stage
from preview_poses import POSES, pose

PACKS_DIR = os.path.join(ROOT, "assets", "packs")

# Partes que mas se notan si faltan, en orden de importancia.
IMPORTANTES = ["head", "torso", "upper_arm", "forearm", "thigh", "shin"]
OPCIONALES = ["hand", "foot"]


def revisar(pack):
    """Avisa de los problemas tipicos al preparar las imagenes."""
    avisos = []

    if not pack.parts:
        avisos.append("ERROR  no hay ningun PNG: el avatar saldra con figuras")
        return avisos

    faltan = [p for p in IMPORTANTES if pack.part(p) is None]
    if faltan:
        avisos.append("falta " + ", ".join(faltan) +
                      " -> esas partes se dibujan con figuras del tema")

    sin_falta = [p for p in OPCIONALES if pack.part(p) is None]
    if sin_falta:
        avisos.append("sin " + ", ".join(sin_falta) + " (opcional, se dibuja una figura)")

    for name, img in sorted(pack.parts.items()):
        alpha = img[:, :, 3]
        opacos = int((alpha > 8).sum())
        total = alpha.size

        if opacos == 0:
            avisos.append("ERROR  " + name + ".png esta completamente transparente")
            continue
        if opacos == total:
            avisos.append("OJO    " + name + ".png no tiene transparencia: " +
                          "se vera como un rectangulo sobre el fondo")
        if min(img.shape[:2]) < 40:
            avisos.append("OJO    " + name + ".png es muy chico (" +
                          str(img.shape[1]) + "x" + str(img.shape[0]) +
                          "): se vera pixelado en el proyector")

        spec = pack.specs.get(name, av.DEFAULT_SPEC)
        if name not in ("head", "hand", "foot"):
            largo = abs(spec.b[1] - spec.a[1])
            if largo < 0.25:
                avisos.append("OJO    " + name + ".png: el dibujo ocupa poco alto; " +
                              "revisa que este orientado de arriba a abajo")

    return avisos


def hoja(pack, w=420, h=560):
    renderer = av.AvatarRenderer()
    bgs = stage.Backgrounds("", (w, h))
    bg = bgs.get(np.zeros((h, w, 3), np.uint8))

    cells = []
    for nombre, puntos in POSES:
        skel = sk.from_landmarks(pose(puntos), w, h)
        canvas = renderer.render(skel, pack)
        frame = stage.composite(bg, canvas, pack.theme.glow, pack.theme.glow_strength)
        stage.draw_hud(frame, [nombre], scale=0.45)
        cells.append(frame)
    return np.hstack(cells)


def main():
    pedido = sys.argv[1] if len(sys.argv) > 1 else None

    packs = [p for p in av.load_packs(PACKS_DIR) if p.parts]
    if pedido:
        packs = [p for p in packs
                 if pedido.lower() in p.name.lower()]
        if not packs:
            carpeta = os.path.join(PACKS_DIR, pedido)
            if os.path.isdir(carpeta):
                packs = [av.AvatarPack.load(carpeta)]
            else:
                print("No encuentro el pack '" + pedido + "' en " + PACKS_DIR)
                return 1

    if not packs:
        print("No hay packs con imagenes en " + PACKS_DIR)
        print("Crea uno con: python tools/crear_pack_ejemplo.py")
        return 1

    filas = []
    for pack in packs:
        print("== " + pack.name + " ==")
        print("  partes: " + (", ".join(sorted(pack.parts)) if pack.parts else "ninguna"))
        avisos = revisar(pack)
        if avisos:
            for a in avisos:
                print("  " + a)
        else:
            print("  todo en orden")

        for name in sorted(pack.parts):
            s = pack.specs.get(name, av.DEFAULT_SPEC)
            print("    " + name.ljust(11) +
                  " articulaciones a=(" + str(round(s.a[0], 2)) + "," + str(round(s.a[1], 2)) +
                  ") b=(" + str(round(s.b[0], 2)) + "," + str(round(s.b[1], 2)) + ")" +
                  ("  size=" + str(round(s.size, 2)) if abs(s.size - 1.0) > 0.01 else ""))
        filas.append(hoja(pack))
        print()

    salida = os.path.join(ROOT, "pack_revision.png")
    cv2.imwrite(salida, np.vstack(filas))
    print("vista guardada en " + os.path.normpath(salida))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
