"""Render de poses sinteticas, sin camara.

Sirve para ajustar proporciones y colores del avatar sin tener que pararse
frente a la camara cada vez. Genera una hoja de contacto en preview.png.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import avatar as av          # noqa: E402
import skeleton as sk        # noqa: E402
import stage                 # noqa: E402


class LM:
    def __init__(self, x, y, z=0.0, v=0.95):
        self.x, self.y, self.z, self.visibility = x, y, z, v


def pose(points, missing_vis=0.05):
    """Construye 33 landmarks a partir de un diccionario indice -> (x, y)."""
    out = []
    for i in range(33):
        if i in points:
            x, y = points[i]
            out.append(LM(x, y, 0.0, 0.95))
        else:
            out.append(LM(0.5, 0.5, 0.0, missing_vis))
    return out


DE_PIE = {
    sk.NOSE: (.50, .155), sk.L_EYE: (.525, .145), sk.R_EYE: (.475, .145),
    sk.L_EAR: (.545, .160), sk.R_EAR: (.455, .160),
    sk.L_SHOULDER: (.590, .300), sk.R_SHOULDER: (.410, .300),
    sk.L_ELBOW: (.640, .450), sk.R_ELBOW: (.360, .450),
    sk.L_WRIST: (.665, .600), sk.R_WRIST: (.335, .600),
    sk.L_INDEX: (.670, .640), sk.R_INDEX: (.330, .640),
    sk.L_HIP: (.555, .570), sk.R_HIP: (.445, .570),
    sk.L_KNEE: (.565, .750), sk.R_KNEE: (.435, .750),
    sk.L_ANKLE: (.570, .925), sk.R_ANKLE: (.430, .925),
    sk.L_FOOT: (.605, .960), sk.R_FOOT: (.395, .960),
}

BRAZOS_ARRIBA = dict(DE_PIE)
BRAZOS_ARRIBA.update({
    sk.L_ELBOW: (.680, .270), sk.R_ELBOW: (.320, .270),
    sk.L_WRIST: (.735, .110), sk.R_WRIST: (.265, .110),
    sk.L_INDEX: (.740, .070), sk.R_INDEX: (.260, .070),
})

CAMINANDO = dict(DE_PIE)
CAMINANDO.update({
    sk.L_KNEE: (.620, .735), sk.R_KNEE: (.410, .760),
    sk.L_ANKLE: (.665, .900), sk.R_ANKLE: (.360, .930),
    sk.L_FOOT: (.705, .925), sk.R_FOOT: (.325, .960),
    sk.L_ELBOW: (.620, .470), sk.R_ELBOW: (.375, .430),
    sk.L_WRIST: (.590, .620), sk.R_WRIST: (.360, .560),
})

# Media persona: simula a alguien sentado o muy cerca de la camara.
MEDIO_CUERPO = {k: v for k, v in DE_PIE.items()
                if k not in (sk.L_KNEE, sk.R_KNEE, sk.L_ANKLE, sk.R_ANKLE,
                             sk.L_FOOT, sk.R_FOOT)}

POSES = [("De pie", DE_PIE), ("Brazos arriba", BRAZOS_ARRIBA),
         ("Caminando", CAMINANDO), ("Medio cuerpo", MEDIO_CUERPO)]


def main():
    w, h = 480, 640
    renderer = av.AvatarRenderer()
    packs = av.load_packs(os.path.join(os.path.dirname(__file__), "..", "assets", "packs"))
    backgrounds = stage.Backgrounds("", (w, h))

    rows = []
    for pack in packs:
        cells = []
        for pose_name, points in POSES:
            skel = sk.from_landmarks(pose(points), w, h)
            canvas = renderer.render(skel, pack)
            bg = backgrounds.get(np.zeros((h, w, 3), np.uint8))
            frame = stage.composite(bg, canvas, pack.theme.glow, pack.theme.glow_strength)
            stage.draw_hud(frame, [pack.name + " / " + pose_name], scale=0.5)
            cells.append(frame)
        rows.append(np.hstack(cells))

    sheet = np.vstack(rows)
    out = os.path.join(os.path.dirname(__file__), "..", "preview.png")
    cv2.imwrite(out, sheet)
    print("preview:", sheet.shape, "->", os.path.normpath(out))


if __name__ == "__main__":
    main()
