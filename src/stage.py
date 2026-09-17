"""Escenario: fondo, brillo y composicion final.

El avatar llega como RGBA transparente. Aqui se pega sobre un fondo y se
le agrega un halo de luz difuminado. El halo es lo que hace que el avatar
se lea bien de lejos en un proyector, que es la condicion real de un stand.
"""

import os

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Fondos generados por codigo (no necesitan ningun archivo)
# --------------------------------------------------------------------------

def _vertical_gradient(w, h, top, bottom):
    ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    top = np.array(top, np.float32).reshape(1, 1, 3)
    bottom = np.array(bottom, np.float32).reshape(1, 1, 3)
    return np.broadcast_to(top * (1.0 - ramp) + bottom * ramp, (h, w, 3)).copy()


def _vignette(img, strength=0.55):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.5
    d = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    mask = np.clip(1.0 - strength * (d / 1.4142) ** 2, 0.0, 1.0)[:, :, None]
    return img * mask


def bg_estudio(w, h):
    img = _vertical_gradient(w, h, (70, 34, 18), (18, 10, 8))
    return _vignette(img, 0.45)


def bg_atardecer(w, h):
    img = _vertical_gradient(w, h, (150, 90, 235), (40, 120, 250))
    return _vignette(img, 0.35)


def bg_rejilla(w, h):
    img = _vertical_gradient(w, h, (40, 18, 12), (10, 6, 6))
    step = max(h // 14, 12)
    line = np.array((120, 60, 30), np.float32)
    for y in range(0, h, step):
        img[y:y + 1, :] += line
    for x in range(0, w, step):
        img[:, x:x + 1] += line
    # El horizonte claro da sensacion de escenario con profundidad.
    glow = cv2.GaussianBlur(img, (0, 0), 9)
    return _vignette(np.clip(img * 0.75 + glow * 0.5, 0, 255), 0.5)


def bg_blanco(w, h):
    return _vertical_gradient(w, h, (250, 248, 245), (215, 212, 208))


PROCEDURAL_BACKGROUNDS = [
    ("Estudio", bg_estudio),
    ("Atardecer", bg_atardecer),
    ("Rejilla", bg_rejilla),
    ("Blanco", bg_blanco),
]


class Backgrounds:
    """Fondos generados + imagenes de assets/backgrounds + camara en vivo.

    Los fondos se calculan una sola vez y quedan cacheados: generarlos en
    cada cuadro costaria mas que todo el render del avatar.
    """

    CAMERA = "Camara"

    def __init__(self, folder, size):
        self.size = size
        self.items = [(name, fn) for name, fn in PROCEDURAL_BACKGROUNDS]

        if os.path.isdir(folder):
            for fname in sorted(os.listdir(folder)):
                if os.path.splitext(fname)[1].lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                    continue
                self.items.append((os.path.splitext(fname)[0], os.path.join(folder, fname)))

        self.items.append((self.CAMERA, None))
        self.index = 0
        self._cache = {}

    @property
    def name(self):
        return self.items[self.index][0]

    def next(self, step=1):
        self.index = (self.index + step) % len(self.items)
        return self.name

    def get(self, frame):
        """Devuelve el fondo BGR del cuadro actual."""
        name, source = self.items[self.index]
        w, h = self.size

        if source is None:                       # camara real, atenuada
            bg = cv2.GaussianBlur(frame, (0, 0), 8)
            return (bg.astype(np.float32) * 0.45).astype(np.uint8)

        if self.index in self._cache:
            return self._cache[self.index]

        if callable(source):
            bg = np.clip(source(w, h), 0, 255).astype(np.uint8)
        else:
            img = cv2.imread(source, cv2.IMREAD_COLOR)
            bg = _fit_cover(img, w, h) if img is not None else np.zeros((h, w, 3), np.uint8)

        self._cache[self.index] = bg
        return bg


def _fit_cover(img, w, h):
    """Escala y recorta la imagen para llenar el lienzo sin deformarla."""
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    resized = cv2.resize(img, (int(np.ceil(iw * scale)), int(np.ceil(ih * scale))),
                         interpolation=cv2.INTER_AREA)
    y0 = (resized.shape[0] - h) // 2
    x0 = (resized.shape[1] - w) // 2
    return resized[y0:y0 + h, x0:x0 + w].copy()


# --------------------------------------------------------------------------
# Composicion
# --------------------------------------------------------------------------

def composite(background, avatar_rgba, glow_color=(255, 190, 40), glow_strength=0.0,
              roi=None, out=None):
    """Pega el avatar sobre el fondo, con halo opcional.

    Todo se hace con operaciones uint8 de OpenCV (vectorizadas en SIMD) en
    vez de aritmetica float de numpy: la version float tardaba 75 ms por
    cuadro en 720p, que por si sola dejaba la demo en 13 FPS.

    roi limita el trabajo a la caja donde realmente hay avatar. El resto
    del cuadro es fondo puro y no necesita mezclarse.
    """
    h, w = background.shape[:2]

    if out is None:
        out = background.copy()
    else:
        np.copyto(out, background)

    x0, y0, x1, y1 = roi if roi else (0, 0, w, h)
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), w), min(int(y1), h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return out

    alpha = avatar_rgba[y0:y1, x0:x1, 3]
    rgb = avatar_rgba[y0:y1, x0:x1, :3]
    dst = out[y0:y1, x0:x1]
    rh, rw = dst.shape[:2]

    if glow_strength > 0.01:
        # Se difumina y se colorea a 1/4 de resolucion: el halo es suave por
        # definicion, asi que nadie nota la diferencia y cuesta 16 veces menos.
        sw, sh = max(rw // 4, 1), max(rh // 4, 1)
        small = cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, (0, 0), 7)
        halo_small = cv2.merge([
            cv2.convertScaleAbs(blurred, alpha=glow_strength * glow_color[i] / 255.0)
            for i in range(3)
        ])
        halo = cv2.resize(halo_small, (rw, rh), interpolation=cv2.INTER_LINEAR)
        cv2.add(dst, halo, dst)

    a3 = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    inv = cv2.bitwise_not(a3)
    cv2.add(cv2.multiply(rgb, a3, scale=1.0 / 255.0),
            cv2.multiply(dst, inv, scale=1.0 / 255.0), dst)
    return out


# --------------------------------------------------------------------------
# Texto en pantalla
# --------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(img, x0, y0, x1, y1, alpha=0.5, color=(18, 14, 12)):
    """Rectangulo oscuro translucido detras del texto.

    Se usa un panel y no un contorno alrededor de las letras porque en
    OpenCV el avance de cada caracter depende del grosor: dibujar el mismo
    texto en negro grueso y encima en blanco fino deja la version negra
    corrida hacia la derecha, con la cola asomando.
    """
    h, w = img.shape[:2]
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), w), min(int(y1), h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(roi, 1.0 - alpha, np.full_like(roi, color, dtype=np.uint8),
                    alpha, 0.0, roi)


def _text(img, txt, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_hud(img, lines, corner=(16, 30), scale=0.6):
    thick = max(int(round(scale * 2)), 1)
    step = int(30 * scale / 0.6)
    widths = [cv2.getTextSize(line, FONT, scale, thick)[0][0] for line in lines]
    pad = int(10 * scale / 0.6)

    _panel(img, corner[0] - pad, corner[1] - int(22 * scale / 0.6),
           corner[0] + max(widths) + pad, corner[1] + step * (len(lines) - 1) + pad)

    y = corner[1]
    for line in lines:
        _text(img, line, (corner[0], y), scale, (255, 255, 255), thick)
        y += step
    return img


def draw_banner(img, title, subtitle=None):
    """Mensaje grande y centrado: se usa cuando no hay nadie frente a la camara."""
    h, w = img.shape[:2]
    scale = w / 900.0
    t_scale, t_thick = 1.4 * scale, max(int(round(3 * scale)), 2)
    s_scale, s_thick = 0.75 * scale, max(int(round(2 * scale)), 1)

    (tw, th), _ = cv2.getTextSize(title, FONT, t_scale, t_thick)
    sw = cv2.getTextSize(subtitle, FONT, s_scale, s_thick)[0][0] if subtitle else 0

    top = (h - th) // 2
    box_w = max(tw, sw) + int(70 * scale)
    box_h = th + (int(70 * scale) if subtitle else int(30 * scale))
    _panel(img, (w - box_w) // 2, top - int(28 * scale),
           (w + box_w) // 2, top + box_h, alpha=0.55)

    _text(img, title, ((w - tw) // 2, top + th), t_scale, (255, 255, 255), t_thick)
    if subtitle:
        _text(img, subtitle, ((w - sw) // 2, top + th + int(48 * scale)),
              s_scale, (205, 205, 205), s_thick)
    return img
