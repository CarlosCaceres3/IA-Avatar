"""La cara real de la persona, recortada sobre la cabeza del avatar.

Es la unica forma de poner "una persona real" en el stand sin problemas de
derechos de imagen: la cara es la de quien esta parado ahi, y esta ahi por
voluntad propia. Y para un stand funciona mejor que cualquier famoso,
porque cada visitante se ve a si mismo convertido en personaje.

No hace falta detector facial. El esqueleto ya calcula donde esta la
cabeza, su tamano y su inclinacion, y el avatar se dibuja con ESA misma
inclinacion: la cara del cuadro ya viene orientada como el avatar, asi que
basta recortarla, escalarla y pegarla con un borde suave.
"""

import cv2
import numpy as np

# Cuanto se agranda la cara respecto al circulo de la cabeza. El radio del
# avatar es estilizado (mas grande que una cabeza real), asi que sin este
# ajuste la cara quedaria flotando chica en medio del circulo.
GAIN = 1.16

# Ancho del degradado del borde, como fraccion del radio. Un recorte duro
# se ve como una calcomania pegada; difuminado se integra.
FEATHER = 0.16

# La cara es mas alta que ancha: el ovalo se estrecha en horizontal.
ASPECT = 1.14

_mask_cache = {}


def _oval_mask(size, feather=FEATHER):
    """Mascara ovalada con borde difuminado, cacheada por tamano."""
    key = (size, round(feather, 3))
    m = _mask_cache.get(key)
    if m is not None:
        return m

    r = size * 0.5
    ejes = np.arange(size, dtype=np.float32) - (size - 1) * 0.5
    dx = ejes[None, :] * ASPECT
    dy = ejes[:, None]
    dist = np.sqrt(dx * dx + dy * dy)

    borde = max(r * feather, 1.0)
    m = np.clip((r * 0.97 - dist) / borde, 0.0, 1.0).astype(np.float32)

    if len(_mask_cache) > 32:
        _mask_cache.clear()
    _mask_cache[key] = m
    return m


def paste(canvas, frame, head_c, head_r, q=1.0, gain=GAIN):
    """Pega la cara del cuadro sobre la cabeza del avatar.

    canvas: lienzo RGBA del avatar (puede estar a escala q del cuadro).
    frame:  cuadro BGR de la camara, a resolucion completa.
    head_c, head_r: cabeza en coordenadas del LIENZO.
    """
    ch, cw = canvas.shape[:2]
    R = float(head_r)
    if R < 4.0:
        return

    lado = int(R * 2.0)
    if lado < 8:
        return

    # El recorte se hace directo con una matriz afin: lleva el centro de la
    # cabeza del cuadro al centro del recuadro de salida, con la escala que
    # convierte el tamano real en el tamano del avatar.
    s = float(q) * float(gain)
    cx_frame = float(head_c[0]) / q
    cy_frame = float(head_c[1]) / q
    medio = (lado - 1) * 0.5

    M = np.array([[s, 0.0, medio - s * cx_frame],
                  [0.0, s, medio - s * cy_frame]], np.float32)
    cara = cv2.warpAffine(frame, M, (lado, lado), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

    # Recuadro de destino, recortado contra los bordes del lienzo.
    x0 = int(round(float(head_c[0]) - medio))
    y0 = int(round(float(head_c[1]) - medio))
    sx0, sy0 = max(-x0, 0), max(-y0, 0)
    dx0, dy0 = max(x0, 0), max(y0, 0)
    dx1, dy1 = min(x0 + lado, cw), min(y0 + lado, ch)
    if dx1 - dx0 < 2 or dy1 - dy0 < 2:
        return

    sub = cara[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    alpha = _oval_mask(lado)[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]

    roi = canvas[dy0:dy1, dx0:dx1]
    a = alpha[..., None]
    roi[:, :, :3] = (sub * a + roi[:, :, :3] * (1.0 - a)).astype(np.uint8)
    np.maximum(roi[:, :, 3], (alpha * 255.0).astype(np.uint8), out=roi[:, :, 3])
