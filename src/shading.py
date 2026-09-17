"""Volumen: extremidades como cilindros y cabeza como esfera.

Lo que hace que un muneco se vea plano no es que este dibujado en 2D, es
que cada parte es un color liso. Una figura con sombreado se lee como un
cuerpo aunque el dibujo siga siendo plano.

Aqui cada parte se pinta como si fuera un cilindro o una esfera de verdad,
con luz difusa, brillo especular y luz de contorno.

Sobre el rendimiento, que aqui es todo el problema: la version directa
(calcular la normal y las potencias pixel a pixel, con np.mgrid) costaba
80 ms por cuadro en 720p, o sea 12 FPS. Dos cambios lo bajan a un digito:

  1. Todo el sombreado de un cilindro cabe en una tabla de 256 colores,
     porque la normal solo cambia a lo ancho, no a lo largo. Las potencias
     se evaluan 256 veces por parte en vez de 300.000.
  2. Las coordenadas se generan como vectores de una dimension que se
     difunden entre si, en vez de dos matrices completas con np.mgrid.

La luz por defecto viene de arriba a la izquierda y algo adelante, que es
la posicion clasica de retrato y la que mejor define un cuerpo.
"""

import cv2
import numpy as np

# Direccion de la luz en 3D (x derecha, y abajo, z hacia la camara).
LIGHT = np.array([-0.45, -0.55, 0.70], dtype=np.float32)
LIGHT /= np.linalg.norm(LIGHT)

# Vector a media distancia entre luz y camara, para el brillo especular.
HALF = LIGHT + np.array([0.0, 0.0, 1.0], np.float32)
HALF /= np.linalg.norm(HALF)

AMBIENT = 0.42        # cuanto se ve la zona en sombra
SPEC_POWER = 22.0     # que tan concentrado es el brillo
SPEC_GAIN = 0.55
RIM_GAIN = 0.34       # luz de contorno: despega la figura del fondo

LUT_N = 256
_TS = np.linspace(-1.0, 1.0, LUT_N, dtype=np.float32)
_HALF_SCALE = np.float32(0.5 * (LUT_N - 1))

# Las tablas se cachean por (direccion redondeada, color): en una escena
# tipica hay cuatro o cinco combinaciones distintas y se repiten cuadro a
# cuadro mientras la persona no gire.
_CACHE = {}
_CACHE_MAX = 512


def _shade(n_x, n_y, n_z, color, rim_color):
    """Color final para un conjunto de normales. Se usa solo en las tablas."""
    difusa = np.clip(n_x * LIGHT[0] + n_y * LIGHT[1] + n_z * LIGHT[2], 0.0, 1.0)
    luz = AMBIENT + (1.0 - AMBIENT) * difusa

    especular = np.power(
        np.clip(n_x * HALF[0] + n_y * HALF[1] + n_z * HALF[2], 0.0, 1.0),
        SPEC_POWER) * SPEC_GAIN
    contorno = np.power(np.clip(1.0 - n_z, 0.0, 1.0), 3.0) * RIM_GAIN

    base = np.asarray(color, np.float32).reshape(-1, 3)
    rim = np.asarray(rim_color, np.float32).reshape(-1, 3)
    rgb = base * luz[..., None] + 255.0 * especular[..., None] + rim * contorno[..., None]
    return np.clip(rgb, 0, 255).astype(np.float32)


def _cylinder_lut(nx, ny, color, rim_color):
    """Tabla de 256 colores para un cilindro iluminado en una direccion."""
    clave = (round(float(nx), 2), round(float(ny), 2),
             tuple(int(c) for c in color), tuple(int(c) for c in rim_color))
    tabla = _CACHE.get(clave)
    if tabla is not None:
        return tabla

    t = _TS
    n_z = np.sqrt(np.clip(1.0 - t * t, 0.0, 1.0))
    tabla = _shade(nx * t, ny * t, n_z, color, rim_color)

    if len(_CACHE) > _CACHE_MAX:
        _CACHE.clear()
    _CACHE[clave] = tabla
    return tabla


def _sphere_lut(color, rim_color, n=64):
    """Tabla 2D chica para una esfera: se estira al tamano que haga falta.

    La esfera si necesita dos dimensiones (la normal cambia en todas las
    direcciones), pero una tabla de 64x64 estirada es indistinguible del
    calculo exacto y cuesta una fraccion.
    """
    clave = ("esf", tuple(int(c) for c in color), tuple(int(c) for c in rim_color), n)
    tabla = _CACHE.get(clave)
    if tabla is not None:
        return tabla

    eje = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    nx = eje[None, :]
    ny = eje[:, None]
    r2 = nx * nx + ny * ny
    n_z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

    plano = _shade(np.broadcast_to(nx, (n, n)).ravel(),
                   np.broadcast_to(ny, (n, n)).ravel(),
                   n_z.ravel(), color, rim_color)
    tabla = plano.reshape(n, n, 3)

    if len(_CACHE) > _CACHE_MAX:
        _CACHE.clear()
    _CACHE[clave] = tabla
    return tabla


def _bbox(cx0, cy0, cx1, cy1, w, h):
    x0 = max(int(np.floor(cx0)), 0)
    y0 = max(int(np.floor(cy0)), 0)
    x1 = min(int(np.ceil(cx1)) + 1, w)
    y1 = min(int(np.ceil(cy1)) + 1, h)
    return (x0, y0, x1, y1) if (x1 > x0 and y1 > y0) else None


def _blend(canvas, box, rgb, cover):
    x0, y0, x1, y1 = box
    roi = canvas[y0:y1, x0:x1]
    a = cover[..., None]
    roi[:, :, :3] = (rgb * a + roi[:, :, :3] * (1.0 - a)).astype(np.uint8)
    np.maximum(roi[:, :, 3], (cover * 255.0).astype(np.uint8), out=roi[:, :, 3])


def capsule(canvas, p0, p1, w0, w1, color, rim_color=(255, 255, 255)):
    """Extremidad con volumen de cilindro y puntas redondeadas."""
    h, w = canvas.shape[:2]
    p0 = np.asarray(p0, np.float32)
    p1 = np.asarray(p1, np.float32)
    w0, w1 = np.float32(w0), np.float32(w1)
    wmax = float(max(w0, w1))

    box = _bbox(min(p0[0], p1[0]) - wmax - 1, min(p0[1], p1[1]) - wmax - 1,
                max(p0[0], p1[0]) + wmax + 1, max(p0[1], p1[1]) + wmax + 1, w, h)
    if box is None:
        return
    x0, y0, x1, y1 = box

    d = p1 - p0
    largo = np.float32(np.linalg.norm(d))
    if largo < 1e-3:
        return
    u = d / largo

    # Coordenadas como vectores 1D que se difunden: evita materializar dos
    # matrices enteras por parte, que era el grueso del costo.
    vx = np.arange(x0, x1, dtype=np.float32)[None, :] - p0[0]
    vy = np.arange(y0, y1, dtype=np.float32)[:, None] - p0[1]

    # Proyeccion sobre el eje, recortada a los extremos: eso convierte el
    # cilindro en capsula (puntas redondeadas).
    s = np.clip((vx * u[0] + vy * u[1]) * (np.float32(1.0) / largo), 0.0, 1.0)
    dx = vx - d[0] * s
    dy = vy - d[1] * s
    dist = np.sqrt(dx * dx + dy * dy)

    radio = np.maximum(w0 + (w1 - w0) * s, np.float32(0.5))
    cover = np.clip(radio - dist + np.float32(0.5), 0.0, 1.0)
    if not cover.any():
        return

    # Distancia al eje CON signo: de que lado del cilindro cae el pixel.
    # Ese es directamente el indice de la tabla de color.
    t = np.clip((vx * -u[1] + vy * u[0]) / radio, -1.0, 1.0)
    idx = ((t + np.float32(1.0)) * _HALF_SCALE).astype(np.int32)

    _blend(canvas, box, _cylinder_lut(-u[1], u[0], color, rim_color)[idx], cover)


def sphere(canvas, center, radius, color, rim_color=(255, 255, 255)):
    """Cabeza, manos y articulaciones con volumen de esfera."""
    h, w = canvas.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    radius = float(radius)

    box = _bbox(cx - radius - 1, cy - radius - 1, cx + radius + 1, cy + radius + 1, w, h)
    if box is None:
        return
    x0, y0, x1, y1 = box

    vx = np.arange(x0, x1, dtype=np.float32)[None, :] - np.float32(cx)
    vy = np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(cy)
    dist = np.sqrt(vx * vx + vy * vy)

    cover = np.clip(np.float32(radius) - dist + np.float32(0.5), 0.0, 1.0)
    if not cover.any():
        return

    tabla = _sphere_lut(color, rim_color)
    n = tabla.shape[0]
    escala = np.float32((n - 1) * 0.5 / max(radius, 0.5))
    ix = np.clip((vx * escala + (n - 1) * 0.5), 0, n - 1).astype(np.int32)
    iy = np.clip((vy * escala + (n - 1) * 0.5), 0, n - 1).astype(np.int32)

    _blend(canvas, box, tabla[iy, ix], cover)


def ground_shadow(canvas, points, scale, strength=0.45):
    """Mancha oscura bajo los pies: apoya la figura en el piso.

    Sin sombra de contacto, un personaje con volumen igual parece flotar.
    """
    h, w = canvas.shape[:2]
    validos = [p for p in points if p is not None]
    if not validos:
        return
    cx = float(np.mean([p[0] for p in validos]))
    cy = float(max(p[1] for p in validos))

    rx = int(max(scale * 0.95, 4))
    ry = int(max(scale * 0.22, 3))
    box = _bbox(cx - rx * 1.6, cy - ry * 1.6, cx + rx * 1.6, cy + ry * 1.6, w, h)
    if box is None:
        return
    x0, y0, x1, y1 = box

    capa = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.ellipse(capa, (int(cx) - x0, int(cy) - y0), (rx, ry), 0, 0, 360, 255, -1)
    capa = cv2.GaussianBlur(capa, (0, 0), max(rx * 0.22, 2.0))

    a = (capa.astype(np.float32) / 255.0 * strength)[..., None]
    roi = canvas[y0:y1, x0:x1]
    roi[:, :, :3] = (roi[:, :, :3] * (1.0 - a)).astype(np.uint8)
    np.maximum(roi[:, :, 3], (a[..., 0] * 255.0).astype(np.uint8), out=roi[:, :, 3])
