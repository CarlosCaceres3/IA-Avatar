"""Render del avatar articulado.

El avatar se dibuja sobre un lienzo RGBA propio (fondo transparente). Asi
el compositor puede aplicarle brillo y pegarlo sobre cualquier fondo sin
que el borde del muneco arrastre pixeles del fondo real.

Cada parte del cuerpo se puede dibujar de dos formas:

  1. Procedural: figuras geometricas con el color del tema. Funciona sin
     ningun archivo y es el modo por defecto.
  2. Con imagen: si el pack tiene un PNG para esa parte, la imagen se
     deforma sobre el hueso (rotada y escalada) en vez de la figura.

Asi ustedes pueden meter sus propios PNGs en assets/packs/<nombre>/ y el
avatar pasa a usarlos sin tocar codigo.
"""

import json
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

import shading
from skeleton import (
    L_ANKLE, L_EAR, L_ELBOW, L_FOOT, L_HIP, L_INDEX, L_KNEE, L_SHOULDER, L_WRIST,
    NOSE, R_ANKLE, R_EAR, R_ELBOW, R_FOOT, R_HIP, R_INDEX, R_KNEE, R_SHOULDER,
    R_WRIST,
)

# Grosor de cada parte como fraccion de la escala del cuerpo.
# (ancho en el extremo inicial, ancho en el extremo final)
PROPORTIONS = {
    "upper_arm": (0.115, 0.098),
    "forearm": (0.098, 0.072),
    "thigh": (0.165, 0.125),
    "shin": (0.125, 0.082),
    "neck": (0.120, 0.120),
}
HAND_R = 0.085
FOOT_R = 0.085

# Tope de grosor de una extremidad como fraccion de su propio largo en 2D.
# Evita el "muñon" cuando un hueso apunta a la camara y se ve muy corto.
FORESHORTEN_CAP = 0.62


@dataclass
class Theme:
    name: str
    suit: tuple            # color principal del traje (BGR)
    skin: tuple            # cabeza y manos
    accent: tuple          # antebrazos y pantorrillas
    outline: tuple         # contorno
    glow: tuple            # color del halo
    glow_strength: float = 0.0
    outline_w: float = 0.022   # fraccion de la escala
    draw_face: bool = True
    volume: bool = False       # sombrear como cilindros y esferas (3D)
    rim: tuple = (255, 255, 255)   # color de la luz de contorno


THEMES = [
    Theme(
        name="Neon",
        suit=(90, 30, 25), skin=(220, 180, 120), accent=(255, 210, 60),
        outline=(255, 245, 210), glow=(255, 190, 40), glow_strength=1.0,
    ),
    Theme(
        name="Robot",
        suit=(150, 140, 130), skin=(190, 185, 180), accent=(40, 130, 240),
        outline=(45, 40, 38), glow=(40, 130, 240), glow_strength=0.45,
    ),
    Theme(
        name="Cartoon",
        suit=(70, 90, 235), skin=(150, 200, 250), accent=(60, 190, 250),
        outline=(30, 28, 32), glow=(60, 190, 250), glow_strength=0.0,
        outline_w=0.030,
    ),
    Theme(
        name="Holograma",
        suit=(210, 160, 70), skin=(235, 205, 130), accent=(245, 235, 180),
        outline=(255, 250, 235), glow=(255, 200, 90), glow_strength=1.3,
        draw_face=False,
    ),
    Theme(
        name="Oro",
        suit=(40, 140, 220), skin=(120, 200, 245), accent=(90, 215, 250),
        outline=(20, 45, 80), glow=(60, 180, 240), glow_strength=0.6,
    ),

    # Temas con volumen: mismas siluetas, pero sombreadas como cilindros y
    # esferas. Es lo que separa un muneco de palos de un personaje.
    Theme(
        name="3D Azul",
        suit=(190, 105, 45), skin=(150, 190, 225), accent=(225, 165, 80),
        outline=(40, 28, 20), glow=(200, 120, 50), glow_strength=0.30,
        outline_w=0.016, volume=True, rim=(255, 225, 180),
    ),
    Theme(
        name="3D Heroe",
        suit=(55, 55, 205), skin=(150, 190, 235), accent=(60, 185, 240),
        outline=(25, 20, 45), glow=(70, 90, 225), glow_strength=0.35,
        outline_w=0.016, volume=True, rim=(220, 240, 255),
    ),
    Theme(
        name="3D Robot",
        suit=(165, 160, 150), skin=(200, 198, 195), accent=(50, 140, 245),
        outline=(38, 34, 30), glow=(60, 150, 245), glow_strength=0.30,
        outline_w=0.016, volume=True, rim=(235, 245, 255),
    ),
    Theme(
        name="3D Oro",
        suit=(45, 150, 230), skin=(150, 205, 240), accent=(90, 200, 250),
        outline=(18, 45, 78), glow=(60, 175, 240), glow_strength=0.45,
        outline_w=0.016, volume=True, rim=(190, 240, 255),
    ),
]


# --------------------------------------------------------------------------
# Primitivas de dibujo sobre lienzo RGBA
# --------------------------------------------------------------------------

def _pt(p):
    return (int(round(float(p[0]))), int(round(float(p[1]))))


def _perp(u):
    return np.array([-u[1], u[0]], dtype=np.float32)


def _rgba(color, alpha=255):
    return (int(color[0]), int(color[1]), int(color[2]), int(alpha))


def tapered(canvas, p0, p1, w0, w1, color):
    """Extremidad como poligono que se adelgaza, con puntas redondeadas."""
    d = np.asarray(p1, np.float32) - np.asarray(p0, np.float32)
    length = float(np.linalg.norm(d))
    if length < 1e-3:
        return
    n = _perp(d / length)
    quad = np.array([p0 + n * w0, p1 + n * w1, p1 - n * w1, p0 - n * w0], np.int32)
    cv2.fillConvexPoly(canvas, quad, color, cv2.LINE_AA)
    cv2.circle(canvas, _pt(p0), max(int(w0), 1), color, -1, cv2.LINE_AA)
    cv2.circle(canvas, _pt(p1), max(int(w1), 1), color, -1, cv2.LINE_AA)


def limb(canvas, p0, p1, w0, w1, fill, outline, ow, theme=None):
    """Extremidad con contorno: se pinta primero el borde, luego el relleno.

    Con el tema en modo volumen, el relleno liso se cambia por un cilindro
    sombreado: misma silueta, pero se lee como un cuerpo y no como un palo.
    """
    if ow > 0.4:
        tapered(canvas, p0, p1, w0 + ow, w1 + ow, outline)
    if theme is not None and theme.volume:
        shading.capsule(canvas, p0, p1, w0, w1, fill[:3], theme.rim)
    else:
        tapered(canvas, p0, p1, w0, w1, fill)


def blob(canvas, center, radius, fill, outline, ow, theme=None):
    if ow > 0.4:
        cv2.circle(canvas, _pt(center), max(int(radius + ow), 1), outline, -1, cv2.LINE_AA)
    if theme is not None and theme.volume:
        shading.sphere(canvas, center, radius, fill[:3], theme.rim)
    else:
        cv2.circle(canvas, _pt(center), max(int(radius), 1), fill, -1, cv2.LINE_AA)


def warp_rgba(canvas, rgba, M):
    """Pega una imagen RGBA transformada por M, componiendo solo en su caja.

    Trabajar sobre la caja envolvente y no sobre todo el lienzo es lo que
    permite pegar 10 PNGs por cuadro sin perder FPS.
    """
    h, w = rgba.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    tc = corners @ M[:, :2].T + M[:, 2]

    x0 = max(int(np.floor(tc[:, 0].min())) - 1, 0)
    y0 = max(int(np.floor(tc[:, 1].min())) - 1, 0)
    x1 = min(int(np.ceil(tc[:, 0].max())) + 2, canvas.shape[1])
    y1 = min(int(np.ceil(tc[:, 1].max())) + 2, canvas.shape[0])
    if x1 <= x0 or y1 <= y0:
        return

    M2 = M.copy()
    M2[0, 2] -= x0
    M2[1, 2] -= y0
    warped = cv2.warpAffine(rgba, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    a = warped[:, :, 3:4].astype(np.float32) / 255.0
    roi = canvas[y0:y1, x0:x1]
    roi[:, :, :3] = (warped[:, :, :3] * a + roi[:, :, :3] * (1.0 - a)).astype(np.uint8)
    roi[:, :, 3:4] = np.maximum(roi[:, :, 3:4], warped[:, :, 3:4])


def warp_on_bone(canvas, rgba, p0, p1, anchor_a=(0.5, 0.0), anchor_b=(0.5, 1.0),
                 width_gain=1.0):
    """Coloca un PNG sobre un hueso alineando sus articulaciones.

    anchor_a y anchor_b dicen DONDE estan las dos articulaciones dentro de
    la imagen, en coordenadas de 0 a 1. Por defecto se asume borde superior
    centrado y borde inferior centrado, que es como estaba antes.

    Poder declararlas es lo que permite usar dibujos de verdad: un brazo
    ilustrado trae contorno, sombra y holgura alrededor, y su articulacion
    casi nunca cae justo en el borde de la imagen. Antes habia que recortar
    al pixel exacto o la pieza quedaba corrida.

    La transformacion es una semejanza (giro + escala uniforme), asi que el
    dibujo nunca sale estirado ni aplastado.
    """
    h, w = rgba.shape[:2]
    if h < 2 or w < 2:
        return

    a = np.array([anchor_a[0] * w, anchor_a[1] * h], np.float32)
    b = np.array([anchor_b[0] * w, anchor_b[1] * h], np.float32)
    d_img = b - a
    len_img = float(np.linalg.norm(d_img))

    d_dst = np.asarray(p1, np.float32) - np.asarray(p0, np.float32)
    len_dst = float(np.linalg.norm(d_dst))
    if len_img < 1.0 or len_dst < 1e-3:
        return

    n_img = _perp(d_img / len_img)
    n_dst = _perp(d_dst / len_dst)

    src = np.array([a, b, a + n_img * (len_img * 0.5)], np.float32)
    dst = np.array([p0, p1, p0 + n_dst * (len_dst * 0.5 * width_gain)], np.float32)
    warp_rgba(canvas, rgba, cv2.getAffineTransform(src, dst))


def warp_centered(canvas, rgba, center, angle_deg, target_h, anchor=(0.5, 0.5)):
    """Coloca un PNG por su punto de anclaje, rotado y escalado.

    anchor es el punto de la imagen que se apoya en 'center'. Para una
    cabeza dibujada, ese punto rara vez es el centro geometrico del PNG:
    si el dibujo trae pelo o sombrero, el centro del craneo esta mas abajo.
    """
    h, w = rgba.shape[:2]
    if h < 2:
        return
    s = float(target_h) / h
    ax, ay = anchor[0] * w, anchor[1] * h
    # El eje Y de la imagen apunta hacia abajo, por eso el angulo va negado.
    M = cv2.getRotationMatrix2D((ax, ay), -angle_deg, s)
    M[0, 2] += float(center[0]) - ax
    M[1, 2] += float(center[1]) - ay
    warp_rgba(canvas, rgba, M)


# --------------------------------------------------------------------------
# Packs: tema de color + PNGs opcionales
# --------------------------------------------------------------------------

PART_NAMES = ["head", "torso", "upper_arm", "forearm", "thigh", "shin", "hand", "foot"]

SIDES = {
    "L": dict(shoulder=L_SHOULDER, elbow=L_ELBOW, wrist=L_WRIST, index=L_INDEX,
              hip=L_HIP, knee=L_KNEE, ankle=L_ANKLE, foot=L_FOOT, ear=L_EAR),
    "R": dict(shoulder=R_SHOULDER, elbow=R_ELBOW, wrist=R_WRIST, index=R_INDEX,
              hip=R_HIP, knee=R_KNEE, ankle=R_ANKLE, foot=R_FOOT, ear=R_EAR),
}


def _parse_color(value, fallback):
    """Acepta un hex tipo #RRGGBB o una lista [B, G, R]."""
    if isinstance(value, str) and value.startswith("#") and len(value) == 7:
        r, g, b = int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)
        return (b, g, r)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(v) for v in value)
    return fallback


@dataclass
class PartSpec:
    """Donde estan las articulaciones dentro del PNG de una parte.

    Para huesos (brazo, muslo...): 'a' es la articulacion de arriba y 'b'
    la de abajo, en coordenadas 0..1 de la imagen.
    Para piezas centradas (cabeza, mano): 'anchor' es el punto que se apoya
    sobre la articulacion, y 'size' multiplica el tamano.
    """

    a: tuple = (0.5, 0.0)
    b: tuple = (0.5, 1.0)
    anchor: tuple = (0.5, 0.5)
    width: float = 1.0
    size: float = 1.0

    def merged(self, cfg):
        """Aplica encima solo los campos presentes en theme.json.

        Se combina en vez de reemplazar: declarar 'size' para retocar el
        tamano no debe hacer perder las articulaciones que ya se dedujeron
        del alfa. Antes, tocar un campo descartaba todo lo demas.
        """
        if not cfg:
            return self

        def point(key, fallback):
            value = cfg.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return (float(value[0]), float(value[1]))
            return fallback

        return PartSpec(
            a=point("a", self.a),
            b=point("b", self.b),
            anchor=point("anchor", self.anchor),
            width=float(cfg.get("width", self.width)),
            size=float(cfg.get("size", self.size)),
        )


DEFAULT_SPEC = PartSpec()


def auto_spec(rgba):
    """Deduce las articulaciones de una parte mirando su canal alfa.

    El dibujo ocupa solo una parte del PNG; el resto es transparente. Esa
    zona transparente es justo el margen que descolocaba las piezas. Aqui
    se mide la caja real del dibujo y se toma:

      a = centro de la franja superior pintada  (articulacion de arriba)
      b = centro de la franja inferior pintada  (articulacion de abajo)

    Se usa una franja y no una sola fila porque la fila del borde suele
    tener cuatro pixeles sueltos del antialiasing.

    Asi un pack funciona sin escribir una sola coordenada a mano, y
    theme.json queda solo para ajustes finos.
    """
    alpha = rgba[:, :, 3]
    h, w = alpha.shape[:2]
    ys, xs = np.nonzero(alpha > 8)
    if ys.size == 0:
        return DEFAULT_SPEC

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    banda = max(1, int((y1 - y0) * 0.06))

    top_xs = xs[ys <= y0 + banda]
    bot_xs = xs[ys >= y1 - banda]
    ax = float(top_xs.mean()) if top_xs.size else (x0 + x1) * 0.5
    bx = float(bot_xs.mean()) if bot_xs.size else (x0 + x1) * 0.5

    return PartSpec(
        a=(ax / w, y0 / h),
        b=(bx / w, (y1 + 1) / h),
        anchor=((x0 + x1 + 1) * 0.5 / w, (y0 + y1 + 1) * 0.5 / h),
        # El tamano se corrige por cuanto del PNG ocupa realmente el dibujo:
        # si no, una pieza con mucho margen se veria mas chica que el resto.
        size=h / max(y1 + 1 - y0, 1),
    )


@dataclass
class AvatarPack:
    """Un avatar seleccionable: colores + imagenes opcionales por parte."""

    name: str
    theme: Theme
    parts: dict = field(default_factory=dict)
    specs: dict = field(default_factory=dict)

    @classmethod
    def builtin(cls, theme):
        return cls(name=theme.name, theme=theme, parts={})

    @classmethod
    def load(cls, folder):
        """Carga assets/packs/<nombre>/: theme.json opcional + PNGs opcionales."""
        name = os.path.basename(os.path.normpath(folder))
        theme = Theme(name=name, suit=(90, 30, 25), skin=(220, 180, 120),
                      accent=(255, 210, 60), outline=(255, 245, 210),
                      glow=(255, 190, 40), glow_strength=0.6)
        overrides = {}

        cfg_path = os.path.join(folder, "theme.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            theme.name = cfg.get("name", name)
            for key in ("suit", "skin", "accent", "outline", "glow"):
                if key in cfg:
                    setattr(theme, key, _parse_color(cfg[key], getattr(theme, key)))
            theme.glow_strength = float(cfg.get("glow_strength", theme.glow_strength))
            theme.outline_w = float(cfg.get("outline_w", theme.outline_w))
            theme.draw_face = bool(cfg.get("draw_face", theme.draw_face))
            for part_name, part_cfg in (cfg.get("parts") or {}).items():
                if isinstance(part_cfg, dict):
                    overrides[part_name.lower()] = part_cfg

        parts = {}
        for fname in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in (".png", ".webp"):
                continue
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 3 and img.shape[2] == 3:      # sin canal alfa -> opaco
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            if img.ndim == 3 and img.shape[2] == 4:
                parts[stem.lower()] = img

        # Base deducida del alfa, y encima lo que declare theme.json.
        specs = {name: auto_spec(img).merged(overrides.get(name))
                 for name, img in parts.items()}

        return cls(name=theme.name, theme=theme, parts=parts, specs=specs)

    def part(self, name, side=None):
        """Busca primero la variante del lado y si no existe cae a la base."""
        if side:
            img = self.parts.get(name + "_" + side.lower())
            if img is not None:
                return img
        return self.parts.get(name)

    def spec(self, name, side=None):
        """La configuracion de articulaciones de esa parte, o la de por defecto."""
        if side:
            found = self.specs.get(name + "_" + side.lower())
            if found is not None:
                return found
        return self.specs.get(name, DEFAULT_SPEC)


def load_packs(packs_dir):
    """Temas incluidos + cualquier carpeta que el equipo deje en assets/packs."""
    packs = [AvatarPack.builtin(t) for t in THEMES]
    if os.path.isdir(packs_dir):
        for entry in sorted(os.listdir(packs_dir)):
            folder = os.path.join(packs_dir, entry)
            if os.path.isdir(folder):
                try:
                    packs.append(AvatarPack.load(folder))
                except Exception as exc:                      # pack mal armado
                    print("[packs] no se pudo cargar " + entry + ": " + str(exc))
    return packs


# --------------------------------------------------------------------------
# Render del cuerpo completo
# --------------------------------------------------------------------------

class AvatarRenderer:
    """Dibuja el avatar sobre un lienzo RGBA transparente."""

    def __init__(self, vis_thr=0.35):
        self.vis_thr = vis_thr

    def blank(self, width, height):
        """Lienzo transparente donde dibujar uno o varios avatares."""
        return np.zeros((height, width, 4), dtype=np.uint8)

    def render(self, sk, pack, canvas=None):
        """Dibuja un avatar. Con 'canvas' se acumulan varios en el mismo.

        Acumular en un lienzo compartido es lo que permite varias personas
        sin pagar una composicion por cada una.
        """
        if canvas is None:
            canvas = self.blank(sk.width, sk.height)
        theme = pack.theme
        ow = theme.outline_w * sk.scale

        near = "L" if sk.left_is_near else "R"
        far = "R" if sk.left_is_near else "L"

        if theme.volume:
            # La sombra va primero, debajo de todo: sin ella una figura con
            # volumen igual parece flotar sobre el fondo.
            pies = [sk.pts[i] if sk.vis[i] > 0.35 else None
                    for i in (L_ANKLE, R_ANKLE, L_FOOT, R_FOOT)]
            shading.ground_shadow(canvas, pies, sk.scale)

        # Orden de atras hacia adelante para que el cuerpo tape lo correcto.
        self._leg(canvas, sk, pack, theme, ow, far)
        self._arm(canvas, sk, pack, theme, ow, far)
        self._torso(canvas, sk, pack, theme, ow)
        self._leg(canvas, sk, pack, theme, ow, near)
        self._arm(canvas, sk, pack, theme, ow, near)
        self._head(canvas, sk, pack, theme, ow)
        return canvas

    def _vis(self, sk, *idx):
        return all(float(sk.vis[i]) > self.vis_thr for i in idx)

    def _bone(self, canvas, sk, pack, theme, ow, name, a, b, fill, side):
        img = pack.part(name, side)
        p0, p1 = sk.pts[a], sk.pts[b]
        if img is not None:
            spec = pack.spec(name, side)
            warp_on_bone(canvas, img, p0, p1, spec.a, spec.b, spec.width)
            return

        w0, w1 = PROPORTIONS[name]
        w0, w1 = w0 * sk.scale, w1 * sk.scale

        # Cuando el brazo apunta a la camara, el hueso mide casi nada en 2D
        # pero su grosor seguia siendo el de un brazo entero: salia un muñon
        # gordo pegado al hombro. El grosor se limita al largo del hueso.
        length = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
        cap = max(length * FORESHORTEN_CAP, 2.0)
        w0, w1 = min(w0, cap), min(w1, cap)

        limb(canvas, p0, p1, w0, w1, _rgba(fill), _rgba(theme.outline), ow, theme)

    def _arm(self, canvas, sk, pack, theme, ow, side):
        s = SIDES[side]
        if self._vis(sk, s["shoulder"], s["elbow"]):
            self._bone(canvas, sk, pack, theme, ow, "upper_arm",
                       s["shoulder"], s["elbow"], theme.suit, side)
        if self._vis(sk, s["elbow"], s["wrist"]):
            self._bone(canvas, sk, pack, theme, ow, "forearm",
                       s["elbow"], s["wrist"], theme.accent, side)
            hand_img = pack.part("hand", side)
            if hand_img is not None:
                spec = pack.spec("hand", side)
                warp_centered(canvas, hand_img, sk.pts[s["wrist"]], sk.torso_angle + 90,
                              HAND_R * sk.scale * 2.4 * spec.size, spec.anchor)
            else:
                blob(canvas, sk.pts[s["wrist"]], HAND_R * sk.scale,
                     _rgba(theme.skin), _rgba(theme.outline), ow, theme)

    def _leg(self, canvas, sk, pack, theme, ow, side):
        s = SIDES[side]
        if self._vis(sk, s["hip"], s["knee"]):
            self._bone(canvas, sk, pack, theme, ow, "thigh",
                       s["hip"], s["knee"], theme.suit, side)
        if self._vis(sk, s["knee"], s["ankle"]):
            self._bone(canvas, sk, pack, theme, ow, "shin",
                       s["knee"], s["ankle"], theme.accent, side)
            foot_img = pack.part("foot", side)
            if self._vis(sk, s["foot"]):
                if foot_img is not None:
                    fs = pack.spec("foot", side)
                    warp_on_bone(canvas, foot_img, sk.pts[s["ankle"]], sk.pts[s["foot"]],
                                 fs.a, fs.b, fs.width)
                else:
                    limb(canvas, sk.pts[s["ankle"]], sk.pts[s["foot"]],
                         FOOT_R * sk.scale, FOOT_R * sk.scale * 0.72,
                         _rgba(theme.outline), _rgba(theme.outline), 0, theme)
            elif foot_img is None:
                blob(canvas, sk.pts[s["ankle"]], FOOT_R * sk.scale,
                     _rgba(theme.outline), _rgba(theme.outline), 0, theme)

    def _torso(self, canvas, sk, pack, theme, ow):
        # El cuello va PRIMERO, debajo del torso. Dibujandolo encima, su
        # extremo redondeado quedaba a la vista en mitad del pecho, como un
        # tubo pegado. Debajo, el torso le tapa la base y solo se ve el
        # tramo entre los hombros y la barbilla, que es lo natural.
        limb(canvas, sk.shoulder_c, sk.head_c,
             PROPORTIONS["neck"][0] * sk.scale, PROPORTIONS["neck"][1] * sk.scale,
             _rgba(theme.skin), _rgba(theme.outline), ow, theme)

        img = pack.part("torso")
        if img is not None:
            spec = pack.spec("torso")
            warp_on_bone(canvas, img, sk.shoulder_c, sk.hip_c,
                         spec.a, spec.b, spec.width)
        elif theme.volume:
            # En modo volumen el torso se trata como una capsula ancha entre
            # hombros y cadera. Un poligono plano al lado de extremidades
            # sombreadas se nota de inmediato como un cartón pegado.
            quad = sk.torso_quad()
            half_top = float(np.linalg.norm(quad[1] - quad[0])) * 0.5
            half_bot = float(np.linalg.norm(quad[2] - quad[3])) * 0.5

            # Una capsula sobresale por sus extremos el valor de su radio.
            # Puesta de hombros a cadera tal cual, la tapa de arriba subia
            # media anchura de hombros por encima y le comia el cuello. El
            # extremo superior se baja para que la cupula quede a la altura
            # de los hombros y no por encima.
            eje = sk.hip_c - sk.shoulder_c
            n = float(np.linalg.norm(eje))
            u = eje / n if n > 1e-3 else np.array([0.0, 1.0], np.float32)
            top = sk.shoulder_c + u * (half_top * 0.75)

            if ow > 0.4:
                tapered(canvas, top, sk.hip_c, half_top + ow, half_bot + ow,
                        _rgba(theme.outline))
            shading.capsule(canvas, top, sk.hip_c, half_top, half_bot,
                            theme.suit, theme.rim)
        else:
            quad = sk.torso_quad()
            center = quad.mean(axis=0)
            margin = 0.075 * sk.scale

            # Cada esquina se empuja hacia afuera: el torso no queda pegado al
            # hueso y las esquinas se ven redondeadas. Primero borde, luego relleno.
            for color, extra in ((_rgba(theme.outline), ow), (_rgba(theme.suit), 0.0)):
                pts = np.empty_like(quad)
                for i, corner in enumerate(quad):
                    d = corner - center
                    n = float(np.linalg.norm(d))
                    pts[i] = corner + (d / n) * (margin + extra) if n > 1e-3 else corner
                cv2.fillConvexPoly(canvas, pts.astype(np.int32), color, cv2.LINE_AA)
                for corner in pts:
                    cv2.circle(canvas, _pt(corner), max(int(margin), 1), color, -1, cv2.LINE_AA)


    def _head(self, canvas, sk, pack, theme, ow):
        img = pack.part("head")
        r = sk.head_r

        if img is not None:
            spec = pack.spec("head")
            warp_centered(canvas, img, sk.head_c, sk.head_angle,
                          r * 2.6 * spec.size, spec.anchor)
            # Si el pack dibuja la cabeza pero deja la cara libre, se le
            # pintan los ojos encima. Son los ojos los que hacen que la
            # gente sienta que el avatar esta vivo y la mira.
            if not theme.draw_face:
                return
        else:
            blob(canvas, sk.head_c, r, _rgba(theme.skin), _rgba(theme.outline), ow, theme)
            if not theme.draw_face:
                return

        self._face(canvas, sk, theme, r)

    def _face(self, canvas, sk, theme, r):
        """Ojos y boca del avatar: mirada viva, sin seguir la cara real."""
        a = math.radians(sk.head_angle)
        ux = np.array([math.cos(a), math.sin(a)], np.float32)     # linea de orejas
        uy = _perp(ux)                                            # hacia la barbilla
        ink = _rgba(theme.outline)

        # Cara fija: ojos abiertos y media sonrisa. Lo unico vivo es la
        # direccion de la mirada, que sale de la nariz y no cuesta nada.
        abierto_l = abierto_r = 1.0
        boca = 0.0
        sonrisa = 0.45
        ceja = 0.0

        # La pupila se corre hacia donde apunta la nariz: da sensacion de mirada.
        gaze = (sk.pts[NOSE] - sk.head_c) / max(r, 1e-3)
        gaze = np.clip(gaze, -0.55, 0.55) * r * 0.16

        eye_r = r * 0.20
        for sign, apertura in ((-1.0, abierto_r), (1.0, abierto_l)):
            eye_c = sk.head_c + ux * (sign * r * 0.36) - uy * (r * 0.10)
            alto = max(int(eye_r * apertura), 1)

            if apertura > 0.22:
                cv2.ellipse(canvas, _pt(eye_c), (max(int(eye_r), 1), alto),
                            sk.head_angle, 0, 360, (250, 250, 250, 255), -1, cv2.LINE_AA)
                pupila = max(int(eye_r * 0.52), 1)
                cv2.ellipse(canvas, _pt(eye_c + gaze),
                            (pupila, max(min(pupila, alto), 1)),
                            sk.head_angle, 0, 360, ink, -1, cv2.LINE_AA)
            else:
                # Ojo cerrado: una linea curva lee mucho mejor que una
                # elipse aplastada, que a esa altura se ve como suciedad.
                cv2.ellipse(canvas, _pt(eye_c), (max(int(eye_r), 1), max(int(eye_r * 0.5), 1)),
                            sk.head_angle, 200, 340, ink, max(int(r * 0.055), 2), cv2.LINE_AA)

            # Ceja: sube cuando la persona levanta las cejas.
            if ceja > 0.12:
                ceja_c = eye_c - uy * (r * (0.26 + 0.12 * ceja))
                cv2.ellipse(canvas, _pt(ceja_c),
                            (max(int(eye_r * 1.05), 1), max(int(eye_r * 0.45), 1)),
                            sk.head_angle, 200, 340, ink, max(int(r * 0.05), 2), cv2.LINE_AA)

        mouth_c = sk.head_c + uy * (r * 0.44)
        if boca > 0.12:
            # Boca abierta: ovalo oscuro que crece con la mandibula.
            rx = max(int(r * (0.22 + 0.08 * sonrisa)), 2)
            ry = max(int(r * (0.05 + 0.34 * boca)), 2)
            cv2.ellipse(canvas, _pt(mouth_c), (rx, ry), sk.head_angle,
                        0, 360, ink, -1, cv2.LINE_AA)
        else:
            # Boca cerrada: arco que se curva con la sonrisa.
            ry = max(int(r * (0.05 + 0.20 * sonrisa)), 1)
            cv2.ellipse(canvas, _pt(mouth_c), (max(int(r * 0.30), 1), ry),
                        sk.head_angle, 15, 165, ink,
                        max(int(r * 0.09), 2), cv2.LINE_AA)


def draw_debug_skeleton(canvas, sk, color=(0, 255, 0, 255)):
    """Huesos crudos encima del avatar, para revisar el tracking."""
    from skeleton import BONES_LEFT, BONES_RIGHT
    for a, b, _ in BONES_LEFT + BONES_RIGHT:
        cv2.line(canvas, _pt(sk.pts[a]), _pt(sk.pts[b]), color, 2, cv2.LINE_AA)
    cv2.line(canvas, _pt(sk.pts[L_SHOULDER]), _pt(sk.pts[R_SHOULDER]), color, 2, cv2.LINE_AA)
    cv2.line(canvas, _pt(sk.pts[L_HIP]), _pt(sk.pts[R_HIP]), color, 2, cv2.LINE_AA)
    for p in sk.pts:
        cv2.circle(canvas, _pt(p), 3, (0, 0, 255, 255), -1, cv2.LINE_AA)
