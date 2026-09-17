"""Seguimiento de la cara: ojos, boca y cejas.

MediaPipe FaceLandmarker entrega 52 "blendshapes": valores de 0 a 1 que
dicen cuanto esta activada cada expresion (un ojo cerrado, la mandibula
abierta, una sonrisa). Aqui se quedan solo los cinco que se van a dibujar
y se convierten a algo directo de usar: cuanto esta ABIERTO cada ojo.

Corre en su propio hilo (modo LIVE_STREAM), igual que el detector de pose,
asi que no frena el bucle principal.
"""

from dataclasses import dataclass

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Los blendshapes vienen nombrados desde el punto de vista de la persona.
# Como el cuadro se voltea antes de detectar (modo espejo), el ojo que la
# persona ve a su izquierda en pantalla es el que hay que mover. Si en la
# prueba en vivo los guiños salen cambiados de lado, invertir esto.
SWAP_EYES = False


def head_crop(frame, head_c, head_r, size=256, pad=2.3):
    """Recorta un cuadrado alrededor de la cabeza y lo lleva a 'size' px.

    Es la diferencia entre que el seguimiento facial funcione o no. Al
    detector se le daba el cuadro entero reducido a 480 px de ancho: a dos
    metros de distancia la cara ocupaba ahi unos 40 px, muy poco para sacar
    parpados y boca. El detector de cuerpo aguanta esa resolucion, el de
    cara no.

    Recortando la cabeza y escalandola a 256 px, el detector recibe la cara
    grande sin importar a que distancia este la persona.

    Como de la cara solo se usan los blendshapes (valores de expresion) y
    no las coordenadas, no hace falta mapear nada de vuelta al cuadro.
    """
    h, w = frame.shape[:2]
    half = max(head_r * pad, 24.0)
    cx, cy = float(head_c[0]), float(head_c[1])

    # El lado se calcula una sola vez y los dos extremos salen de el. Si se
    # redondean X e Y por separado, el recorte puede quedar de 388x371 en
    # vez de cuadrado y el pegado al lienzo revienta.
    lado = max(int(round(half * 2.0)), 8)
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = x0 + lado, y0 + lado

    # Si la cabeza queda parcialmente fuera del cuadro, se rellena en negro
    # en vez de desplazar el recorte: mover el centro deformaria la cara.
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, w), min(y1, h)
    if sx1 - sx0 < 8 or sy1 - sy0 < 8:
        return None

    lienzo = np.zeros((lado, lado, 3), dtype=frame.dtype)
    lienzo[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]

    interp = cv2.INTER_AREA if lado > size else cv2.INTER_LINEAR
    return cv2.resize(lienzo, (size, size), interpolation=interp)


@dataclass
class Expression:
    """Estado de la cara, ya listo para dibujar. 0 = nada, 1 = al maximo."""

    eye_left: float = 1.0      # 1 = ojo abierto, 0 = cerrado
    eye_right: float = 1.0
    mouth_open: float = 0.0
    smile: float = 0.0
    brow: float = 0.0
    valid: bool = False

    @staticmethod
    def neutral():
        return Expression()


class FaceTracker:
    """Detector de cara asincrono que mantiene la ultima expresion vista."""

    def __init__(self, model_path, num_faces=1):
        self._expr = Expression.neutral()
        self._raw = {}

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            result_callback=self._on_result,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    # -- entrada ---------------------------------------------------------

    def submit(self, mp_image, timestamp_ms):
        self.landmarker.detect_async(mp_image, timestamp_ms)

    # -- salida ----------------------------------------------------------

    @property
    def expression(self):
        return self._expr

    @property
    def raw(self):
        """Los 52 valores crudos, para el modo de depuracion."""
        return self._raw

    def reset(self):
        self._expr = Expression.neutral()
        self._raw = {}

    # -- interno ---------------------------------------------------------

    def _on_result(self, result, output_image, timestamp_ms):
        if not result.face_blendshapes:
            self._expr = Expression.neutral()
            return

        bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
        self._raw = bs

        # 'blink' vale 1 cuando el ojo esta CERRADO; se invierte porque
        # dibujar es mas natural en terminos de apertura.
        left = 1.0 - bs.get("eyeBlinkLeft", 0.0)
        right = 1.0 - bs.get("eyeBlinkRight", 0.0)
        if SWAP_EYES:
            left, right = right, left

        smile = 0.5 * (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0))
        brow = max(bs.get("browInnerUp", 0.0),
                   0.5 * (bs.get("browOuterUpLeft", 0.0) + bs.get("browOuterUpRight", 0.0)))

        nuevo = Expression(
            eye_left=_clamp(left),
            eye_right=_clamp(right),
            mouth_open=_clamp(bs.get("jawOpen", 0.0)),
            smile=_clamp(smile),
            brow=_clamp(brow),
            valid=True,
        )

        # Suavizado corto. Un parpadeo dura unos 100 ms, o sea tres cuadros
        # a 30 FPS: filtrar mas fuerte se los comeria y el efecto se pierde.
        previo = self._expr
        if previo.valid:
            nuevo = Expression(
                eye_left=_mix(previo.eye_left, nuevo.eye_left, 0.65),
                eye_right=_mix(previo.eye_right, nuevo.eye_right, 0.65),
                mouth_open=_mix(previo.mouth_open, nuevo.mouth_open, 0.55),
                smile=_mix(previo.smile, nuevo.smile, 0.35),
                brow=_mix(previo.brow, nuevo.brow, 0.35),
                valid=True,
            )
        self._expr = nuevo

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


def _clamp(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))


def _mix(a, b, t):
    return a + (b - a) * t
