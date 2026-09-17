"""Del resultado crudo de MediaPipe a un esqueleto listo para dibujar.

MediaPipe entrega 33 puntos normalizados (0..1). Aqui se convierten a
pixeles del lienzo de salida y se calculan las medidas derivadas que el
render necesita: centro de hombros, centro de cadera, radio y giro de la
cabeza, escala del cuerpo y que lado esta mas cerca de la camara.
"""

import math

import numpy as np

# Indices de los 33 landmarks de MediaPipe Pose.
NOSE = 0
L_EYE, R_EYE = 2, 5
L_EAR, R_EAR = 7, 8
MOUTH_L, MOUTH_R = 9, 10
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_FOOT, R_FOOT = 31, 32

# Huesos que se dibujan como extremidades: (inicio, fin, nombre logico).
BONES_LEFT = [
    (L_SHOULDER, L_ELBOW, "upper_arm"),
    (L_ELBOW, L_WRIST, "forearm"),
    (L_HIP, L_KNEE, "thigh"),
    (L_KNEE, L_ANKLE, "shin"),
]
BONES_RIGHT = [
    (R_SHOULDER, R_ELBOW, "upper_arm"),
    (R_ELBOW, R_WRIST, "forearm"),
    (R_HIP, R_KNEE, "thigh"),
    (R_KNEE, R_ANKLE, "shin"),
]


# Proporciones humanas aproximadas, expresadas como multiplos de una
# "unidad de cuerpo" u (mas o menos el ancho de hombros de un adulto).
#
# Sirven para estimar u a partir de cualquier hueso que se vea. La clave:
# la proyeccion en 2D de un hueso SIEMPRE mide igual o menos que el hueso
# real (el escorzo acorta, nunca alarga). Asi que tomando la estimacion
# mas alta entre varios huesos se recupera el tamano verdadero aunque la
# persona este de perfil o agachada.
SEGMENT_RATIOS = [
    ((11, 12), 1.00),   # hombros
    ((23, 24), 0.62),   # cadera
    ((11, 13), 0.82), ((12, 14), 0.82),   # brazos
    ((13, 15), 0.74), ((14, 16), 0.74),   # antebrazos
    ((23, 25), 1.15), ((24, 26), 1.15),   # muslos
    ((25, 27), 1.10), ((26, 28), 1.10),   # pantorrillas
    ((11, 23), 1.35), ((12, 24), 1.35),   # costados del torso
]


class BodyState:
    """Memoria entre cuadros: tamano del cuerpo y que lado esta al frente."""

    def __init__(self):
        from smoothing import ScaleTracker
        self.unit = ScaleTracker()
        self.left_near = True
        self.vis = None

    def reset(self):
        self.unit.reset()
        self.vis = None

    def smooth_visibility(self, vis, alpha=0.35):
        """Promedia la visibilidad de cada punto con la de los cuadros previos.

        De cuerpo completo los tobillos quedan al borde del cuadro y su
        visibilidad oscila alrededor del umbral: las piernas aparecian y
        desaparecian cuadro a cuadro. Promediando, una pierna tarda unos
        cuadros en irse y el parpadeo desaparece.
        """
        if self.vis is None or self.vis.shape != vis.shape:
            self.vis = vis.copy()
        else:
            self.vis += alpha * (vis - self.vis)
        return self.vis


def _mid(a, b):
    return (a + b) * 0.5


class Skeleton:
    """Esqueleto en pixeles con todo lo derivado ya calculado."""

    def __init__(self, pts, vis, z, width, height, state=None):
        self.pts = pts              # (33, 2) en pixeles
        self.vis = vis              # (33,) visibilidad 0..1
        self.z = z                  # (33,) profundidad relativa
        self.width = width
        self.height = height

        self.shoulder_c = _mid(pts[L_SHOULDER], pts[R_SHOULDER])
        self.hip_c = _mid(pts[L_HIP], pts[R_HIP])

        self.shoulder_w = float(np.linalg.norm(pts[L_SHOULDER] - pts[R_SHOULDER]))
        self.torso_len = float(np.linalg.norm(self.shoulder_c - self.hip_c))

        # Tamano del cuerpo. Es lo que fija el grosor de todas las partes,
        # asi que tiene que depender de la DISTANCIA a la camara y de nada
        # mas. Medirlo del ancho de hombros del cuadro lo ataba a la postura:
        # de perfil los hombros se juntan y el avatar se volvia un palo.
        measured = self._measure_unit()
        self.scale = state.unit.update(measured) if state else measured

        self.head_c, self.head_r, self.head_angle = self._head()

        # Angulo del torso: de cadera hacia hombros.
        d = self.shoulder_c - self.hip_c
        self.torso_angle = math.degrees(math.atan2(d[1], d[0]))

        self.left_is_near = self._depth_order(state)

    def _measure_unit(self):
        """Estima la unidad de cuerpo a partir de todos los huesos visibles."""
        est = []
        for (a, b), ratio in SEGMENT_RATIOS:
            if self.vis[a] > 0.5 and self.vis[b] > 0.5:
                length = float(np.linalg.norm(self.pts[a] - self.pts[b]))
                if length > 1.0:
                    est.append(length / ratio)

        if not est:
            # Nada confiable a la vista: se cae al metodo viejo.
            return max(self.shoulder_w, self.torso_len * 0.75, 1e-3)

        # Percentil alto y no el maximo: el escorzo hace que los huesos
        # subestimen, pero un landmark mal puesto puede sobreestimar, y ese
        # caso queda descartado arriba del percentil.
        return float(np.percentile(est, 80))

    def _depth_order(self, state):
        """Que lado del cuerpo esta mas cerca de la camara (z menor)."""
        diff = float(self.z[R_SHOULDER] + self.z[R_HIP]) - float(self.z[L_SHOULDER] + self.z[L_HIP])
        if state is None:
            return diff > 0.0
        # Banda muerta: cuando la persona esta de frente los dos lados estan
        # casi a la misma profundidad y el orden de dibujo parpadeaba.
        if abs(diff) > 0.06:
            state.left_near = diff > 0.0
        return state.left_near

    def _head(self):
        pts, vis = self.pts, self.vis
        nose = pts[NOSE]

        # El radio sale de la unidad de cuerpo ya estabilizada, no de medidas
        # del cuadro. La distancia entre orejas se encoge al girar la cabeza
        # y antes eso hacia que la cabeza latiera al mirar de lado.
        radius = float(np.clip(0.36 * self.scale, 8.0, self.height * 0.45))

        # Centro: la nariz esta al frente de la cara, el centro del craneo
        # queda desplazado hacia atras/arriba respecto de ella.
        ear_c = _mid(pts[L_EAR], pts[R_EAR]) if (vis[L_EAR] > 0.4 and vis[R_EAR] > 0.4) else nose
        center = _mid(nose, ear_c)
        up = center - self.shoulder_c
        n = float(np.linalg.norm(up))
        if n > 1e-3:
            center = center + (up / n) * (radius * 0.22)

        # Giro: la linea entre orejas u ojos define la inclinacion.
        if vis[L_EAR] > 0.4 and vis[R_EAR] > 0.4:
            d = pts[L_EAR] - pts[R_EAR]
        else:
            d = pts[L_EYE] - pts[R_EYE]
        angle = math.degrees(math.atan2(d[1], d[0])) if float(np.linalg.norm(d)) > 1e-3 else 0.0
        return center, radius, angle

    def visible(self, idx, thr=0.35):
        return float(self.vis[idx]) > thr

    def torso_quad(self):
        """Las cuatro esquinas del torso, con ancho minimo garantizado.

        De perfil el ancho de hombros en 2D cae casi a cero y el torso se
        volvia una linea. Aqui se conserva la direccion real de hombros y
        cadera (para que el giro del tronco se siga viendo) pero se le
        impone un grosor minimo proporcional al tamano del cuerpo.
        """
        axis = self.shoulder_c - self.hip_c
        n = float(np.linalg.norm(axis))
        fallback = np.array([-axis[1], axis[0]], np.float32) / n if n > 1e-3 \
            else np.array([1.0, 0.0], np.float32)

        def edge(a, b, center, min_half):
            d = self.pts[a] - self.pts[b]
            length = float(np.linalg.norm(d))
            direction = d / length if length > 1e-3 else fallback
            half = max(length * 0.5, min_half)
            return center + direction * half, center - direction * half

        top_l, top_r = edge(L_SHOULDER, R_SHOULDER, self.shoulder_c, 0.30 * self.scale)
        bot_l, bot_r = edge(L_HIP, R_HIP, self.hip_c, 0.24 * self.scale)
        return np.array([top_r, top_l, bot_l, bot_r], np.float32)

    def bbox(self, margin_ratio=0.6):
        """Caja que encierra al avatar, con margen para grosor, borde y halo.

        El compositor la usa para mezclar solo donde hay algo que mezclar.
        El margen es generoso a proposito: recortar de mas se ve como un
        avatar con los bordes cortados, y ahorrar esos pixeles no vale eso.
        """
        usable = self.pts[self.vis > 0.3]
        if usable.shape[0] == 0:
            usable = self.pts
        m = self.scale * margin_ratio
        x0 = float(np.min(usable[:, 0])) - m
        y0 = float(np.min(usable[:, 1])) - m
        x1 = float(np.max(usable[:, 0])) + m
        y1 = float(np.max(usable[:, 1])) + m
        return (max(int(x0), 0), max(int(y0), 0),
                min(int(x1) + 1, self.width), min(int(y1) + 1, self.height))

    def bone(self, a, b):
        return self.pts[a], self.pts[b]


def from_landmarks(landmarks, width, height, smoother=None, fps=None, state=None):
    """Convierte los landmarks normalizados de MediaPipe en un Skeleton.

    landmarks: lista de 33 NormalizedLandmark del PoseLandmarkerResult.
    state: BodyState opcional, la memoria entre cuadros del tamano del
           cuerpo y del orden de profundidad.
    """
    n = len(landmarks)
    pts = np.empty((n, 2), dtype=np.float32)
    vis = np.empty(n, dtype=np.float32)
    z = np.empty(n, dtype=np.float32)

    for i, lm in enumerate(landmarks):
        pts[i, 0] = lm.x * width
        pts[i, 1] = lm.y * height
        z[i] = lm.z
        # En algunas versiones visibility puede venir en None.
        v = getattr(lm, "visibility", None)
        vis[i] = 1.0 if v is None else float(v)

    if smoother is not None:
        pts = smoother(pts, freq=fps)
    if state is not None:
        vis = state.smooth_visibility(vis)

    return Skeleton(pts, vis, z, width, height, state=state)
